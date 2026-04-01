#!/usr/bin/env python
"""
Point-cloud autoencoder pipeline for 6-channel input point clouds.

Input convention (required):
- Shape: (N, P, 6)
- Channels: [xyz(3), normals/features(3)]

Exports:
- Encoded latent grids with shape (N, latent_size, latent_size)
  suitable for DDPM pipeline consumption.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# -----------------------------
# Reproducibility / runtime
# -----------------------------
def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model blocks
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class FeatureFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz_feats: torch.Tensor, normal_feats: torch.Tensor) -> torch.Tensor:
        # xyz_feats: (B, P, C1), normal_feats: (B, P, C2)
        x = torch.cat([xyz_feats, normal_feats], dim=-1)
        return self.net(x)


class ResidualLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        x = F.relu(x + identity, inplace=True)
        return x


class PointCloudAE(nn.Module):
    def __init__(self, point_size: int = 1000, latent_size: int = 32):
        super().__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        self.latent_dim = latent_size * latent_size

        # Separate XYZ and normal/feature branches
        self.xyz_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        self.norm_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        self.fusion = FeatureFusion(in_dim=512, out_dim=512)
        self.attn = SelfAttention(dim=512)

        # Point-wise to global latent vector
        self.encoder_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(512, self.latent_dim)

        # Decoder: latent -> point_size x 6 with residual linear layers
        self.decoder_in = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
        )
        self.decoder_res1 = ResidualLinear(1024)
        self.decoder_res2 = ResidualLinear(1024)
        self.decoder_out = nn.Linear(1024, point_size * 6)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, 6)
        xyz = x[..., :3]
        nrm = x[..., 3:]

        xyz_feats = self.xyz_branch(xyz)
        nrm_feats = self.norm_branch(nrm)

        fused = self.fusion(xyz_feats, nrm_feats)
        attended = self.attn(fused)

        h = self.encoder_head(attended)
        global_feat, _ = torch.max(h, dim=1)  # (B, 512)
        latent_vec = self.to_latent(global_feat)  # (B, latent_dim)
        return latent_vec.view(-1, self.latent_size, self.latent_size)

    def decode(self, latent_grid: torch.Tensor) -> torch.Tensor:
        # latent_grid: (B, latent_size, latent_size)
        z = latent_grid.view(latent_grid.size(0), -1)
        h = self.decoder_in(z)
        h = self.decoder_res1(h)
        h = self.decoder_res2(h)
        out = self.decoder_out(h)
        return out.view(-1, self.point_size, 6)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_grid = self.encode(x)
        recon = self.decode(latent_grid)
        return recon, latent_grid


# -----------------------------
# Data / checkpointing
# -----------------------------
def load_point_clouds(path: str) -> np.ndarray:
    pcs = np.load(path)
    if pcs.ndim != 3 or pcs.shape[-1] != 6:
        raise ValueError(f"Expected (N, P, 6), got {pcs.shape}")
    return pcs.astype(np.float32)


@dataclass
class TrainConfig:
    pointcloud_path: str = "data/normalized_rotated_point_clouds6.npy"
    save_dir: str = "."
    ckpt_dir: str = "./checkpoints_autoencoder"
    out_encoded: str = "encoded_features_CURVATURE.npy"
    latent_size: int = 32
    batch_size: int = 16
    epochs: int = 300
    patience: int = 30
    lr: float = 1e-4
    val_split: float = 0.1
    seed: int = 42
    encode_batch_size: int = 64


def _meta_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "ae_meta.json")


def _ckpt_paths(ckpt_dir: str) -> Dict[str, str]:
    return {
        "latest": os.path.join(ckpt_dir, "ae_latest.pth"),
        "best": os.path.join(ckpt_dir, "ae_best.pth"),
        "final": os.path.join(ckpt_dir, "ae_final.pth"),
    }


def save_meta(cfg: TrainConfig, point_size: int, train_size: int, val_size: int, best_val: float, epochs_done: int) -> None:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "point_size": point_size,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_loss": best_val,
        "epochs_done": epochs_done,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    with open(_meta_path(cfg.ckpt_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_dataloaders(data: np.ndarray, batch_size: int, val_split: float, seed: int) -> Tuple[DataLoader, DataLoader, int, int]:
    tensor = torch.from_numpy(data)
    ds = TensorDataset(tensor)

    n = len(ds)
    val_size = int(round(n * val_split))
    val_size = max(1, min(val_size, n - 1))
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, train_size, val_size


def build_model(point_size: int, latent_size: int, device: torch.device) -> PointCloudAE:
    model = PointCloudAE(point_size=point_size, latent_size=latent_size)
    return model.to(device)


def _load_resume_if_available(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: str, device: torch.device):
    latest_path = _ckpt_paths(ckpt_dir)["latest"]
    if not os.path.exists(latest_path):
        return 0, float("inf")

    state = torch.load(latest_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_val = float(state.get("best_val_loss", float("inf")))
    print(f"[resume] Loaded {latest_path} (start_epoch={start_epoch}, best_val={best_val:.6f})")
    return start_epoch, best_val


def _save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_loss: float, cfg: TrainConfig):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": asdict(cfg),
        },
        path,
    )


# -----------------------------
# Train / encode
# -----------------------------
def train_autoencoder(cfg: TrainConfig):
    set_deterministic(cfg.seed)
    device = get_device()

    data = load_point_clouds(cfg.pointcloud_path)
    point_size = data.shape[1]

    train_loader, val_loader, train_size, val_size = create_dataloaders(
        data=data,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    model = build_model(point_size=point_size, latent_size=cfg.latent_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpts = _ckpt_paths(cfg.ckpt_dir)

    start_epoch, best_val_loss = _load_resume_if_available(model, optimizer, cfg.ckpt_dir, device)
    epochs_without_improvement = 0

    print(f"[train] device={device} | samples={len(data)} | point_size={point_size} | latent={cfg.latent_size}x{cfg.latent_size}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_losses = []

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon, _ = model(batch)
                loss = F.mse_loss(recon, batch)
                val_losses.append(float(loss.item()))

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        improved = mean_val < best_val_loss
        if improved:
            best_val_loss = mean_val
            epochs_without_improvement = 0
            _save_ckpt(ckpts["best"], model, optimizer, epoch, best_val_loss, cfg)
        else:
            epochs_without_improvement += 1

        _save_ckpt(ckpts["latest"], model, optimizer, epoch, best_val_loss, cfg)

        print(
            f"[epoch {epoch + 1:03d}/{cfg.epochs}] "
            f"train_loss={mean_train:.6f} val_loss={mean_val:.6f} "
            f"best={best_val_loss:.6f} {'*' if improved else ''}"
        )

        if epochs_without_improvement >= cfg.patience:
            print(f"[early-stop] no improvement for {cfg.patience} epochs")
            break

    final_epoch = epoch if 'epoch' in locals() else start_epoch
    _save_ckpt(ckpts["final"], model, optimizer, final_epoch, best_val_loss, cfg)
    save_meta(cfg, point_size=point_size, train_size=train_size, val_size=val_size, best_val=best_val_loss, epochs_done=final_epoch + 1)
    print(f"[train] checkpoints saved to {cfg.ckpt_dir}")


def _load_model_for_encode(cfg: TrainConfig, point_size: int, device: torch.device) -> PointCloudAE:
    model = build_model(point_size=point_size, latent_size=cfg.latent_size, device=device)
    ckpts = _ckpt_paths(cfg.ckpt_dir)

    load_path = ckpts["best"] if os.path.exists(ckpts["best"]) else ckpts["latest"]
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.ckpt_dir}. Expected one of: {ckpts['best']} or {ckpts['latest']}"
        )

    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[encode] loaded checkpoint: {load_path}")
    return model


def encode_dataset(cfg: TrainConfig):
    set_deterministic(cfg.seed)
    device = get_device()

    data = load_point_clouds(cfg.pointcloud_path)
    point_size = data.shape[1]

    model = _load_model_for_encode(cfg, point_size=point_size, device=device)

    ds = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(ds, batch_size=cfg.encode_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    encoded_chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            latent = model.encode(batch)
            encoded_chunks.append(latent.cpu().numpy())

    encoded = np.concatenate(encoded_chunks, axis=0).astype(np.float32)
    if encoded.ndim != 3 or encoded.shape[1:] != (cfg.latent_size, cfg.latent_size):
        raise RuntimeError(f"Encoded shape mismatch. Expected (N, {cfg.latent_size}, {cfg.latent_size}), got {encoded.shape}")

    out_path = cfg.out_encoded if os.path.isabs(cfg.out_encoded) else os.path.join(cfg.save_dir, cfg.out_encoded)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, encoded)
    print(f"[encode] saved {out_path} with shape {encoded.shape}")


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="6-channel point cloud autoencoder pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser):
        p.add_argument("--pointcloud_path", type=str, default="data/normalized_rotated_point_clouds6.npy")
        p.add_argument("--save_dir", type=str, default=".")
        p.add_argument("--ckpt_dir", type=str, default="./checkpoints_autoencoder")
        p.add_argument("--out_encoded", type=str, default="encoded_features_CURVATURE.npy")
        p.add_argument("--latent_size", type=int, default=32)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--epochs", type=int, default=300)
        p.add_argument("--patience", type=int, default=30)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--val_split", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--encode_batch_size", type=int, default=64)

    p_train = sub.add_parser("train", help="train AE")
    add_common_args(p_train)

    p_encode = sub.add_parser("encode", help="encode dataset using trained AE")
    add_common_args(p_encode)

    p_both = sub.add_parser("train_and_encode", help="train AE then encode dataset")
    add_common_args(p_both)

    return parser


def args_to_cfg(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        pointcloud_path=args.pointcloud_path,
        save_dir=args.save_dir,
        ckpt_dir=args.ckpt_dir,
        out_encoded=args.out_encoded,
        latent_size=args.latent_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        encode_batch_size=args.encode_batch_size,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_cfg(args)

    if args.command == "train":
        train_autoencoder(cfg)
    elif args.command == "encode":
        encode_dataset(cfg)
    elif args.command == "train_and_encode":
        train_autoencoder(cfg)
        encode_dataset(cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
