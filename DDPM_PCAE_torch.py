#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange

'''
python3 DDPM_PCAE_torch.py train_ae \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae \
  --latent_size 32 \
  --ae_batch_size 16 \
  --ae_epochs 300 \
  --patience 30 \
  --ae_lr 1e-4 \
  --max_train_samples 1000 \
  --max_val_samples 1000 \
  --seed 42 \
  --verbose

# If max_train_samples / max_val_samples are omitted, val_split is used.
python3 DDPM_PCAE_torch.py train_ae \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae \
  --latent_size 32 \
  --ae_batch_size 16 \
  --ae_epochs 300 \
  --patience 30 \
  --ae_lr 1e-4 \
  --val_split 0.1 \
  --seed 42 \
  --verbose

python3 DDPM_PCAE_torch.py encode_dataset \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae \
  --encoded_features encoded_features.npy \
  --encode_batch_size 64 \
  --seed 42 \
  --verbose

python3 DDPM_PCAE_torch.py train_ddpm \
  --encoded_features encoded_features.npy \
  --obb_vectors obb_vectors_open3d_euler.npy \
  --scale_coeffs normalized_scale_coefficients6.npy \
  --curvatures average_mean_curvatures.npy \
  --ddpm_ckpt_dir checkpoints/ddpm \
  --batch_size 64 \
  --epochs 300 \
  --patience 30 \
  --lr 1e-4 \
  --max_train_samples 1000 \
  --max_val_samples 1000 \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.025 \
  --seed 42 \
  --verbose

# If max_train_samples / max_val_samples are omitted, val_split is used.
python3 DDPM_PCAE_torch.py train_ddpm \
  --encoded_features encoded_features.npy \
  --obb_vectors obb_vectors_open3d_euler.npy \
  --scale_coeffs normalized_scale_coefficients6.npy \
  --curvatures average_mean_curvatures.npy \
  --ddpm_ckpt_dir checkpoints/ddpm \
  --batch_size 64 \
  --epochs 300 \
  --patience 30 \
  --lr 1e-4 \
  --val_split 0.2 \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.025 \
  --seed 42 \
  --verbose
'''
# -----------------------------
# Config / constants
# -----------------------------
DEFAULT_ENCODED_FEATURES = "encoded_features.npy"
DEFAULT_OBB = "obb_vectors_open3d_euler.npy"
DEFAULT_SCALE = "normalized_scale_coefficients6.npy"
DEFAULT_CURV = "average_mean_curvatures.npy"
DEFAULT_POINTCLOUDS = "data/normalized_rotated_point_clouds6.npy"
DEFAULT_SUBGRAPHS = "UNPERTURBED_center_subgraphs450_8.npz"

DEFAULT_DDPM_CKPT_DIR = "checkpoints/ddpm"
DEFAULT_AE_CKPT_DIR = "checkpoints/ae"
DEFAULT_LATENT_NODE_DIR = "outputs/generated_latents"
DEFAULT_LATENT_GRAPH_DIR = "outputs/generated_graph_npz"
DEFAULT_DECODED_DIR = "outputs/decoded_pointclouds"
DEFAULT_ASSEMBLY_DIR = "outputs/assemblies"

TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.025


# -----------------------------
# Utility functions
# -----------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# DDPM dataset preprocessing
# -----------------------------
@dataclass
class LabelNormStats:
    min_dim: np.ndarray
    dim_den: np.ndarray
    curv_min: float
    curv_den: float
    scale_min: float
    scale_den: float


@dataclass
class LatentNormStats:
    latent_max: float


def preprocess_obb_np(obb_vectors: np.ndarray, min_dim: np.ndarray, dim_den: np.ndarray) -> np.ndarray:
    angles = obb_vectors[:, :3]
    dims = (obb_vectors[:, 3:] - min_dim[None, :]) / dim_den[None, :]
    return np.concatenate([angles, dims], axis=1).astype(np.float32)


def normalize_scalar_array(a: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mn = float(np.min(a))
    mx = float(np.max(a))
    den = max(mx - mn, 1e-12)
    return ((a - mn) / den).astype(np.float32), mn, den


def build_ddpm_arrays(
    encoded_path: str,
    obb_path: str,
    scale_path: str,
    curv_path: str,
) -> Tuple[np.ndarray, np.ndarray, LabelNormStats, LatentNormStats]:
    ensure_file(encoded_path)
    ensure_file(obb_path)
    ensure_file(scale_path)
    ensure_file(curv_path)

    encoded = np.load(encoded_path).astype(np.float32)
    obb = np.load(obb_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)
    curv = np.load(curv_path).astype(np.float32)

    n = min(len(encoded), len(obb), len(scale), len(curv))
    encoded = encoded[:n]
    obb = obb[:n]
    scale = scale[:n]
    curv = curv[:n]

    if encoded.ndim == 3:
        encoded = encoded[..., None]
    if encoded.ndim != 4:
        raise ValueError(f"encoded_features must be 3D or 4D, got {encoded.shape}")
    if encoded.shape[-1] != 1:
        raise ValueError(f"Expected encoded channel=1, got {encoded.shape}")
    if obb.shape[1] < 6:
        raise ValueError(f"Expected OBB shape (N, >=6), got {obb.shape}")

    min_dim = obb[:, 3:6].min(axis=0)
    max_dim = obb[:, 3:6].max(axis=0)
    dim_den = np.maximum(max_dim - min_dim, 1e-12)

    obb_lbl = preprocess_obb_np(obb[:, :6], min_dim, dim_den)
    curv_norm, curv_min, curv_den = normalize_scalar_array(curv)
    scale_norm, scale_min, scale_den = normalize_scalar_array(scale)

    labels = np.hstack([obb_lbl, curv_norm.reshape(-1, 1), scale_norm.reshape(-1, 1)]).astype(np.float32)

    latent_max = float(np.max(np.abs(encoded)))
    latent_max = max(latent_max, 1e-6)

    lbl_stats = LabelNormStats(
        min_dim=min_dim.astype(np.float32),
        dim_den=dim_den.astype(np.float32),
        curv_min=curv_min,
        curv_den=curv_den,
        scale_min=scale_min,
        scale_den=scale_den,
    )
    lat_stats = LatentNormStats(latent_max=latent_max)
    return encoded.astype(np.float32), labels, lbl_stats, lat_stats


def preprocess_latent_torch(x: torch.Tensor, latent_max: float) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    return torch.clamp(x / latent_max, -1.0, 1.0)


def postprocess_latent_np(x: np.ndarray, latent_max: float) -> np.ndarray:
    return (x * latent_max).astype(np.float32)


# -----------------------------
# DDPM model definitions
# -----------------------------
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(tnn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, x):
        half_dim = self.dim // 2
        inv_freq_scale = math.log(self.max_positions) / max(half_dim - 1, 1)
        positions = torch.arange(half_dim, device=x.device, dtype=x.dtype)
        emb = torch.exp(positions * -inv_freq_scale)
        emb = x[:, None].to(emb.dtype) * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Identity(tnn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class Residual(tnn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return tnn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


def Downsample(dim):
    return tnn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)


class LayerNorm(tnn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = tnn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = tnn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.g + self.b


class PreNorm(tnn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class SiLU(tnn.Module):
    def forward(self, x):
        return F.silu(x)


class GELU(tnn.Module):
    def forward(self, x):
        return F.gelu(x)


class Block(tnn.Module):
    def __init__(self, dim, dim_out=None, groups=8):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = tnn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = tnn.GroupNorm(num_groups=min(groups, dim_out), num_channels=dim_out, eps=1e-5)
        self.act = SiLU()

    def forward(self, x, gamma_beta=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1.0) + beta
        return self.act(x)


class ResnetBlock(tnn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = tnn.Sequential(SiLU(), tnn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = tnn.Conv2d(dim, dim_out, kernel_size=1, stride=1) if dim != dim_out else Identity()

    def forward(self, x, time_emb=None):
        gamma_beta = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            gamma_beta = torch.chunk(time_emb, chunks=2, dim=1)
        h = self.block1(x, gamma_beta=gamma_beta)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(tnn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = tnn.Conv2d(dim, self.hidden_dim * 3, kernel_size=1, stride=1, bias=False)
        self.to_out = tnn.Sequential(tnn.Conv2d(self.hidden_dim, dim, kernel_size=1, stride=1), LayerNorm(dim))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        qkv = torch.chunk(self.to_qkv(x), chunks=3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (hh c) x y -> b hh c (x y)", hh=self.heads), qkv)
        q = torch.softmax(q, dim=-2) * self.scale
        k = torch.softmax(k, dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b hh c (x y) -> b (hh c) x y", hh=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(tnn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = tnn.Conv2d(dim, self.hidden_dim * 3, kernel_size=1, stride=1, bias=False)
        self.to_out = tnn.Conv2d(self.hidden_dim, dim, kernel_size=1, stride=1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        qkv = torch.chunk(self.to_qkv(x), chunks=3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (hh c) x y -> b hh c (x y)", hh=self.heads), qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = torch.softmax(sim, dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b hh (x y) d -> b (hh d) x y", hh=self.heads, x=h, y=w)
        return self.to_out(out)


class MLP(tnn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = tnn.Sequential(
            tnn.Linear(1, hidden_dim),
            GELU(),
            tnn.LayerNorm(hidden_dim),
            tnn.Linear(hidden_dim, hidden_dim),
            GELU(),
            tnn.LayerNorm(hidden_dim),
            tnn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x)


class UnetConditional(tnn.Module):
    def __init__(
        self,
        dim=64,
        init_dim=64,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        resnet_block_groups=4,
        learned_variance=False,
        sinusoidal_cond_mlp=True,
        class_emb_dim_obb=256,
        class_emb_dim_curvature=64,
        class_emb_dim_scale=64,
        obb_length=6,
        curvature_length=1,
        scale_length=1,
        in_res=32,
    ):
        super().__init__()
        self.channels = channels
        self.in_res = in_res
        self.obb_length = obb_length
        self.curvature_length = curvature_length
        self.scale_length = scale_length

        init_dim = init_dim if init_dim is not None else dim // 2
        self.init_conv = tnn.Conv2d(channels, init_dim, kernel_size=7, stride=1, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = tnn.Sequential(
            SinusoidalPosEmb(dim),
            tnn.Linear(dim, time_dim),
            tnn.GELU(),
            tnn.Linear(time_dim, time_dim),
        ) if sinusoidal_cond_mlp else MLP(time_dim)

        self.obb_embedding = tnn.Linear(obb_length, class_emb_dim_obb)
        self.curvature_embedding = tnn.Linear(curvature_length, class_emb_dim_curvature)
        self.scale_embedding = tnn.Linear(scale_length, class_emb_dim_scale)

        block_klass = lambda din, dout, tdim: ResnetBlock(din, dout, time_emb_dim=tdim, groups=resnet_block_groups)

        self.downs = tnn.ModuleList()
        self.ups = tnn.ModuleList()
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            self.downs.append(tnn.ModuleList([
                block_klass(dim_in, dim_out, time_dim),
                block_klass(dim_out, dim_out, time_dim),
                block_klass(dim_out, dim_out, time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind == (num_resolutions - 1)
            self.ups.append(tnn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_dim),
                block_klass(dim_in, dim_in, time_dim),
                block_klass(dim_in, dim_in, time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity(),
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, lambda: default_out_dim)
        self.final_conv = tnn.Sequential(block_klass(dim * 2, dim, time_dim), tnn.Conv2d(dim, self.out_dim, kernel_size=1, stride=1))

    def forward(self, x, time=None, class_value=None):
        x = self.init_conv(x)
        t = self.time_mlp(time.to(torch.float32))
        obb, curvature, scale = torch.split(class_value, [self.obb_length, self.curvature_length, self.scale_length], dim=-1)
        class_emb = torch.cat([
            self.obb_embedding(obb),
            self.curvature_embedding(curvature),
            self.scale_embedding(scale),
        ], dim=-1)
        t = torch.cat([t, class_emb], dim=-1)

        h = []
        for block1, block2, block3, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = block3(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, block3, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = block3(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# -----------------------------
# DDPM train / eval / sample
# -----------------------------
class DDPMTrainer:
    def __init__(self, model: UnetConditional, timesteps: int = TIMESTEPS, beta_start: float = BETA_START, beta_end: float = BETA_END, device: Optional[torch.device] = None):
        self.model = model
        self.timesteps = timesteps
        self.device = device or next(model.parameters()).device
        beta = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha, axis=0)

        self.alpha_t = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.alpha_bar_t = torch.tensor(alpha_bar, dtype=torch.float32, device=self.device)
        self.beta_t = torch.tensor(beta, dtype=torch.float32, device=self.device)
        self.sqrt_alpha_bar = torch.tensor(np.sqrt(alpha_bar), dtype=torch.float32, device=self.device)
        self.sqrt_one_minus_alpha_bar = torch.tensor(np.sqrt(1.0 - alpha_bar), dtype=torch.float32, device=self.device)

    def forward_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sa = self.sqrt_alpha_bar[t].to(x0.dtype)[:, None, None, None]
        osa = self.sqrt_one_minus_alpha_bar[t].to(x0.dtype)[:, None, None, None]
        xt = sa * x0 + osa * noise
        return xt, noise

    @staticmethod
    def loss_fn(real_noise, pred_noise):
        pred_noise = pred_noise.to(real_noise.dtype)
        return F.mse_loss(pred_noise, real_noise)

    def train_step(self, opt, batch_x, batch_c, p_uncond=0.1):
        b = batch_x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=batch_x.device, dtype=torch.long)
        xt, noise = self.forward_noise(batch_x, t)

        # Classifier-free guidance: randomly drop condition (Eq. 29)
        mask = (torch.rand((b, 1), device=batch_c.device) >= p_uncond).to(batch_c.dtype)
        batch_c_masked = batch_c * mask

        opt.zero_grad(set_to_none=True)
        pred = self.model(xt, t, batch_c_masked)
        loss = self.loss_fn(noise, pred)
        loss.backward()
        opt.step()
        return loss

    def val_step(self, batch_x, batch_c):
        b = batch_x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=batch_x.device, dtype=torch.long)
        xt, noise = self.forward_noise(batch_x, t)
        pred = self.model(xt, t, batch_c)
        return self.loss_fn(noise, pred)

    def reverse_step(self, x_t, pred_noise, t):
        t = t.reshape(-1).long()
        a_t = self.alpha_t[t].to(x_t.dtype)[:, None, None, None]
        ab_t = self.alpha_bar_t[t].to(x_t.dtype)[:, None, None, None]
        b_t = self.beta_t[t].to(x_t.dtype)[:, None, None, None]
        one = torch.tensor(1.0, dtype=x_t.dtype, device=x_t.device)
        eps_coef = (one - a_t) / torch.sqrt(one - ab_t)
        mean = (one / torch.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(b_t) * noise

    def _guidance_weight(self, t):
        """Corrected Eq. 32: guidance strong early (high t), diminishing as denoising proceeds."""
        T = torch.tensor(float(self.timesteps), dtype=torch.float32, device=self.device)
        t_f = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        w_min, w_max, lam = 0.2, 0.7, 5.0
        return w_min + (w_max - w_min) * torch.exp(-lam * (T - t_f) / T)

    def _reverse_step_no_noise(self, x_t, pred_noise, t):
        """Reverse step returning mean only (no stochastic noise). Used at t=0."""
        t = t.reshape(-1).long()
        a_t = self.alpha_t[t].to(x_t.dtype)[:, None, None, None]
        ab_t = self.alpha_bar_t[t].to(x_t.dtype)[:, None, None, None]
        one = torch.tensor(1.0, dtype=x_t.dtype, device=x_t.device)
        eps_coef = (one - a_t) / torch.sqrt(one - ab_t)
        return (one / torch.sqrt(a_t)) * (x_t - eps_coef * pred_noise)

    def sample(self, x_init, class_vec):
        x = x_init
        b = x_init.shape[0]
        uncond_vec = torch.zeros_like(class_vec)

        for i in range(self.timesteps - 1, 0, -1):
            t = torch.full((b,), i, dtype=torch.long, device=x.device)
            eps_cond = self.model(x, t, class_vec)
            eps_uncond = self.model(x, t, uncond_vec)
            w = self._guidance_weight(i).to(x.dtype)
            guided = eps_cond + w * (eps_cond - eps_uncond)
            x = self.reverse_step(x, guided, t)

        # Final step t=0: no noise (Ho et al. 2020, Alg. 2: z=0 when t=1)
        t0 = torch.zeros((b,), dtype=torch.long, device=x.device)
        eps_cond = self.model(x, t0, class_vec)
        eps_uncond = self.model(x, t0, uncond_vec)
        w = self._guidance_weight(0).to(x.dtype)
        guided = eps_cond + w * (eps_cond - eps_uncond)
        x = self._reverse_step_no_noise(x, guided, t0)
        return x


def build_ddpm_model(feature_size: int) -> UnetConditional:
    model = UnetConditional(in_res=feature_size)
    dummy_x = torch.zeros((1, 1, feature_size, feature_size), dtype=torch.float32)
    dummy_t = torch.zeros((1,), dtype=torch.long)
    dummy_c = torch.zeros((1, 8), dtype=torch.float32)
    _ = model(dummy_x, dummy_t, dummy_c)
    return model


class DDPMDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, latent_max: float):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.latent_max = latent_max

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.permute(2, 0, 1).contiguous()
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        x = preprocess_latent_torch(x, self.latent_max)
        y = torch.from_numpy(self.y[idx]).float()
        return x, y


def make_torch_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    latent_max: float,
    val_split: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int((1.0 - val_split) * n)
    tr_idx_full, va_idx_full = idx[:split], idx[split:]
    tr_cap = len(tr_idx_full) if max_train_samples is None else min(len(tr_idx_full), max_train_samples)
    va_cap = len(va_idx_full) if max_val_samples is None else min(len(va_idx_full), max_val_samples)
    tr_idx = tr_idx_full[:tr_cap]
    va_idx = va_idx_full[:va_cap]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[va_idx], Y[va_idx]

    train_ds = DDPMDataset(Xtr, Ytr, latent_max)
    val_ds = DDPMDataset(Xva, Yva, latent_max)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def train_ddpm(args) -> None:
    max_train_samples = getattr(args, "max_train_samples", None)
    max_val_samples = getattr(args, "max_val_samples", None)
    X, Y, lbl_stats, lat_stats = build_ddpm_arrays(
        args.encoded_features,
        args.obb_vectors,
        args.scale_coeffs,
        args.curvatures,
    )
    feature_size = int(X.shape[1])
    if X.shape[1] != X.shape[2]:
        raise ValueError(f"DDPM expects square latent grids, got {X.shape}")

    ensure_dir(args.ddpm_ckpt_dir)
    train_ds, val_ds = make_torch_dataloaders(
        X, Y, args.batch_size, lat_stats.latent_max, args.val_split, args.seed,
        max_train_samples=max_train_samples, max_val_samples=max_val_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ddpm_model(feature_size).to(device)
    trainer = DDPMTrainer(model, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # NOTE: this decays once per epoch (unlike the previous TF per-step decay); kept unchanged in this pass.
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    latest_path = os.path.join(args.ddpm_ckpt_dir, "ddpm_latest.pt")
    start_epoch = 1
    if os.path.exists(latest_path):
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            lr_sched.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        logging.info("Restored DDPM checkpoint: %s", latest_path)

    best_val = float("inf")
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        tr_losses = []
        for xb, yb in train_ds:
            xb, yb = xb.to(device), yb.to(device)
            tr_losses.append(float(trainer.train_step(opt, xb, yb).item()))

        va_losses = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_ds:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(float(trainer.val_step(xb, yb).item()))
        model.train()

        tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va = float(np.mean(va_losses)) if va_losses else float("inf")
        logging.info("DDPM epoch %d/%d | train=%.6f val=%.6f", epoch, args.epochs, tr, va)

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.ddpm_ckpt_dir, "unet_best.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info("DDPM early stopping at epoch %d", epoch)
                break
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": lr_sched.state_dict(), "best_val": best_val},
            latest_path,
        )
        lr_sched.step()

    torch.save(model.state_dict(), os.path.join(args.ddpm_ckpt_dir, "unet_final.pt"))
    save_json(
        {
            "feature_size": feature_size,
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "latent_max": lat_stats.latent_max,
            "min_dim": lbl_stats.min_dim.tolist(),
            "dim_den": lbl_stats.dim_den.tolist(),
            "curv_min": lbl_stats.curv_min,
            "curv_den": lbl_stats.curv_den,
            "scale_min": lbl_stats.scale_min,
            "scale_den": lbl_stats.scale_den,
        },
        os.path.join(args.ddpm_ckpt_dir, "ddpm_meta.json"),
    )
    logging.info("DDPM training complete")


# -----------------------------
# Point-cloud dataset utilities
# -----------------------------
class PointCloudDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        if arr.ndim != 3 or arr.shape[-1] != 6:
            raise ValueError(f"Point cloud array must have shape (N,P,6), got {arr.shape}")
        self.data = torch.from_numpy(arr.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_pointcloud_loaders(
    path: str,
    batch_size: int,
    val_split: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    ensure_file(path)
    arr = np.load(path).astype(np.float32)

    n_total = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_val_full = int(n_total * val_split)
    n_train_full = n_total - n_val_full

    train_idx_full = idx[:n_train_full]
    val_idx_full = idx[n_train_full:]
    train_cap = len(train_idx_full) if max_train_samples is None else min(len(train_idx_full), max_train_samples)
    val_cap = len(val_idx_full) if max_val_samples is None else min(len(val_idx_full), max_val_samples)

    train_ds = PointCloudDataset(arr[train_idx_full[:train_cap]])
    val_ds = PointCloudDataset(arr[val_idx_full[:val_cap]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, arr.shape[1]


# -----------------------------
# Point-cloud AE model
# -----------------------------
class SelfAttention(tnn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = tnn.Conv1d(in_channels, max(in_channels // 8, 1), 1)
        self.key_conv = tnn.Conv1d(in_channels, max(in_channels // 8, 1), 1)
        self.value_conv = tnn.Conv1d(in_channels, in_channels, 1)
        self.softmax = tnn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        d_k = q.shape[1]  # key dimension for proper scaling
        attn = self.softmax(torch.bmm(q.permute(0, 2, 1), k) / (d_k ** 0.5))
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out + x


class FeatureFusion(tnn.Module):
    def __init__(self, in_xyz, in_normal, out_channels):
        super().__init__()
        self.fusion_conv = tnn.Conv1d(in_xyz + in_normal, out_channels, 1)
        self.bn = tnn.BatchNorm1d(out_channels)

    def forward(self, xyz_features, normal_features):
        combined = torch.cat((xyz_features, normal_features), dim=1)
        return F.relu(self.bn(self.fusion_conv(combined)))


class PointCloudAE(tnn.Module):
    def __init__(self, point_size: int, latent_size: int):
        super().__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        feature_size = latent_size ** 2

        self.conv1_xyz = tnn.Conv1d(3, 64, 1)
        self.conv2_xyz = tnn.Conv1d(64, 128, 1)
        self.conv3_xyz = tnn.Conv1d(128, feature_size // 2, 1)
        self.bn1_xyz = tnn.BatchNorm1d(64)
        self.bn2_xyz = tnn.BatchNorm1d(128)
        self.bn3_xyz = tnn.BatchNorm1d(feature_size // 2)

        self.conv1_normal = tnn.Conv1d(3, 32, 1)
        self.conv2_normal = tnn.Conv1d(32, 64, 1)
        self.conv3_normal = tnn.Conv1d(64, feature_size // 2, 1)
        self.bn1_normal = tnn.BatchNorm1d(32)
        self.bn2_normal = tnn.BatchNorm1d(64)
        self.bn3_normal = tnn.BatchNorm1d(feature_size // 2)

        self.feature_fusion = FeatureFusion(feature_size // 2, feature_size // 2, feature_size)
        self.self_attention = SelfAttention(feature_size)

        self.fc1 = tnn.Linear(feature_size, 1024)
        self.fc2 = tnn.Linear(1024, 2048)
        self.fc3 = tnn.Linear(2048, point_size * 6)
        self.fc_res1 = tnn.Linear(1024, 2048)
        self.fc_res2 = tnn.Linear(2048, point_size * 6)
        self.act = tnn.LeakyReLU(negative_slope=0.1)

    def encoder(self, x):
        xyz, normals = torch.split(x, 3, dim=1)
        xyz = self.act(self.bn1_xyz(self.conv1_xyz(xyz)))
        xyz = self.act(self.bn2_xyz(self.conv2_xyz(xyz)))
        xyz = self.act(self.bn3_xyz(self.conv3_xyz(xyz)))

        normals = self.act(self.bn1_normal(self.conv1_normal(normals)))
        normals = self.act(self.bn2_normal(self.conv2_normal(normals)))
        normals = self.act(self.bn3_normal(self.conv3_normal(normals)))

        fused = self.feature_fusion(xyz, normals)
        x = self.self_attention(fused)
        x = F.adaptive_max_pool1d(x, 1)
        return x.view(-1, self.latent_size, self.latent_size)

    def decoder(self, x):
        x = x.view(-1, self.latent_size ** 2)
        x = self.act(self.fc1(x))
        res1 = x
        x = self.act(self.fc2(x))
        x = x + self.fc_res1(res1)
        res2 = x
        x = self.fc3(x)
        x = x + self.fc_res2(res2)
        return x.view(-1, self.point_size, 6)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# -----------------------------
# AE train / eval / encode / decode
# -----------------------------
def _chamfer_normal_loss(recon, target, lambda_p=10.0):
    """Eq. 20-22: L = lambda_p * L_chamfer + L_normal."""
    coords_out = recon[:, :, :3]
    coords_tgt = target[:, :, :3]

    # Pairwise squared distances [B, N, M]
    diff = coords_out.unsqueeze(2) - coords_tgt.unsqueeze(1)
    dist = (diff ** 2).sum(-1)

    min_dist_x = torch.min(dist, dim=2)[0]  # [B, N]
    min_dist_y = torch.min(dist, dim=1)[0]  # [B, M]
    chamfer = torch.mean(min_dist_x, dim=1) + torch.mean(min_dist_y, dim=1)
    chamfer = chamfer.mean()

    # Normal consistency loss if normals available
    if recon.shape[-1] >= 6 and target.shape[-1] >= 6:
        normals_out = F.normalize(recon[:, :, 3:6], dim=-1)
        normals_tgt = F.normalize(target[:, :, 3:6], dim=-1)

        idx_x = torch.argmin(dist, dim=2)  # [B, N]
        idx_y = torch.argmin(dist, dim=1)  # [B, M]

        nearest_y_n = torch.gather(normals_tgt, 1, idx_x.unsqueeze(-1).expand(-1, -1, 3))
        nearest_x_n = torch.gather(normals_out, 1, idx_y.unsqueeze(-1).expand(-1, -1, 3))

        cos_x = (normals_out * nearest_y_n).sum(-1)
        cos_y = (normals_tgt * nearest_x_n).sum(-1)

        nloss = (1 - torch.abs(cos_x)).mean(1) + (1 - torch.abs(cos_y)).mean(1)
        nloss = nloss.mean()
    else:
        nloss = torch.tensor(0.0, device=recon.device)

    return lambda_p * chamfer + nloss


def train_epoch_ae(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        inp = batch.permute(0, 2, 1)
        recon = model(inp)
        loss = _chamfer_normal_loss(recon, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def validate_epoch_ae(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inp = batch.permute(0, 2, 1)
            recon = model(inp)
            loss = _chamfer_normal_loss(recon, batch)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def save_ae_epoch_visual(model, fixed_batch, device, out_path: str, fig_count: int = 4) -> None:
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = fixed_batch.to(device)
        inp = batch.permute(0, 2, 1).contiguous()
        latent = model.encoder(inp)
        recon = model(inp)

    gt_np = batch.detach().cpu().numpy()
    latent_np = latent.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()

    n = min(fig_count, gt_np.shape[0])
    if n <= 0:
        return

    fig = plt.figure(figsize=(3.2 * n, 8.5))
    for i in range(n):
        ax1 = fig.add_subplot(3, n, i + 1, projection="3d")
        ax1.scatter(gt_np[i, :, 0], gt_np[i, :, 1], gt_np[i, :, 2], s=1, c="#2F6DB3")
        ax1.set_title(f"GT {i}")
        ax1.axis("off")

        ax2 = fig.add_subplot(3, n, n + i + 1)
        im = ax2.imshow(latent_np[i], cmap="coolwarm")
        ax2.set_title(f"Latent {i}")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(3, n, 2 * n + i + 1, projection="3d")
        ax3.scatter(recon_np[i, :, 0], recon_np[i, :, 1], recon_np[i, :, 2], s=1, c="#E68613")
        ax3.set_title(f"Recon {i}")
        ax3.axis("off")

    ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if model_was_training:
        model.train()


def train_ae(args) -> None:
    max_train_samples = getattr(args, "max_train_samples", None)
    max_val_samples = getattr(args, "max_val_samples", None)
    ensure_dir(args.ae_ckpt_dir)
    viz_dir = os.path.join(args.ae_ckpt_dir, "epoch_viz")
    ensure_dir(viz_dir)
    train_loader, val_loader, point_size = get_pointcloud_loaders(
        args.pointcloud_path,
        args.ae_batch_size,
        args.val_split,
        args.seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointCloudAE(point_size=point_size, latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ae_lr)
    fixed_val_batch = next(iter(val_loader), None)

    start_epoch = 1
    best_val = float("inf")
    latest_path = os.path.join(args.ae_ckpt_dir, "ae_latest.pth")
    if os.path.exists(latest_path):
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", best_val)
        logging.info("Restored AE from epoch %d", state["epoch"])

    no_improve = 0
    for epoch in range(start_epoch, args.ae_epochs + 1):
        tr = train_epoch_ae(model, train_loader, optimizer, device)
        va = validate_epoch_ae(model, val_loader, device)
        logging.info("AE epoch %d/%d | train=%.6f val=%.6f", epoch, args.ae_epochs, tr, va)
        if fixed_val_batch is not None:
            save_ae_epoch_visual(
                model,
                fixed_val_batch,
                device,
                os.path.join(viz_dir, f"ae_epoch_{epoch:04d}.png"),
            )

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val": best_val}, latest_path)

        ckpt_path = os.path.join(args.ae_ckpt_dir, f"ae_epoch_{epoch:04d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.ae_ckpt_dir, "ae_best.pth"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info("AE early stopping at epoch %d", epoch)
                break

    torch.save(model.state_dict(), os.path.join(args.ae_ckpt_dir, "ae_final.pth"))
    save_json({"point_size": point_size, "latent_size": args.latent_size}, os.path.join(args.ae_ckpt_dir, "ae_meta.json"))
    logging.info("AE training complete")


def load_ae_for_decode(ae_ckpt_dir: str, prefer_best: bool = True, device: Optional[torch.device] = None) -> PointCloudAE:
    meta = load_json(os.path.join(ae_ckpt_dir, "ae_meta.json"))
    point_size = int(meta["point_size"])
    latent_size = int(meta["latent_size"])
    model = PointCloudAE(point_size=point_size, latent_size=latent_size)

    weights_path = os.path.join(ae_ckpt_dir, "ae_best.pth" if prefer_best else "ae_final.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ae_ckpt_dir, "ae_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device or "cpu"))
    model.eval()
    if device is not None:
        model.to(device)
    return model


def encode_dataset(args) -> None:
    ensure_file(args.pointcloud_path)
    ensure_dir(str(Path(args.encoded_features).parent))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae_for_decode(args.ae_ckpt_dir, prefer_best=True, device=device)
    ae.eval()

    arr = np.load(args.pointcloud_path).astype(np.float32)
    if arr.shape[1] != ae.point_size:
        raise ValueError(
            f"Point count mismatch: dataset has {arr.shape[1]} points per cloud, "
            f"but AE expects {ae.point_size}"
        )
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError(f"Point cloud array must have shape (N,P,6), got {arr.shape}")

    batch_size = int(args.encode_batch_size)
    latents = []

    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            end = min(start + batch_size, len(arr))
            batch = torch.from_numpy(arr[start:end]).to(device)          # [B, P, 6]
            batch = batch.permute(0, 2, 1).contiguous()                 # [B, 6, P]
            z = ae.encoder(batch)                                       # [B, latent_size, latent_size]
            latents.append(z.cpu().numpy().astype(np.float32))

    encoded = np.concatenate(latents, axis=0)
    np.save(args.encoded_features, encoded)

    logging.info("Encoded dataset saved to %s", args.encoded_features)
    logging.info("Encoded shape: %s", encoded.shape)
    logging.info("Encoded min/max: %.6f / %.6f", float(encoded.min()), float(encoded.max()))
    
# -----------------------------
# Graph latent generation
# -----------------------------
def load_subgraphs(npz_path: str):
    ensure_file(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "subgraphs" not in data:
        raise ValueError(f"Expected key 'subgraphs' in {npz_path}, got {list(data.keys())}")
    return data["subgraphs"]


def ddpm_load_for_sampling(ddpm_ckpt_dir: str) -> Tuple[UnetConditional, DDPMTrainer, Dict]:
    meta_path = os.path.join(ddpm_ckpt_dir, "ddpm_meta.json")
    weights_path = os.path.join(ddpm_ckpt_dir, "unet_best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ddpm_ckpt_dir, "unet_final.pt")

    meta = load_json(meta_path)
    feature_size = int(meta["feature_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ddpm_model(feature_size).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    trainer = DDPMTrainer(model, timesteps=int(meta["timesteps"]), beta_start=float(meta["beta_start"]), beta_end=float(meta["beta_end"]), device=device)
    return model, trainer, meta


def _node_condition_vector(node_obb: np.ndarray, node_curv: float, node_scale: float, meta: Dict) -> np.ndarray:
    min_dim = np.asarray(meta["min_dim"], dtype=np.float32)
    dim_den = np.asarray(meta["dim_den"], dtype=np.float32)
    obb = np.asarray(node_obb, dtype=np.float32)
    if obb.shape[0] < 6:
        raise ValueError(f"Node OBB has invalid shape: {obb.shape}")
    obb6 = np.concatenate([obb[:3], (obb[3:6] - min_dim) / np.maximum(dim_den, 1e-12)], axis=0)

    curv_norm = (float(node_curv) - float(meta["curv_min"])) / max(float(meta["curv_den"]), 1e-12)
    scale_norm = (float(node_scale) - float(meta["scale_min"])) / max(float(meta["scale_den"]), 1e-12)

    return np.concatenate([obb6, [curv_norm], [scale_norm]], axis=0).astype(np.float32)


def generate_latents(args) -> None:
    ensure_dir(args.latent_node_dir)
    ensure_dir(args.latent_graph_dir)

    _, trainer, meta = ddpm_load_for_sampling(args.ddpm_ckpt_dir)
    subgraphs = load_subgraphs(args.subgraphs_path)

    max_graphs = min(args.max_graphs, len(subgraphs)) if args.max_graphs > 0 else len(subgraphs)
    feature_size = int(meta["feature_size"])
    latent_max = float(meta["latent_max"])

    for g_id in range(max_graphs):
        sg = subgraphs[g_id]
        obb_arr = np.asarray(sg["obb_euler"])
        curv_arr = np.asarray(sg["curvatures"]).reshape(-1)
        scale_arr = np.asarray(sg["scales"]).reshape(-1)

        n_nodes = min(len(obb_arr), len(curv_arr), len(scale_arr))
        graph_dir = Path(args.latent_node_dir) / f"graph_{g_id:03d}"
        graph_dir.mkdir(parents=True, exist_ok=True)

        node_ids = []
        labels = []
        latents = []

        for start in range(0, n_nodes, args.gen_batch_size):
            end = min(start + args.gen_batch_size, n_nodes)
            cond = np.stack([_node_condition_vector(obb_arr[i], curv_arr[i], scale_arr[i], meta) for i in range(start, end)], axis=0)
            device = trainer.device
            cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)
            x = torch.randn((end - start, 1, feature_size, feature_size), device=device, dtype=torch.float32)
            with torch.no_grad():
                sampled = trainer.sample(x, cond_t).cpu().numpy()
            sampled = postprocess_latent_np(sampled, latent_max)

            for i in range(end - start):
                node_id = start + i
                latent = sampled[i, 0, ...].astype(np.float32)
                np.save(graph_dir / f"node_{node_id:03d}.npy", latent)
                node_ids.append(node_id)
                labels.append(cond[i])
                latents.append(latent)

        graph_npz = Path(args.latent_graph_dir) / f"graph_{g_id:03d}_ddpm_generated.npz"
        np.savez(
            graph_npz,
            features=np.stack(latents, axis=0) if latents else np.zeros((0, feature_size, feature_size), dtype=np.float32),
            node_ids=np.asarray(node_ids, dtype=np.int32),
            labels=np.stack(labels, axis=0) if labels else np.zeros((0, 8), dtype=np.float32),
        )
        logging.info("Generated graph %03d with %d nodes", g_id, n_nodes)


# -----------------------------
# Graph decode
# -----------------------------
def decode_latents(args) -> None:
    ensure_dir(args.decoded_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae_for_decode(args.ae_ckpt_dir, prefer_best=True, device=device)

    latent_graph_dirs = sorted([p for p in Path(args.latent_node_dir).glob("graph_*") if p.is_dir()])
    if not latent_graph_dirs:
        raise FileNotFoundError(f"No graph folders found in {args.latent_node_dir}")

    for gdir in latent_graph_dirs:
        out_gdir = Path(args.decoded_dir) / gdir.name
        out_gdir.mkdir(parents=True, exist_ok=True)
        node_files = sorted(gdir.glob("node_*.npy"))

        for nf in node_files:
            latent = np.load(nf).astype(np.float32)
            if latent.ndim != 2:
                raise ValueError(f"Latent file {nf} must be 2D grid, got {latent.shape}")
            z = torch.from_numpy(latent[None, ...]).to(device)
            with torch.no_grad():
                decoded = ae.decoder(z).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(out_gdir / nf.name, decoded)

        logging.info("Decoded %s (%d nodes)", gdir.name, len(node_files))


# -----------------------------
# Graph assembly
# -----------------------------
def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def try_get_translation(node_meta: Dict, node_idx: int) -> np.ndarray:
    for key in ["centers", "center", "translations", "translation", "positions", "position"]:
        if key in node_meta:
            arr = np.asarray(node_meta[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[node_idx, :3].astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def transform_pointcloud(pc: np.ndarray, obb_euler: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = pc.copy()
    xyz = out[:, :3]

    rot = euler_xyz_to_matrix(float(obb_euler[0]), float(obb_euler[1]), float(obb_euler[2])) if len(obb_euler) >= 3 else np.eye(3, dtype=np.float32)
    if len(obb_euler) >= 6:
        scale = np.asarray(obb_euler[3:6], dtype=np.float32)
    else:
        scale = np.ones(3, dtype=np.float32)

    xyz_world = (xyz * scale[None, :]) @ rot.T + translation[None, :]
    out[:, :3] = xyz_world.astype(np.float32)
    if out.shape[1] >= 6:
        normals = out[:, 3:6]
        out[:, 3:6] = (normals @ rot.T).astype(np.float32)
    return out


def save_ply_if_available(points: np.ndarray, path: Path) -> bool:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return False
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    o3d.io.write_point_cloud(str(path), pc)
    return True


def assemble(args) -> None:
    ensure_dir(args.assembly_dir)
    subgraphs = load_subgraphs(args.subgraphs_path)

    decoded_graph_dirs = sorted([p for p in Path(args.decoded_dir).glob("graph_*") if p.is_dir()])
    if not decoded_graph_dirs:
        raise FileNotFoundError(f"No decoded graph folders found in {args.decoded_dir}")

    for gdir in decoded_graph_dirs:
        g_id = int(gdir.name.split("_")[-1])
        if g_id >= len(subgraphs):
            logging.warning("Skipping %s; graph id out of bounds", gdir.name)
            continue

        sg = subgraphs[g_id]
        obb_arr = np.asarray(sg["obb_euler"])

        full_points = []
        missing = 0
        node_files = sorted(gdir.glob("node_*.npy"))
        for nf in node_files:
            node_idx = int(nf.stem.split("_")[-1])
            if node_idx >= len(obb_arr):
                logging.warning("Graph %03d node %03d metadata missing", g_id, node_idx)
                missing += 1
                continue
            node_pc = np.load(nf).astype(np.float32)
            if node_pc.ndim != 2 or node_pc.shape[1] < 3:
                logging.warning("Invalid node point cloud %s shape=%s", str(nf), node_pc.shape)
                missing += 1
                continue
            tr = try_get_translation(sg, node_idx)
            transformed = transform_pointcloud(node_pc, obb_arr[node_idx], tr)
            full_points.append(transformed)

        if not full_points:
            logging.warning("Graph %03d had no valid decoded nodes", g_id)
            continue

        assembled = np.concatenate(full_points, axis=0).astype(np.float32)
        out_base = Path(args.assembly_dir) / f"graph_{g_id:03d}_assembly"
        np.save(str(out_base) + ".npy", assembled)
        np.savez(str(out_base) + ".npz", points=assembled, graph_id=g_id, missing_nodes=missing)
        if args.export_ply:
            ok = save_ply_if_available(assembled, Path(str(out_base) + ".ply"))
            if not ok:
                logging.warning("open3d not available; skipped PLY export for graph %03d", g_id)

        logging.info("Assembled graph %03d | points=%d | missing_nodes=%d", g_id, len(assembled), missing)


# -----------------------------
# CLI
# -----------------------------
def add_common_paths(parser):
    parser.add_argument("--encoded_features", default=DEFAULT_ENCODED_FEATURES)
    parser.add_argument("--obb_vectors", default=DEFAULT_OBB)
    parser.add_argument("--scale_coeffs", default=DEFAULT_SCALE)
    parser.add_argument("--curvatures", default=DEFAULT_CURV)
    parser.add_argument("--pointcloud_path", default=DEFAULT_POINTCLOUDS)
    parser.add_argument("--subgraphs_path", default=DEFAULT_SUBGRAPHS)



def build_parser():
    p = argparse.ArgumentParser(description="End-to-end DDPM + PointCloudAE pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_ddpm = sub.add_parser("train_ddpm", help="Train conditional DDPM")
    add_common_paths(p_ddpm)
    p_ddpm.add_argument("--ddpm_ckpt_dir", default=DEFAULT_DDPM_CKPT_DIR)
    p_ddpm.add_argument("--batch_size", type=int, default=64)
    p_ddpm.add_argument("--epochs", type=int, default=300)
    p_ddpm.add_argument("--patience", type=int, default=30)
    p_ddpm.add_argument("--lr", type=float, default=1e-4)
    p_ddpm.add_argument("--val_split", type=float, default=0.2)
    p_ddpm.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p_ddpm.add_argument("--beta_start", type=float, default=BETA_START)
    p_ddpm.add_argument("--beta_end", type=float, default=BETA_END)
    p_ddpm.add_argument("--max_train_samples", type=int, default=None)
    p_ddpm.add_argument("--max_val_samples", type=int, default=None)
    
    p_ae = sub.add_parser("train_ae", help="Train point-cloud autoencoder")
    add_common_paths(p_ae)
    p_ae.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_ae.add_argument("--latent_size", type=int, default=32)
    p_ae.add_argument("--ae_batch_size", type=int, default=16)
    p_ae.add_argument("--ae_epochs", type=int, default=300)
    p_ae.add_argument("--patience", type=int, default=30)
    p_ae.add_argument("--ae_lr", type=float, default=1e-4)
    p_ae.add_argument("--val_split", type=float, default=0.1)
    p_ae.add_argument("--max_train_samples", type=int, default=None)
    p_ae.add_argument("--max_val_samples", type=int, default=None)
    
    p_enc = sub.add_parser("encode_dataset", help="Encode full point-cloud dataset into AE latent grids")
    add_common_paths(p_enc)
    p_enc.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_enc.add_argument("--encode_batch_size", type=int, default=64)
    
    p_gen = sub.add_parser("generate_latents", help="Generate graph-node latent grids from DDPM")
    add_common_paths(p_gen)
    p_gen.add_argument("--ddpm_ckpt_dir", default=DEFAULT_DDPM_CKPT_DIR)
    p_gen.add_argument("--latent_node_dir", default=DEFAULT_LATENT_NODE_DIR)
    p_gen.add_argument("--latent_graph_dir", default=DEFAULT_LATENT_GRAPH_DIR)
    p_gen.add_argument("--max_graphs", type=int, default=116)
    p_gen.add_argument("--gen_batch_size", type=int, default=64)

    p_dec = sub.add_parser("decode_latents", help="Decode DDPM node latents to node point clouds with AE")
    add_common_paths(p_dec)
    p_dec.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_dec.add_argument("--latent_node_dir", default=DEFAULT_LATENT_NODE_DIR)
    p_dec.add_argument("--decoded_dir", default=DEFAULT_DECODED_DIR)

    p_asm = sub.add_parser("assemble", help="Assemble decoded node point clouds into graph assemblies")
    add_common_paths(p_asm)
    p_asm.add_argument("--decoded_dir", default=DEFAULT_DECODED_DIR)
    p_asm.add_argument("--assembly_dir", default=DEFAULT_ASSEMBLY_DIR)
    p_asm.add_argument("--export_ply", action="store_true")

    p_all = sub.add_parser("run_all", help="Run full pipeline")
    add_common_paths(p_all)
    p_all.add_argument("--ddpm_ckpt_dir", default=DEFAULT_DDPM_CKPT_DIR)
    p_all.add_argument("--ae_ckpt_dir", default=DEFAULT_AE_CKPT_DIR)
    p_all.add_argument("--latent_node_dir", default=DEFAULT_LATENT_NODE_DIR)
    p_all.add_argument("--latent_graph_dir", default=DEFAULT_LATENT_GRAPH_DIR)
    p_all.add_argument("--decoded_dir", default=DEFAULT_DECODED_DIR)
    p_all.add_argument("--assembly_dir", default=DEFAULT_ASSEMBLY_DIR)
    p_all.add_argument("--encode_batch_size", type=int, default=64)

    p_all.add_argument("--batch_size", type=int, default=64)
    p_all.add_argument("--epochs", type=int, default=300)
    p_all.add_argument("--patience", type=int, default=30)
    p_all.add_argument("--lr", type=float, default=1e-4)
    p_all.add_argument("--val_split", type=float, default=0.2)
    p_all.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p_all.add_argument("--beta_start", type=float, default=BETA_START)
    p_all.add_argument("--beta_end", type=float, default=BETA_END)

    p_all.add_argument("--latent_size", type=int, default=32)
    p_all.add_argument("--ae_batch_size", type=int, default=16)
    p_all.add_argument("--ae_epochs", type=int, default=300)
    p_all.add_argument("--ae_lr", type=float, default=1e-4)
    p_all.add_argument("--max_train_samples", type=int, default=None)
    p_all.add_argument("--max_val_samples", type=int, default=None)

    p_all.add_argument("--max_graphs", type=int, default=116)
    p_all.add_argument("--gen_batch_size", type=int, default=64)
    p_all.add_argument("--export_ply", action="store_true")

    return p


def run_all(args):
    train_ae(args)
    encode_dataset(args)
    train_ddpm(args)
    generate_latents(args)
    decode_latents(args)
    assemble(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    seed_everything(args.seed)

    if args.cmd == "train_ddpm":
        train_ddpm(args)
    elif args.cmd == "train_ae":
        train_ae(args)
    elif args.cmd == "encode_dataset":
        encode_dataset(args)
    elif args.cmd == "generate_latents":
        generate_latents(args)
    elif args.cmd == "decode_latents":
        decode_latents(args)
    elif args.cmd == "assemble":
        assemble(args)
    elif args.cmd == "run_all":
        run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
