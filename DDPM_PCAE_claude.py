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

import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import GroupNormalization

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from einops import rearrange
from einops.layers.tensorflow import Rearrange

'''
python3 DDPM_PCAE_claude.py train_ae \
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
python3 DDPM_PCAE_claude.py train_ae \
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

python3 DDPM_PCAE_claude.py encode_dataset \
  --pointcloud_path data/normalized_rotated_point_clouds6.npy \
  --ae_ckpt_dir checkpoints/ae \
  --encoded_features encoded_features.npy \
  --encode_batch_size 64 \
  --seed 42 \
  --verbose

python3 DDPM_PCAE_claude.py train_ddpm \
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
python3 DDPM_PCAE_claude.py train_ddpm \
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
    tf.random.set_seed(seed)
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
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    val_split: float = 0.2,
    seed: int = 42,
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

    if max_train_samples is not None or max_val_samples is not None:
        vt = int(round(val_split * n))
        tt = n - vt

        train_cap = tt if max_train_samples is None else min(tt, max_train_samples)
        val_cap = vt if max_val_samples is None else min(vt, max_val_samples)

        total_cap = train_cap + val_cap
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)[:total_cap]

        encoded = encoded[idx]
        obb = obb[idx]
        scale = scale[idx]
        curv = curv[idx]

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


def preprocess_latent_tf(x: tf.Tensor, latent_max: float) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value(x / latent_max, -1.0, 1.0)


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


class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        half_dim = self.dim // 2
        inv_freq_scale = math.log(self.max_positions) / max(half_dim - 1, 1)
        positions = tf.cast(tf.range(half_dim), x.dtype)
        emb = tf.exp(positions * -inv_freq_scale)
        emb = tf.cast(x[:, None], emb.dtype) * emb[None, :]
        return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)


class Identity(Layer):
    def call(self, x, training=True):
        return tf.identity(x)


class Residual(Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x


def Upsample(dim):
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding="SAME")


def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding="SAME")


class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.g = self.add_weight(name="g", shape=[1, 1, 1, dim], initializer="ones", trainable=True)
        self.b = self.add_weight(name="b", shape=[1, 1, 1, dim], initializer="zeros", trainable=True)

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps) * self.g + self.b


class PreNorm(Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)


class SiLU(Layer):
    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


class GELU(Layer):
    def call(self, x, training=True):
        return tf.keras.activations.gelu(x)


class Block(Layer):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding="SAME")
        self.norm = GroupNormalization(groups=min(groups, dim), epsilon=1e-5)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1.0) + beta
        return self.act(x)


class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = Sequential([SiLU(), nn.Dense(units=dim_out * 2)]) if exists(time_emb_dim) else None
        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = nn.Conv2D(filters=dim_out, kernel_size=1, strides=1) if dim != dim_out else Identity()

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)
        h = self.block1(x, gamma_beta=gamma_beta, training=training)
        h = self.block2(h, training=training)
        return h + self.res_conv(x)


class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = Sequential([nn.Conv2D(filters=dim, kernel_size=1, strides=1), LayerNorm(dim)])

    def call(self, x, training=True):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b x y (hh c) -> b hh c (x y)", hh=self.heads), qkv)
        q = tf.nn.softmax(q, axis=-2) * self.scale
        k = tf.nn.softmax(k, axis=-1)
        context = einsum("b h d n, b h e n -> b h d e", k, v)
        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b hh c (x y) -> b x y (hh c)", hh=self.heads, x=h, y=w)
        return self.to_out(out, training=training)


class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b x y (hh c) -> b hh c (x y)", hh=self.heads), qkv)
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = tf.nn.softmax(sim, axis=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b hh (x y) d -> b x y (hh d)", hh=self.heads, x=h, y=w)
        return self.to_out(out, training=training)


class MLP(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.net = Sequential([
            Rearrange("... -> ... 1"),
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)


class UnetConditional(tf.keras.Model):
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
        self.init_conv = tf.keras.layers.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding="SAME")

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = tf.keras.Sequential(
            [
                SinusoidalPosEmb(dim),
                tf.keras.layers.Dense(units=time_dim),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Dense(units=time_dim),
            ],
            name="time_embeddings",
        ) if sinusoidal_cond_mlp else MLP(time_dim)

        self.obb_embedding = tf.keras.layers.Dense(units=class_emb_dim_obb)
        self.curvature_embedding = tf.keras.layers.Dense(units=class_emb_dim_curvature)
        self.scale_embedding = tf.keras.layers.Dense(units=class_emb_dim_scale)

        block_klass = lambda din, dout, tdim: ResnetBlock(din, dout, time_emb_dim=tdim, groups=resnet_block_groups)

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            self.downs.append([
                block_klass(dim_in, dim_out, time_dim),
                block_klass(dim_out, dim_out, time_dim),
                block_klass(dim_out, dim_out, time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity(),
            ])

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind == (num_resolutions - 1)
            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_dim),
                block_klass(dim_in, dim_in, time_dim),
                block_klass(dim_in, dim_in, time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity(),
            ])

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, lambda: default_out_dim)
        self.final_conv = tf.keras.Sequential([block_klass(dim * 2, dim, time_dim), tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)], name="output")

    def call(self, x, time=None, class_value=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(tf.cast(time, tf.float32))
        obb, curvature, scale = tf.split(class_value, [self.obb_length, self.curvature_length, self.scale_length], axis=-1)
        class_emb = tf.concat([
            self.obb_embedding(obb),
            self.curvature_embedding(curvature),
            self.scale_embedding(scale),
        ], axis=-1)
        t = tf.concat([t, class_emb], axis=-1)

        h = []
        for block1, block2, block3, attn, downsample in self.downs:
            x = block1(x, t, training=training)
            x = block2(x, t, training=training)
            x = block3(x, t, training=training)
            x = attn(x, training=training)
            h.append(x)
            x = downsample(x, training=training)

        x = self.mid_block1(x, t, training=training)
        x = self.mid_attn(x, training=training)
        x = self.mid_block2(x, t, training=training)

        for block1, block2, block3, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t, training=training)
            x = block2(x, t, training=training)
            x = block3(x, t, training=training)
            x = attn(x, training=training)
            x = upsample(x, training=training)

        return self.final_conv(x)


# -----------------------------
# DDPM train / eval / sample
# -----------------------------
class DDPMTrainer:
    def __init__(self, model: UnetConditional, timesteps: int = TIMESTEPS, beta_start: float = BETA_START, beta_end: float = BETA_END):
        self.model = model
        self.timesteps = timesteps
        beta = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha, axis=0)

        self.alpha_tf = tf.constant(alpha, dtype=tf.float32)
        self.alpha_bar_tf = tf.constant(alpha_bar, dtype=tf.float32)
        self.beta_tf = tf.constant(beta, dtype=tf.float32)
        self.sqrt_alpha_bar = tf.constant(np.sqrt(alpha_bar), dtype=tf.float32)
        self.sqrt_one_minus_alpha_bar = tf.constant(np.sqrt(1.0 - alpha_bar), dtype=tf.float32)

    @tf.function
    def forward_noise(self, x0, t):
        noise = tf.random.normal(tf.shape(x0), dtype=x0.dtype)
        sa = tf.cast(tf.gather(self.sqrt_alpha_bar, t), x0.dtype)[:, None, None, None]
        osa = tf.cast(tf.gather(self.sqrt_one_minus_alpha_bar, t), x0.dtype)[:, None, None, None]
        xt = sa * x0 + osa * noise
        return xt, noise

    @staticmethod
    @tf.function
    def loss_fn(real_noise, pred_noise):
        pred_noise = tf.cast(pred_noise, real_noise.dtype)
        return tf.reduce_mean(tf.math.squared_difference(real_noise, pred_noise))

    @tf.function
    def train_step(self, opt, batch_x, batch_c, p_uncond=0.1):
        b = tf.shape(batch_x)[0]
        t = tf.random.uniform([b], minval=0, maxval=self.timesteps, dtype=tf.int32)
        xt, noise = self.forward_noise(batch_x, t)

        # Classifier-free guidance: randomly drop condition (Eq. 29)
        mask = tf.cast(tf.random.uniform([b, 1]) >= p_uncond, batch_c.dtype)
        batch_c_masked = batch_c * mask

        with tf.GradientTape() as tape:
            pred = self.model(xt, t, batch_c_masked, training=True)
            loss = self.loss_fn(noise, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def val_step(self, batch_x, batch_c):
        b = tf.shape(batch_x)[0]
        t = tf.random.uniform([b], minval=0, maxval=self.timesteps, dtype=tf.int32)
        xt, noise = self.forward_noise(batch_x, t)
        pred = self.model(xt, t, batch_c, training=False)
        return self.loss_fn(noise, pred)

    @tf.function
    def reverse_step(self, x_t, pred_noise, t):
        t = tf.cast(tf.reshape(t, [-1]), tf.int32)
        a_t = tf.cast(tf.gather(self.alpha_tf, t), x_t.dtype)[:, None, None, None]
        ab_t = tf.cast(tf.gather(self.alpha_bar_tf, t), x_t.dtype)[:, None, None, None]
        b_t = tf.cast(tf.gather(self.beta_tf, t), x_t.dtype)[:, None, None, None]
        one = tf.constant(1.0, dtype=x_t.dtype)
        eps_coef = (one - a_t) / tf.sqrt(one - ab_t)
        mean = (one / tf.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
        noise = tf.random.normal(tf.shape(x_t), dtype=x_t.dtype)
        return mean + tf.sqrt(b_t) * noise

    @tf.function
    def _guidance_weight(self, t):
        """Corrected Eq. 32: guidance strong early (high t), diminishing as denoising proceeds."""
        T = tf.cast(self.timesteps, tf.float32)
        t_f = tf.cast(t, tf.float32)
        w_min, w_max, lam = 0.2, 0.7, 5.0
        return w_min + (w_max - w_min) * tf.exp(-lam * (T - t_f) / T)

    @tf.function
    def _reverse_step_no_noise(self, x_t, pred_noise, t):
        """Reverse step returning mean only (no stochastic noise). Used at t=0."""
        t = tf.cast(tf.reshape(t, [-1]), tf.int32)
        a_t = tf.cast(tf.gather(self.alpha_tf, t), x_t.dtype)[:, None, None, None]
        ab_t = tf.cast(tf.gather(self.alpha_bar_tf, t), x_t.dtype)[:, None, None, None]
        one = tf.constant(1.0, dtype=x_t.dtype)
        eps_coef = (one - a_t) / tf.sqrt(one - ab_t)
        return (one / tf.sqrt(a_t)) * (x_t - eps_coef * pred_noise)

    @tf.function
    def sample(self, x_init, class_vec):
        x = x_init
        b = tf.shape(x_init)[0]
        uncond_vec = tf.zeros_like(class_vec)

        def cond(i, _x):
            return i > 0  # Stop before t=0; final step handled separately

        def body(i, _x):
            t = tf.fill([b], tf.cast(i, tf.int32))
            # Classifier-free guidance (Eq. 31)
            eps_cond = self.model(_x, t, class_vec, training=False)
            eps_uncond = self.model(_x, t, uncond_vec, training=False)
            w = self._guidance_weight(i)
            guided = eps_cond + w * (eps_cond - eps_uncond)
            _x = self.reverse_step(_x, guided, t)
            return i - 1, _x

        i0 = tf.constant(self.timesteps - 1, dtype=tf.int32)
        _, x = tf.while_loop(cond, body, [i0, x], shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, None, 1])])

        # Final step t=0: no noise (Ho et al. 2020, Alg. 2: z=0 when t=1)
        t0 = tf.fill([b], tf.constant(0, tf.int32))
        eps_cond = self.model(x, t0, class_vec, training=False)
        eps_uncond = self.model(x, t0, uncond_vec, training=False)
        w = self._guidance_weight(0)
        guided = eps_cond + w * (eps_cond - eps_uncond)
        x = self._reverse_step_no_noise(x, guided, t0)
        return x


def build_ddpm_model(feature_size: int) -> UnetConditional:
    model = UnetConditional(in_res=feature_size)
    dummy_x = tf.zeros((1, feature_size, feature_size, 1), dtype=tf.float32)
    dummy_t = tf.zeros((1,), dtype=tf.int32)
    dummy_c = tf.zeros((1, 8), dtype=tf.float32)
    _ = model(dummy_x, dummy_t, dummy_c, training=False)
    return model


def make_tf_datasets(X: np.ndarray, Y: np.ndarray, batch_size: int, latent_max: float, val_split: float, seed: int):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int((1.0 - val_split) * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[va_idx], Y[va_idx]

    def _map(x, y):
        x = preprocess_latent_tf(x, latent_max)
        y = tf.cast(y, tf.float32)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(len(Xtr), seed=seed).map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((Xva, Yva)).map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train_ddpm(args) -> None:
    X, Y, lbl_stats, lat_stats = build_ddpm_arrays(
        args.encoded_features,
        args.obb_vectors,
        args.scale_coeffs,
        args.curvatures,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        val_split=args.val_split,
        seed=args.seed,
    )
    feature_size = int(X.shape[1])
    if X.shape[1] != X.shape[2]:
        raise ValueError(f"DDPM expects square latent grids, got {X.shape}")

    ensure_dir(args.ddpm_ckpt_dir)
    train_ds, val_ds = make_tf_datasets(X, Y, args.batch_size, lat_stats.latent_max, args.val_split, args.seed)

    model = build_ddpm_model(feature_size)
    trainer = DDPMTrainer(model, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)

    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, decay_steps=10000, decay_rate=0.9, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    ckpt = tf.train.Checkpoint(unet=model, optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, args.ddpm_ckpt_dir, max_to_keep=5)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        logging.info("Restored DDPM checkpoint: %s", manager.latest_checkpoint)

    best_val = float("inf")
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        tr_losses = []
        for xb, yb in train_ds:
            tr_losses.append(float(trainer.train_step(opt, xb, yb).numpy()))
        va_losses = [float(trainer.val_step(xb, yb).numpy()) for xb, yb in val_ds]

        tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va = float(np.mean(va_losses)) if va_losses else float("inf")
        logging.info("DDPM epoch %d/%d | train=%.6f val=%.6f", epoch, args.epochs, tr, va)

        if va < best_val:
            best_val = va
            no_improve = 0
            manager.save(checkpoint_number=epoch)
            model.save_weights(os.path.join(args.ddpm_ckpt_dir, "unet_best.weights.h5"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info("DDPM early stopping at epoch %d", epoch)
                break

    model.save_weights(os.path.join(args.ddpm_ckpt_dir, "unet_final.weights.h5"))
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
    n_val_full = int(n_total * val_split)
    n_train_full = n_total - n_val_full

    train_cap = n_train_full if max_train_samples is None else min(n_train_full, max_train_samples)
    val_cap = n_val_full if max_val_samples is None else min(n_val_full, max_val_samples)

    total_cap = train_cap + val_cap
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)[:total_cap]
    arr = arr[idx]

    ds = PointCloudDataset(arr)
    n_val = val_cap
    n_train = train_cap

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

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


def train_ae(args) -> None:
    ensure_dir(args.ae_ckpt_dir)
    train_loader, val_loader, point_size = get_pointcloud_loaders(
        args.pointcloud_path,
        args.ae_batch_size,
        args.val_split,
        args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointCloudAE(point_size=point_size, latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ae_lr)

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
    weights_path = os.path.join(ddpm_ckpt_dir, "unet_best.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ddpm_ckpt_dir, "unet_final.weights.h5")

    meta = load_json(meta_path)
    feature_size = int(meta["feature_size"])
    model = build_ddpm_model(feature_size)
    model.load_weights(weights_path)

    trainer = DDPMTrainer(model, timesteps=int(meta["timesteps"]), beta_start=float(meta["beta_start"]), beta_end=float(meta["beta_end"]))
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
            cond_tf = tf.convert_to_tensor(cond, dtype=tf.float32)
            x = tf.random.normal((end - start, feature_size, feature_size, 1), dtype=tf.float32)
            sampled = trainer.sample(x, cond_tf).numpy()
            sampled = postprocess_latent_np(sampled, latent_max)

            for i in range(end - start):
                node_id = start + i
                latent = sampled[i, ..., 0].astype(np.float32)
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

    # Keep TF/PyTorch device handling isolated
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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
