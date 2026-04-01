import os
import math
from functools import partial
from inspect import isfunction

import numpy as np
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from tensorflow import einsum, keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def configure_runtime():
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    tf.get_logger().setLevel('ERROR')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.optimizer.set_jit(True)


def preprocess_curv_np(c):
    mn, mx = c.min(), c.max()
    return (c - mn) / (mx - mn)


def build_labels(obb_vectors, curv_coeffs, scale_coeffs):
    min_dim = obb_vectors[:, 3:].min(axis=0)
    max_dim = obb_vectors[:, 3:].max(axis=0)

    angles = obb_vectors[:, :3]
    dims = (obb_vectors[:, 3:] - min_dim) / (max_dim - min_dim)
    obb_np = np.concatenate([angles, dims], axis=1)
    curv_np = preprocess_curv_np(curv_coeffs)
    scale_np = preprocess_curv_np(scale_coeffs)
    return np.hstack([obb_np, curv_np.reshape(-1, 1), scale_np.reshape(-1, 1)])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        half_dim = self.dim // 2
        inv_freq_scale = math.log(self.max_positions) / (half_dim - 1)
        positions = tf.cast(tf.range(half_dim), x.dtype)
        emb = tf.exp(positions * -inv_freq_scale)
        emb = x[:, None] * emb[None, :]
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
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')


def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')


class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

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
        x = self.norm(x)
        return self.fn(x)


class SiLU(Layer):
    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


class GELU(Layer):
    def __init__(self, approximate=False):
        super().__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)


class Block(Layer):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta
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
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
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
        b, h, w, c = x.shape
        qkv = tf.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = tf.nn.softmax(q, axis=-2) * self.scale
        k = tf.nn.softmax(k, axis=-1)
        context = einsum('b h d n, b h e n -> b h d e', k, v)
        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
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
        b, h, w, c = x.shape
        qkv = tf.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim_max = tf.cast(tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1)), x.dtype)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        return self.to_out(out, training=training)


class Unet_conditional(Model):
    def __init__(self, dim=128, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                 resnet_block_groups=8, learned_variance=False, sinusoidal_cond_mlp=True,
                 class_emb_dim_obb=256, class_emb_dim_curvature=64, class_emb_dim_scale=64,
                 obb_length=6, curvature_length=1, scale_length=1, in_res=16):
        super().__init__()
        self.channels = channels
        self.in_res = in_res
        self.obb_length = obb_length
        self.curvature_length = curvature_length
        self.scale_length = scale_length

        init_dim = init_dim if init_dim is not None else dim // 2
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        if sinusoidal_cond_mlp:
            self.time_mlp = Sequential([SinusoidalPosEmb(dim), nn.Dense(units=time_dim), nn.Activation('gelu'), nn.Dense(units=time_dim)])
        else:
            self.time_mlp = Sequential([Rearrange('... -> ... 1'), nn.Dense(units=time_dim), GELU(), LayerNorm(time_dim), nn.Dense(units=time_dim)])

        self.obb_embedding = nn.Dense(units=class_emb_dim_obb)
        self.curvature_embedding = nn.Dense(units=class_emb_dim_curvature)
        self.scale_embedding = nn.Dense(units=class_emb_dim_scale)

        self.downs, self.ups = [], []
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity(),
            ])

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind == (num_resolutions - 1)
            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity(),
            ])

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, lambda: default_out_dim)
        self.final_conv = Sequential([block_klass(dim * 2, dim), nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)])

    def call(self, x, time=None, class_value=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)
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


class DDPMPipeline:
    def __init__(self, feature_size=32, timesteps=1000, endbeta=0.025, batch_size=128):
        self.feature_size = feature_size
        self.timesteps = timesteps
        self.endbeta = endbeta
        self.batch_size = batch_size

        beta = np.linspace(0.0001, endbeta, timesteps)
        alpha = 1 - beta
        alpha_bar = np.cumprod(alpha)
        self.sqrt_alpha_bar = tf.constant(np.sqrt(alpha_bar), dtype=tf.float32)
        self.one_minus_sqrt_alpha_bar = tf.constant(np.sqrt(1 - alpha_bar), dtype=tf.float32)

    def forward_noise(self, x_0, t):
        noise = tf.random.normal(tf.shape(x_0), dtype=x_0.dtype)
        sa = tf.reshape(tf.cast(tf.gather(self.sqrt_alpha_bar, t), x_0.dtype), [-1, 1, 1, 1])
        osa = tf.reshape(tf.cast(tf.gather(self.one_minus_sqrt_alpha_bar, t), x_0.dtype), [-1, 1, 1, 1])
        return sa * x_0 + osa * noise, noise

