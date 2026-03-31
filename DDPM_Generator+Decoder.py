#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from PIL import Image



# 0 = all logs, 1 = filter out INFO,  2 = filter out WARNING, 3 = filter out ERROR
import tensorflow as tf
# OPTIONAL: disable oneDNN custom-op round-off warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress tf.hub warnings
tf.get_logger().setLevel("ERROR")
# 0 = all logs, 1 = filter out INFO, 2 = filter out WARNING, 3 = filter out ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
# also use the Python logger

from tensorflow import keras, einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
import tensorflow_datasets as tfds
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from einops import rearrange
from einops.layers.tensorflow import Rearrange
from functools import partial
from inspect import isfunction
# — TF-2 GPU setup & performance tuning —
# (Replaces your old tf.compat.v1.ConfigProto / Session block)

# OPTIONAL: disable oneDNN custom-op rounding warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# discover GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # grow GPU memory as needed (avoid OOM)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # pin to GPU 0 only (comment out to use all GPUs)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU found, running on CPU.")

# Mixed-precision for ~2–3× throughput on modern NVIDIA cards
from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

# Enable XLA JIT to fuse kernels
tf.config.optimizer.set_jit(True)



# In[3]:


# ─────── Load & slice raw numpy arrays ───────
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ─────── Load & slice raw numpy arrays ───────
encoded_features = np.load("encoded_features_CURVATURE.npy")
obb_vectors      = np.load("obb_vectors_open3d_euler.npy")
scale_coeffs     = np.load("normalized_scale_coefficients6.npy")
curv_coeffs      = np.load("average_mean_curvatures.npy")

# Compute per-axis mins/maxs for OBB dims
min_dim = obb_vectors[:, 3:].min(axis=0)
max_dim = obb_vectors[:, 3:].max(axis=0)

# Preprocess in numpy once (cheap), to produce final label array
def preprocess_obb_np(v):
    angles = v[:, :3]  # assume already normalized
    dims   = (v[:, 3:] - min_dim) / (max_dim - min_dim)
    return np.concatenate([angles, dims], axis=1)

def preprocess_curv_np(c):
    mn, mx = c.min(), c.max()
    return (c - mn) / (mx - mn)

# build labels
obb_np   = preprocess_obb_np(obb_vectors)
curv_np  = preprocess_curv_np(curv_coeffs)
scale_np = preprocess_curv_np(scale_coeffs)
labels   = np.hstack([obb_np, curv_np.reshape(-1,1), scale_np.reshape(-1,1)])

# subset
N = 11892
X = encoded_features[:N]
Y = labels[:N]

# ─────── Train/test split ───────
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Compute normalization constant once
MAX = X.max() / 2.0

# ─────── Define per-batch GPU‐friendly transforms ───────
def preprocess(x, y):
    # x: raw feature map, y: class vector
    x = tf.cast(x, tf.float32) / MAX - 1.0
    y = tf.cast(y, tf.float32)
    return x, y

def postprocess(x):
    # undo the normalization for saving or plotting
    return (x + 1.0) * MAX

# ─────── GPU‐friendly tf.data pipelines ───────
BATCH_SIZE = 128

train_dataset = (
    tf.data.Dataset
      .from_tensor_slices((X_train, Y_train))
      .cache()
      .shuffle(buffer_size=X_train.shape[0])
      .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset
      .from_tensor_slices((X_test, Y_test))
      .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

print(f"▶️ Train batches: {len(X_train)//BATCH_SIZE}")
print(f"▶️ Test  batches: {len(X_test)//BATCH_SIZE}")
print(f"🖥️ Pipeline ready for GPU at BATCH_SIZE = {BATCH_SIZE}")

# ─────── DDPM noise schedule ───────

timesteps = 1000

endbeta = 0.025
beta = np.linspace(0.0001, endbeta, timesteps)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

# ─── Convert schedule arrays into TF constants ───
# (so gather stays in-graph and on-device)
sqrt_alpha_bar = tf.constant(np.sqrt(alpha_bar), dtype=tf.float32)
one_minus_sqrt_alpha_bar = tf.constant(np.sqrt(1 - alpha_bar), dtype=tf.float32)


# ─── Convert all schedules into TF constants ───
alpha_tf     = tf.constant(alpha,     dtype=tf.float32)
alpha_bar_tf = tf.constant(alpha_bar, dtype=tf.float32)
beta_tf      = tf.constant(beta,      dtype=tf.float32)


def set_key(key):
    np.random.seed(key)

def forward_noise(x_0, t):
    # sample noise directly on GPU
    noise = tf.random.normal(tf.shape(x_0), dtype=x_0.dtype)
    # cast the gathered α-bars to match x_0’s dtype
    sa  = tf.reshape(
              tf.cast(tf.gather(sqrt_alpha_bar, t), x_0.dtype),
              [-1,1,1,1]
          )
    osa = tf.reshape(
              tf.cast(tf.gather(one_minus_sqrt_alpha_bar, t), x_0.dtype),
              [-1,1,1,1]
          )
    noisy_image = sa * x_0 + osa * noise
    return noisy_image, noise


def generate_timestamp(key, num):
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)

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
        # build the integer range, then cast to the activation dtype
        positions = tf.cast(tf.range(half_dim), x.dtype)        # int32→float16 or float32
        emb = tf.exp(positions * -inv_freq_scale)               # now emb is x.dtype
        emb = x[:, None] * emb[None, :]                         # broadcast multiply
        return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        
class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)

class Residual(Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x

def Upsample(dim):
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')

def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=True):
        var  = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        γ = self.g# tf.cast(self.g, tf.float16)      # cast scale to float16
        β = self.b#tf.cast(self.b, tf.float16)      # cast bias  to float16
        return (x - mean) / tf.sqrt(var + self.eps) * γ + β


class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        x = self.norm(x)
        return self.fn(x)

class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)

class Block(Layer):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta
        x = self.act(x)
        return x

class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        self.mlp = Sequential([
            SiLU(),
            nn.Dense(units=dim_out * 2)
        ]) if exists(time_emb_dim) else None

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
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.attend = nn.Softmax()
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            LayerNorm(dim)
        ])

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)

        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)

        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
        out = self.to_out(out, training=training)

        return out

class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, x.dtype)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x = h, y = w)
        out = self.to_out(out, training=training)

        return out
    
class MLP(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.net = Sequential([
            Rearrange('... -> ... 1'), # expand_dims(axis=-1)
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
    
class ClassConditioning(Layer):
    def __init__(self, res, num_channels=1):
        super().__init__()
        self.block = Sequential([
            nn.Dense(res * res * num_channels),
            SiLU(),
            nn.Reshape((res, res, num_channels))
        ])

        self.block.compile()

    def call(self, x):
        return self.block(x)

class Unet_conditional(tf.keras.Model):
    def __init__(self,
                 dim=128,  # Increased initial dimension
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),  # More levels of upscaling and downscaling
                 channels=1,
                 resnet_block_groups=8,  # More groups in group normalization
                 learned_variance=False,
                 sinusoidal_cond_mlp=True,
                 class_emb_dim_obb=256,
                 class_emb_dim_curvature=64,
                 class_emb_dim_scale=64,
                 obb_length=6,
                 curvature_length=1,
                 scale_length=1,
                 in_res=16):
        super(Unet_conditional, self).__init__()
        
        self.channels = channels
        self.in_res = in_res
        self.obb_length = obb_length
        self.curvature_length = curvature_length
        self.scale_length = scale_length

        
        init_dim = init_dim if init_dim is not None else dim // 2
        self.init_conv = tf.keras.layers.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        time_dim = dim * 4
        if sinusoidal_cond_mlp:
            self.time_mlp = tf.keras.Sequential([
                SinusoidalPosEmb(dim),
                tf.keras.layers.Dense(units=time_dim),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dense(units=time_dim)
            ], name="time_embeddings")
        else:
            self.time_mlp = MLP(time_dim)

        # Embedding layers for each metric with adjusted dimensions
        self.obb_embedding = tf.keras.layers.Dense(units=class_emb_dim_obb)
        self.curvature_embedding = tf.keras.layers.Dense(units=class_emb_dim_curvature)
        self.scale_embedding = tf.keras.layers.Dense(units=class_emb_dim_scale)

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)
        now_res = in_res

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),  # Additional block
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity()
            ])
            now_res //= 2 if not is_last else 1

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind == (num_resolutions - 1)
            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),  # Additional block
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()
            ])
            now_res *= 2 if not is_last else 1

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, lambda: default_out_dim)
        self.final_conv = tf.keras.Sequential([
            block_klass(dim * 2, dim),
            tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ], name="output")

    def call(self, x, time=None, class_value=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        
        # Separate class vector into components
        obb, curvature, scale = tf.split(class_value, [self.obb_length, self.curvature_length, self.scale_length], axis=-1)
        
        # Create embeddings for each component
        obb_emb = self.obb_embedding(obb)
        curvature_emb = self.curvature_embedding(curvature)
        scale_emb = self.scale_embedding(scale)
        
        # Concatenate embeddings
        class_emb = tf.concat([obb_emb, curvature_emb, scale_emb], axis=-1)
        
        # Concatenate class embeddings with time embeddings
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

        x = self.final_conv(x)
        return x


# In[4]:


encoded_features.shape


# In[5]:


pc_array = np.load("data/normalized_rotated_point_clouds6.npy")
pc_array.shape


# In[6]:


curvature_coefficients = np.load("average_mean_curvatures.npy")
curvature_coefficients.shape


# In[7]:


def test_forward_noise():
    """
    Tests the forward noise function by applying noise to a 2D image
    and displaying the results at different timesteps.
    """
    import matplotlib.pyplot as plt

    # Generate a random 2D image (e.g., 32x32) as a test input
    test_image = encoded_features[0]
    #np.random.rand(1, 32, 32, 1).astype(np.float32)  # Shape: (1, 32, 32, 1)

    # Define timesteps to visualize
    timesteps_to_test = [0, timesteps // 4, timesteps // 2, timesteps - 1]

    fig, axes = plt.subplots(1, len(timesteps_to_test), figsize=(15, 5))

    for i, t in enumerate(timesteps_to_test):
        t_tensor = tf.convert_to_tensor([t], dtype=tf.int32)

        # Apply forward noise function
        noisy_image, _ = forward_noise(test_image, t_tensor)

        # Convert to numpy if it's a tensor
        if isinstance(noisy_image, tf.Tensor):
            noisy_image = noisy_image.numpy()

        # Remove batch and channel dimensions for visualization
        noisy_image = np.squeeze(noisy_image, axis=(0, -1))

        axes[i].imshow(noisy_image, cmap="gray")
        axes[i].set_title(f"Timestep: {t}")
        axes[i].axis("off")

    plt.show()

# Run the test
test_forward_noise()


# In[8]:


test_image = encoded_features[0]
# Define timesteps to test
timestep_samples = [0, 250, 500, 750, 999]
msd_values = []  # Mean Squared Difference (MSD) tracking
# Define diffusion schedule
endbeta = 0.025
beta = np.linspace(0.0001, endbeta, timesteps)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

# ─── Convert schedule arrays into TF constants ───
# (so gather stays in-graph and on-device)
sqrt_alpha_bar = tf.constant(np.sqrt(alpha_bar), dtype=tf.float32)
one_minus_sqrt_alpha_bar = tf.constant(np.sqrt(1 - alpha_bar), dtype=tf.float32)

# Run forward_noise for different timesteps
noisy_images = []
for t in timestep_samples:
    noisy_img, _ = forward_noise(test_image, np.array([t]))
    noisy_images.append(noisy_img)

    # Compute Mean Squared Difference (MSD)
    msd = np.mean((noisy_img - test_image) ** 2)
    msd_values.append(msd)

# ---- PLOT 1: NOISY IMAGE DEGRADATION ----
fig, axes = plt.subplots(1, len(timestep_samples), figsize=(15, 5))

for i, t in enumerate(timestep_samples):
    img_to_plot = np.squeeze(noisy_images[i], axis=(0, -1))  # Remove batch & channel dimensions

    axes[i].imshow(img_to_plot, cmap="gray")
    axes[i].set_title(f"Timestep: {t}")
    axes[i].axis("off")

plt.suptitle("2D Image Forward Diffusion Process")
plt.show()

# ---- PLOT 2: NOISE MEASUREMENT (MSD) ----
plt.figure(figsize=(7, 5))
plt.plot(timestep_samples, msd_values, marker='o', linestyle='-', color='red', label="MSD")
plt.xlabel("Timestep")
plt.ylabel("Mean Squared Difference (MSD)")
plt.title("MSD vs. Timestep (2D Image Noise Progression)")
plt.legend()
plt.grid()
plt.show()


# In[9]:


feature_size = 32
init_dim = 64  # Increased initial dimension
multend = 8
dim_mults = (1, 2, 4, multend)  # More levels of upscaling and downscaling

unet = Unet_conditional(
    dim=64,  # Increased initial dimension
    init_dim=init_dim,
    out_dim=None,
    dim_mults=(1, 2, 4, 8),  # More levels of upscaling and downscaling
    channels=1,
    resnet_block_groups=4,  # More groups in group normalization
    learned_variance=False,
    sinusoidal_cond_mlp=True,
    class_emb_dim_obb=256,  # Increased embedding dimension for OBB
    class_emb_dim_curvature=64,  # Embedding dimension for curvature
    class_emb_dim_scale=64,  # Embedding dimension for scale
    obb_length=6,  # Specify the length for OBB
    curvature_length=1,  # Specify the length for curvature
    scale_length=1,  # Specify the length for scale
    in_res=feature_size
)


# Example class vector for a single instance
test_class_value = labels[0].reshape(1, -1)
print("test_class_value", test_class_value.shape)

# — CHECKPOINT & RESTORE SETUP —
ckpt = tf.train.Checkpoint(unet=unet)
checkpoint_dir = (
    f"./checkpoints_AMC1_vector_separate_euler_"
    f"{endbeta}beta_{BATCH_SIZE}batch_es_"
    f"{init_dim}dim_{multend}mult_4"
)
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_manager = tf.train.CheckpointManager(
    ckpt,
    checkpoint_dir,
    max_to_keep=3
)
# Try to restore latest checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"✅ Restored from {ckpt_manager.latest_checkpoint}")
else:
    print("⚠️ Initializing from scratch.")



# In[10]:


initial_learning_rate = 1e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)




# In[11]:


@tf.function
def loss_fn(real, generated):
    # bring generated up to real’s dtype (float32) so they match
    generated = tf.cast(generated, real.dtype)
    return tf.reduce_mean(tf.math.squared_difference(real, generated))


@tf.function
def train_step(batch, class_vec):
    # 1) sample timesteps on GPU in one shot
    batch_size = tf.shape(batch)[0]
    t = tf.random.uniform([batch_size], 0, timesteps, dtype=tf.int32)

    # 2) forward pass (forward_noise now uses tf.random.normal)
    noised, noise = forward_noise(batch, t)

    # 3) compute loss & gradients
    with tf.GradientTape() as tape:
        pred = unet(noised, t, class_vec, training=True)
        loss = loss_fn(noise, pred)

    grads = tape.gradient(loss, unet.trainable_variables)
    opt.apply_gradients(zip(grads, unet.trainable_variables))
    return loss



@tf.function
def validation_step(images, class_vec):
    # sample timesteps on GPU
    batch_size = tf.shape(images)[0]
    t = tf.random.uniform([batch_size], 0, timesteps, dtype=tf.int32)

    noised, noise = forward_noise(images, t)
    pred = unet(noised, t, class_vec, training=False)
    return loss_fn(noise, pred)


# In[ ]:


'''
from tensorflow.keras.utils import Progbar
import numpy as np

# Hyperparameters
epochs   = 300
patience = 30

# Track best validation loss
best_val_loss  = float('inf')
no_improvement = 0

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    # Progress bar for training
    steps = tf.data.experimental.cardinality(train_dataset).numpy()
    bar = Progbar(steps, stateful_metrics=['loss'])

    # — TRAIN —
    train_losses = []
    for step, (x_batch, class_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, class_batch)
        train_losses.append(float(loss))
        bar.update(step + 1, [('loss', float(loss))])
    avg_train = np.mean(train_losses)
    print(f" ➜ train loss: {avg_train:.4f}")

    # — VALIDATION —
    val_losses = []
    for x_val, class_val in test_dataset:
        val_losses.append(float(validation_step(x_val, class_val)))
    avg_val = np.mean(val_losses)
    print(f" ➜ val   loss: {avg_val:.4f}")

    # — CHECKPOINT & EARLY STOPPING —
    if avg_val < best_val_loss:
        best_val_loss  = avg_val
        no_improvement = 0
        ckpt_manager.save(checkpoint_number=epoch)
        print(f" ✓ checkpoint saved (val_loss {avg_val:.4f})")
    else:
        no_improvement += 1
        print(f" ✗ no improvement for {no_improvement} epochs")
        if no_improvement >= patience:
            print("⏹️ Early stopping")
            break

# — SAVE FINAL WEIGHTS —
final_weights_path = os.path.join(checkpoint_dir, "unet_final.weights.h5")
unet.save_weights(final_weights_path)
print(f"✅ Final weights saved to {final_weights_path}")
'''


# In[ ]:





# In[ ]:





# In[12]:


# Rebuild the same model
unet = Unet_conditional(
    dim=64,  # Increased initial dimension
    init_dim=init_dim,
    out_dim=None,
    dim_mults=(1, 2, 4, 8),  # More levels of upscaling and downscaling
    channels=1,
    resnet_block_groups=4,  # More groups in group normalization
    learned_variance=False,
    sinusoidal_cond_mlp=True,
    class_emb_dim_obb=256,  # Increased embedding dimension for OBB
    class_emb_dim_curvature=64,  # Embedding dimension for curvature
    class_emb_dim_scale=64,  # Embedding dimension for scale
    obb_length=6,  # Specify the length for OBB
    curvature_length=1,  # Specify the length for curvature
    scale_length=1,  # Specify the length for scale
    in_res=feature_size
)

# Build the model with dummy inputs

dummy_x   = tf.zeros((1, feature_size, feature_size, 1), dtype=tf.float32)
dummy_t   = tf.zeros((1,), dtype=tf.int32)                        # ← int32 here
dummy_cls = tf.zeros((1, 6 + 1 + 1), dtype=tf.float32)

_ = unet(dummy_x, dummy_t, dummy_cls)

                     
# Load weights
checkpoint_dir = (
    f"./checkpoints_AMC1_vector_separate_euler_"
    f"{endbeta}beta_{BATCH_SIZE}batch_es_"
    f"{init_dim}dim_{multend}mult_4"
)
weights_path = os.path.join(checkpoint_dir, "unet_final.weights.h5")

unet.load_weights(weights_path)

print("✅ Weights loaded successfully!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# after
timesteps   = 1000
endbeta     = 0.025
beta        = np.linspace(0.0001, endbeta,   timesteps)
alpha       = 1.0 - beta
alpha_bar   = np.cumprod(alpha)

# ─── Convert all schedules into TF constants ───
alpha_tf     = tf.constant(alpha,     dtype=tf.float32)
alpha_bar_tf = tf.constant(alpha_bar, dtype=tf.float32)
beta_tf      = tf.constant(beta,      dtype=tf.float32)

# (you can still keep these  




# Ensure the directory exists
def save_images(img_list, path=""):
    os.makedirs(path, exist_ok=True)
    for idx, im in enumerate(img_list):
        im_processed = postprocess(im)
        file_path = os.path.join(path, f"image_{idx}.npy")
        np.save(file_path, im_processed)

# Denoising step
@tf.function
def ddpm(x_t, pred_noise, t):
    # ensure t is a 1D int32 tensor
    pred_noise = tf.cast(pred_noise, x_t.dtype)
    t = tf.cast(tf.reshape(t, [-1]), tf.int32)

    # gather schedules and cast them to match x_t.dtype
    a_t     = tf.cast(tf.gather(alpha_tf,     t), x_t.dtype)
    a_bar_t = tf.cast(tf.gather(alpha_bar_tf, t), x_t.dtype)
    b_t     = tf.cast(tf.gather(beta_tf,      t), x_t.dtype)

    # posterior mean
    # compute everything in x_t.dtype
    one = tf.constant(1.0, dtype=x_t.dtype)
    eps_coef = (one - a_t) / tf.sqrt(one - a_bar_t)
    mean     = (one / tf.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
    # posterior variance & noise sampling
    noise    = tf.random.normal(tf.shape(x_t), dtype=x_t.dtype)
    return mean + tf.sqrt(b_t) * noise


# Save feature vector output
def save_feature_vector(feature_vector, path, index, curv_value):
    os.makedirs(path, exist_ok=True)
    feature_vector_processed = postprocess(feature_vector)
    feature_vector_processed = np.squeeze(feature_vector_processed)
    if len(feature_vector_processed.shape) == 3:
        feature_vector_processed = feature_vector_processed[0]
    file_path = os.path.join(path, f"encoded_vector_idx{index}_curvs{curv_value:.2f}.npy")
    np.save(file_path, feature_vector_processed)

# Load data
obb_vectors = np.load("obb_vectors_open3d_euler.npy")
curvature_coefficients = np.load("average_mean_curvatures.npy")
scale_coefficients = np.load("normalized_scale_coefficients6.npy")

# Normalize data
min_angle = obb_vectors[:, :3].min(axis=0)
max_angle = obb_vectors[:, :3].max(axis=0)
min_dim = obb_vectors[:, 3:].min(axis=0)
max_dim = obb_vectors[:, 3:].max(axis=0)
labels = preprocess_obb_np(obb_vectors)

normalized_curv = (curvature_coefficients - curvature_coefficients.min()) / (curvature_coefficients.max() - curvature_coefficients.min())
normalized_coefficients = (scale_coefficients - scale_coefficients.min()) / (scale_coefficients.max() - scale_coefficients.min())

# Append curvature and scale
labels = np.hstack((labels, normalized_curv.reshape(-1, 1), normalized_coefficients.reshape(-1, 1)))

# Parameters
random_indices = random.sample(range(1, 11893), 4)
curvature_values = [0.0, 0.50, 0.75, 1.0]

# Main loop
# ─── Main loop (fixed) ───
for index in random_indices:
    for curv_val in curvature_values:
        # 1) Prepare the conditioning vector (float32)
        _class = labels[index].copy()
        _class[-2] = curv_val                       # curvature
        _class[-1] = normalized_coefficients[index] #curv_val # scale (keep correct value)
        _class = _class.reshape(1, -1).astype(np.float32)

        # 2) Start from pure Gaussian noise (float32)
        x = tf.random.normal((1, feature_size, feature_size, 1),
                             dtype=tf.float32)
        img_list = []
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))
        
                # 3) 1,000 DDPM steps, no per-step numpy() or plotting
        for i in tqdm(range(timesteps - 1),
                      desc=f"Index {index}, Curv {curv_val:.2f}"):
            t = tf.constant([timesteps - i - 1], dtype=tf.int32)
            # inference mode
            pred_noise = unet(x, t, _class, training=False)
            x = ddpm(x, pred_noise, t)
            img_list.append(np.squeeze(x.numpy(), axis=(0, -1)))  # Adjust dimensions appropriately

            if i % 25 == 0:
                plt.imshow(img_list[-1], cmap='coolwarm')
                #plt.show()
                
        # 4) Postprocess + save *once* at the end
        #final_vec = postprocess(x[0, ..., 0].numpy())
        save_path = "./GGenerated_encoded_feature_vectors_CURVS2"
        save_feature_vector(np.squeeze(x.numpy(), axis=(0, -1)), save_path, index, curv_val)
        print(f"✅ Saved idx {index}, curv {curv_val:.2f}")
'''


# In[14]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# ------------------------------------------------------------
# ASSUMES:
#   - unet (weights already loaded)
#   - feature_size
#   - postprocess(x)
# ------------------------------------------------------------

# ------------------------------------------------------------
# DDPM schedule
# ------------------------------------------------------------
timesteps = 1000
endbeta   = 0.025

beta      = np.linspace(0.0001, endbeta, timesteps).astype(np.float32)
alpha     = (1.0 - beta).astype(np.float32)
alpha_bar = np.cumprod(alpha).astype(np.float32)

alpha_tf     = tf.constant(alpha,     dtype=tf.float32)
alpha_bar_tf = tf.constant(alpha_bar, dtype=tf.float32)
beta_tf      = tf.constant(beta,      dtype=tf.float32)


@tf.function
def ddpm(x_t, pred_noise, t):
    """
    x_t: (B,H,W,1)
    pred_noise: (B,H,W,1)
    t: (B,) int32
    """
    pred_noise = tf.cast(pred_noise, x_t.dtype)
    t = tf.cast(tf.reshape(t, [-1]), tf.int32)  # (B,)

    # Gather per-batch scalars, then reshape to broadcast over (H,W,C)
    a_t     = tf.gather(alpha_tf,     t)  # (B,)
    a_bar_t = tf.gather(alpha_bar_tf, t)  # (B,)
    b_t     = tf.gather(beta_tf,      t)  # (B,)

    a_t     = tf.cast(a_t,     x_t.dtype)[:, None, None, None]
    a_bar_t = tf.cast(a_bar_t, x_t.dtype)[:, None, None, None]
    b_t     = tf.cast(b_t,     x_t.dtype)[:, None, None, None]

    one = tf.constant(1.0, dtype=x_t.dtype)

    eps_coef = (one - a_t) / tf.sqrt(one - a_bar_t)
    mean     = (one / tf.sqrt(a_t)) * (x_t - eps_coef * pred_noise)

    noise = tf.random.normal(tf.shape(x_t), dtype=x_t.dtype)
    return mean + tf.sqrt(b_t) * noise


# ------------------------------------------------------------
# FULL GPU REVERSE DIFFUSION (batched)
# ------------------------------------------------------------
@tf.function(jit_compile=True)
def ddpm_sample(x, class_vec):

    batch_size = tf.shape(x)[0]

    def cond(i, x):
        return i >= 0

    def body(i, x):
        t = tf.fill([batch_size], tf.cast(i, tf.int32))
        pred_noise = unet(x, t, class_vec, training=False)
        x = ddpm(x, pred_noise, t)
        return i - 1, x

    i0 = tf.constant(timesteps - 1, dtype=tf.int32)

    _, x = tf.while_loop(
        cond,
        body,
        loop_vars=[i0, x],
        shape_invariants=[
            i0.get_shape(),
            tf.TensorShape([None, feature_size, feature_size, 1])
        ]
    )

    return tf.cast(x, tf.float32)



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def save_feature_vector_npy(feature_vector, out_dir, graph_id, node_id, curv_value, scale_value):

    graph_dir = os.path.join(out_dir, f"graph_{graph_id:03d}")
    os.makedirs(graph_dir, exist_ok=True)

    fv = postprocess(feature_vector)
    fv = np.squeeze(fv)

    fname = f"node_{node_id:03d}_curv{curv_value:.4f}_scale{scale_value:.6f}.npy"

    np.save(os.path.join(graph_dir, fname), fv)


def load_training_normalizers():
    obb_vectors = np.load("obb_vectors_open3d_euler.npy")
    curvature_coefficients = np.load("average_mean_curvatures.npy")
    scale_coefficients = np.load("normalized_scale_coefficients6.npy")

    min_dim = obb_vectors[:, 3:].min(axis=0)
    max_dim = obb_vectors[:, 3:].max(axis=0)
    dim_den = np.maximum(max_dim - min_dim, 1e-12)

    curv_min = float(curvature_coefficients.min())
    curv_max = float(curvature_coefficients.max())
    curv_den = max(curv_max - curv_min, 1e-12)

    scale_min = float(scale_coefficients.min())
    scale_max = float(scale_coefficients.max())
    scale_den = max(scale_max - scale_min, 1e-12)

    return min_dim, dim_den, curv_min, curv_den, scale_min, scale_den


def preprocess_obb_np(obb_vec, min_dim, dim_den):
    angles = obb_vec[:3]
    dims   = (obb_vec[3:] - min_dim) / dim_den
    return np.concatenate([angles, dims], axis=0)


def make_node_map_first_k(subgraphs, k=32):
    node_map = []
    for g_id, sg in enumerate(subgraphs[:k]):
        n = sg["obb_euler"].shape[0]
        for node_id in range(n):
            node_map.append((g_id, node_id))
    return node_map


# ------------------------------------------------------------
# MAIN GENERATION
# ------------------------------------------------------------
def generate_for_first_32_graphs(
    subgraph_npz_path="UNPERTURBED_center_subgraphs450_8.npz",
    out_dir_nodes="./DDPM_GEN_nodes_100",
    out_dir_graph_npz="./DDPM_GEN_graph_npz_116",
    k_graphs=116,
    sweep_curvatures=None
):

    os.makedirs(out_dir_nodes, exist_ok=True)
    os.makedirs(out_dir_graph_npz, exist_ok=True)

    data = np.load(subgraph_npz_path, allow_pickle=True)
    subgraphs = data["subgraphs"]

    k = min(k_graphs, len(subgraphs))
    min_dim, dim_den, curv_min, curv_den, scale_min, scale_den = load_training_normalizers()

    node_map = make_node_map_first_k(subgraphs, k=k)

    per_graph_features = {g: [] for g in range(k)}
    per_graph_node_ids = {g: [] for g in range(k)}
    per_graph_labels   = {g: [] for g in range(k)}

    batch_size_nodes = 64   # adjust to GPU memory
    
    for start in range(0, len(node_map), batch_size_nodes):
    
        batch_pairs = node_map[start:start + batch_size_nodes]
    
        batch_classes = []
        meta = []
    
        for (g_id, node_id) in batch_pairs:
    
            sg = subgraphs[g_id]
    
            obb   = sg["obb_euler"][node_id].astype(np.float32)
            curv  = float(sg["curvatures"][node_id])
            scale = float(sg["scales"][node_id])
    
            obb6 = preprocess_obb_np(obb, min_dim=min_dim, dim_den=dim_den)
            curv_norm  = (curv  - curv_min)  / curv_den
            scale_norm = (scale - scale_min) / scale_den
    
            if sweep_curvatures is None:
                sweep_list = [curv_norm]
            else:
                sweep_list = list(sweep_curvatures)
    
            for curv_used in sweep_list:
    
                class_vec = np.concatenate(
                    [obb6, [float(curv_used)], [scale_norm]],
                    axis=0
                ).astype(np.float32)
    
                batch_classes.append(class_vec)
                meta.append((g_id, node_id, curv, scale))
    
        batch_classes = tf.convert_to_tensor(batch_classes, dtype=tf.float32)
    
        # --------------------------------------------------
        # GPU sampling (BIG SPEEDUP happens here)
        # --------------------------------------------------
    
        x = tf.random.normal(
            (len(batch_classes), feature_size, feature_size, 1),
            dtype=tf.float32
        )
    
        x = ddpm_sample(x, batch_classes)
    
        gen_batch = x.numpy()
    
        # --------------------------------------------------
        # Save results + bookkeeping
        # --------------------------------------------------
    
        labels_np = batch_classes.numpy()
        for i, (g_id, node_id, curv, scale) in enumerate(meta):
    
            gen = np.squeeze(gen_batch[i], axis=-1)
    
            save_feature_vector_npy(
                gen,
                out_dir_nodes,
                g_id,
                node_id,
                curv_value=curv,
                scale_value=scale
            )
    
            per_graph_features[g_id].append(gen.astype(np.float32))
            per_graph_node_ids[g_id].append(node_id)
            per_graph_labels[g_id].append(labels_np[i])
    
            print(f"Saved: graph {g_id}, node {node_id}")



    # Save NPZ per graph
    for g_id in range(k):
        feats = np.stack(per_graph_features[g_id], axis=0)
        node_ids = np.array(per_graph_node_ids[g_id], dtype=np.int32)
        labels = np.stack(per_graph_labels[g_id], axis=0)

        out_path = os.path.join(out_dir_graph_npz, f"graph_{g_id:03d}_ddpm_generated.npz")
        np.savez(
            out_path,
            features=feats,
            node_ids=node_ids,
            labels=labels,
            norm_min_dim=min_dim,
            norm_dim_den=dim_den,
            norm_curv_min=curv_min,
            norm_curv_den=curv_den,
            norm_scale_min=scale_min,
            norm_scale_den=scale_den,
        )

    print("Done.")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_for_first_32_graphs(
        subgraph_npz_path="UNPERTURBED_center_subgraphs450_8.npz",
        out_dir_nodes="./DDPM_GEN_nodes_100",
        out_dir_graph_npz="./DDPM_GEN_graph_npz_100",
        k_graphs=100,
        sweep_curvatures=None
    )


# In[ ]:





# In[ ]:


len(sweep_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## Point Cloud Autoencoder Decoding


# In[13]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
#improved with residual 
import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import os
import shutil

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
#import model
import torch.optim as optim



class ReadDataset(Dataset):
    def __init__(self, source):
        self.data = torch.from_numpy(source).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets) * train_set_percentage), len(datasets) - int(len(datasets) * train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(npArray, batch_size, train_set_percentage=0.9, shuffle=True, num_workers=0, pin_memory=True):
    pc = ReadDataset(npArray)
    train_set, test_set = RandomSplit(pc, train_set_percentage)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, train_set, test_set


    


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Key, query, value projections for the attention
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Batch, Channels, Points
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Calculate the attention scores
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)

        # Apply attention to the values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out += x  # Residual connection
        return out
        
class FeatureFusion(nn.Module):
    def __init__(self, in_channels_xyz, in_channels_normal, out_channels):
        super(FeatureFusion, self).__init__()
        self.fusion_conv = nn.Conv1d(in_channels_xyz + in_channels_normal, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, xyz_features, normal_features):
        # Concatenate features along channel dimension
        combined_features = torch.cat((xyz_features, normal_features), 1)
        # Fuse combined features
        fused_features = F.relu(self.bn(self.fusion_conv(combined_features)))
        return fused_features

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        feature_size = latent_size ** 2

        # XYZ Path
        self.conv1_xyz = nn.Conv1d(3, 64, 1)
        self.conv2_xyz = nn.Conv1d(64, 128, 1)
        self.conv3_xyz = nn.Conv1d(128, feature_size // 2, 1)
        self.bn1_xyz = nn.BatchNorm1d(64)
        self.bn2_xyz = nn.BatchNorm1d(128)
        self.bn3_xyz = nn.BatchNorm1d(feature_size // 2)

        # Normals Path
        self.conv1_normal = nn.Conv1d(3, 32, 1)
        self.conv2_normal = nn.Conv1d(32, 64, 1)
        self.conv3_normal = nn.Conv1d(64, feature_size // 2, 1)
        self.bn1_normal = nn.BatchNorm1d(32)
        self.bn2_normal = nn.BatchNorm1d(64)
        self.bn3_normal = nn.BatchNorm1d(feature_size // 2)

        # Feature Fusion
        self.feature_fusion = FeatureFusion(feature_size // 2, feature_size // 2, feature_size)

        # Self-Attention on fused features
        self.self_attention = SelfAttention(feature_size)

        # Decoder
        self.fc1 = nn.Linear(feature_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, point_size * 6)
        
        # Residual connections
        self.fc_res1 = nn.Linear(1024, 2048)
        self.fc_res2 = nn.Linear(2048, point_size * 6)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def encoder(self, x):
        xyz, normals = torch.split(x, 3, 1)

        # XYZ Path
        xyz = self.leaky_relu(self.bn1_xyz(self.conv1_xyz(xyz)))
        xyz = self.leaky_relu(self.bn2_xyz(self.conv2_xyz(xyz)))
        xyz = self.leaky_relu(self.bn3_xyz(self.conv3_xyz(xyz)))

        # Normals Path
        normals = self.leaky_relu(self.bn1_normal(self.conv1_normal(normals)))
        normals = self.leaky_relu(self.bn2_normal(self.conv2_normal(normals)))
        normals = self.leaky_relu(self.bn3_normal(self.conv3_normal(normals)))

        # Feature Fusion
        fused_features = self.feature_fusion(xyz, normals)

        # Self-Attention
        x = self.self_attention(fused_features)

        # Pooling and Reshaping
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(-1, self.latent_size, self.latent_size)
        return x

    def decoder(self, x):
        x = x.view(-1, self.latent_size ** 2)
        x = self.leaky_relu(self.fc1(x))
        res1 = x  # First residual connection
        x = self.leaky_relu(self.fc2(x))
        x = x + self.fc_res1(res1)  # Apply first residual connection
        res2 = x  # Second residual connection
        x = self.fc3(x)
        x = x + self.fc_res2(res2)  # Apply second residual connection
        return x.view(-1, self.point_size, 6)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        

        




batch_size = 16
save_results = True # save the results to output_folder
#use_GPU = True # use GPU, False to use CPU
latent_size = 32 # bottleneck size of the Autoencoder model

#from Dataloaders import GetDataLoaders

pc_array = np.load("data/normalized_rotated_point_clouds6.npy")
print(pc_array.shape)
pc_array_verify = pc_array[11000:11010,:,:]
#pc_array = pc_array[:1000,:,:]
print(pc_array.shape)

# load dataset from numpy array and divide 90%-10% randomly for train and test sets
train_loader, test_loader, train_set, test_set = GetDataLoaders(npArray=pc_array, batch_size=batch_size)

# Assuming all models have the same size, get the point size from the first model
point_size = len(train_loader.dataset[0])
print(point_size)
print(latent_size)



# Select a fixed set of samples from the test set for plotting
#fixed_test_samples = next(iter(DataLoader(test_set, batch_size=10, shuffle=False)))


# Create a DataLoader directly from the desired range of pc_array
fixed_test_samples_loader = DataLoader(ReadDataset(pc_array_verify[:10,:,:]), batch_size=10, shuffle=False)
fixed_test_samples = next(iter(fixed_test_samples_loader))


# Ensure to convert `fixed_test_samples` to the same device as your model (e.g., GPU or CPU)



model = PointCloudAE(point_size, latent_size)  # Make sure these are defined correctly
model_path = "Pointautoencoder_Fusion_aniso_nodropout_10600_32_CURVATURE/saved_model.pth"

# Load the saved model's state dictionary
model.load_state_dict(torch.load(model_path))

# Switch the model to evaluation mode
model.eval()


#PLOT A SPECIFC PLOT AND RECONSTRUCTION SIDE BY SIDE
import torch

# Assuming `net` and `test_loader` have been correctly set up and initialized
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load a single test sample
#single_sample = pc_array_verify[-2]  # Assuming this is your data
single_sample = pc_array[4286]  # Assuming this is your data	#4286

print("single_sample", single_sample.shape)

# Convert the NumPy array to a PyTorch tensor, reshape it to add a batch dimension, and move to the specified device
single_sample = torch.tensor(single_sample, dtype=torch.float32).unsqueeze(0).to(device)  # Adds a batch dimension
print("single_sample", single_sample.shape)

# Perform encoding and decoding
with torch.no_grad():
    encoded = model.encoder(single_sample.permute(0, 2, 1))  # Permute to (batch_size, dimensions, num_points)
    decoded = model.decoder(encoded).view(-1, single_sample.shape[1], 6)  # Reshape decoded output to match input dimensions

print("encoded",encoded.shape)
print("decoded",decoded.shape)
# Extract the original and decoded point clouds for plotting
original_pc = single_sample[0, :, :3].cpu().numpy()  # Extract only the first three coordinates
reconstructed_pc = decoded[0, :, :3].cpu().numpy()
print("reconstructed_pc",reconstructed_pc.shape)
# Plot the point clouds
#plot_point_clouds(original_pc, reconstructed_pc)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons

def plot_overlay(original_pc, reconstructed_pc, title='Point Cloud Comparison', figsize=(12, 6)):
    fig = plt.figure(figsize=figsize)
    
    # Create two subplot axes for original and reconstructed point clouds
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the original point cloud
    p1 = ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], c='#0077b6', marker='.', alpha=0.7, s=10, label='Original')
    ax1.set_title('Original')
    ax1.set_xlim([-0.6, 0.6])
    ax1.set_ylim([-0.6, 0.6])
    ax1.set_zlim([-0.6, 0.6])
    ax1.set_axis_off()

    # Plot the reconstructed point cloud
    p2 = ax2.scatter(reconstructed_pc[:, 0], reconstructed_pc[:, 1], reconstructed_pc[:, 2], c='#ff7f50', marker='.', alpha=0.7, s=10, label='Reconstructed')
    ax2.set_title('Reconstructed')
    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([-0.6, 0.6])
    ax2.set_zlim([-0.6, 0.6])
    ax2.set_axis_off()

    # Function to update rotation
    def update_rotation(num):
        ax1.view_init(azim=num)
        ax2.view_init(azim=num)

    # Animation object to rotate both plots
    ani = FuncAnimation(fig, update_rotation, frames=range(0, 360, 2), interval=50, repeat=True)

    plt.show()

# Example usage
# original_pc and reconstructed_pc should be numpy arrays of shape (n_points, 3)
# original_pc = np.random.rand(100, 3) - 0.5
# reconstructed_pc = np.random.rand(100, 3) - 0.5
# plot_overlay(original_pc, reconstructed_pc)


# Example usage with your data
# plot_overlay(original_pc, reconstructed_pc)


import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_reconstructed(reconstructed_pc, title='Reconstructed Point Cloud', figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Convert tensor to numpy array after moving to CPU
    if isinstance(reconstructed_pc, torch.Tensor):
        reconstructed_pc = reconstructed_pc.cpu().detach().numpy()

    # Plot reconstructed point cloud
    p2 = ax.scatter(reconstructed_pc[:, 0], reconstructed_pc[:, 1], reconstructed_pc[:, 2], c='#ff7f50', marker='.', alpha=0.7, s=10, label='Reconstructed')

    # Setting up the plot aesthetics
    ax.set_title(title)
    ax.set_xlim([-0.9, 0.9])
    ax.set_ylim([-0.9, 0.9])
    ax.set_zlim([-0.9, 0.9])
    ax.set_axis_off()
    ax.legend()

    # Adding toggle button for rotation
    ax_check = plt.axes([0.8, 0.4, 0.1, 0.15])
    check = CheckButtons(ax_check, ['Rotate'], [False])

    # Animation function to rotate the plot
    def update_rotation(num):
        ax.view_init(azim=num)

    ani = None

    def toggle_visibility(label):
        nonlocal ani
        if label == 'Rotate':
            if ani is None:
                ani = FuncAnimation(fig, update_rotation, frames=range(0, 360, 2), interval=50, repeat=True)
            else:
                ani.event_source.stop()
                ani = None
        plt.draw()

    check.on_clicked(toggle_visibility)

    plt.show()


# Usage example, assuming reconstructed_pc is defined:
# plot_reconstructed(reconstructed_pc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



INDEX = 11786
gifline = f'GGenerated_encoded_feature_vectors'
#single_sample = pc_array[2]  # Assuming this is your data
encoded_feature_vector = np.load(gifline+"/encoded_feature_vector_0001.npy")

encoded_feature_vector = torch.tensor(encoded_feature_vector, dtype=torch.float32).unsqueeze(0).to(device)  # Adjust the path as necessary
print(encoded_feature_vector.shape)
# Perform encoding and decoding

# Decode the tensor
with torch.no_grad():
    decoded_output = model.decoder(encoded_feature_vector).view(-1, model.point_size, 6)  # Reshape decoded output to match input dimensions


torch.cuda.empty_cache()


# Assuming `model.point_size` corresponds to `single_sample.shape[1]`
reconstructed_EFV = decoded_output[0, :, :3].cpu().numpy()  # Extract the spatial coordinates from the first sample in the batch



#######plot_reconstructed(reconstructed_EFV)




import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons

def plot_point_cloud_with_obb_open3d(points, obb_vector):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='tab:orange')

    # Extract OBB properties
    euler_angles = obb_vector[:3]
    extent = obb_vector[3:6]

    # Convert Euler angles to rotation matrix
    r = R.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()

    # Define half-dimensions
    half_extent = extent / 2

    # Define the corners of the OBB in the local coordinate system
    local_corners = np.array([
        [-half_extent[0], -half_extent[1], -half_extent[2]],
        [-half_extent[0], -half_extent[1],  half_extent[2]],
        [-half_extent[0],  half_extent[1], -half_extent[2]],
        [-half_extent[0],  half_extent[1],  half_extent[2]],
        [ half_extent[0], -half_extent[1], -half_extent[2]],
        [ half_extent[0], -half_extent[1],  half_extent[2]],
        [ half_extent[0],  half_extent[1], -half_extent[2]],
        [ half_extent[0],  half_extent[1],  half_extent[2]]
    ])

    # Transform the corners to the global coordinate system
    global_corners = local_corners @ rotation_matrix.T

    # Define the edges of the OBB
    edges = [
        [global_corners[0], global_corners[1]], [global_corners[0], global_corners[2]], [global_corners[0], global_corners[4]],
        [global_corners[1], global_corners[3]], [global_corners[1], global_corners[5]],
        [global_corners[2], global_corners[3]], [global_corners[2], global_corners[6]],
        [global_corners[3], global_corners[7]],
        [global_corners[4], global_corners[5]], [global_corners[4], global_corners[6]],
        [global_corners[5], global_corners[7]],
        [global_corners[6], global_corners[7]]
    ]

    # Plot OBB as a wireframe
    for edge in edges:
        ax.plot3D(*zip(*edge), color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-0.9, 0.9)
    ax.set_ylim3d(-0.9, 0.9)
    ax.set_zlim3d(-0.9, 0.9)

    plt.show()


# Example class vector with the specified length
_class = np.array([[-4.39259899e+01,  3.25698778e+01,  6.24181127e+01, 7.11718472e-01,  3.67293177e-01,  2.66415934e-01, 1.00000000e-04]], dtype=np.float32)
_class =  _class.reshape(-1,1)[:,-1]



#AUGMENTED DATASET FOR THE DDPM VS THE GRAPHRNN
obb_vectors = np.load("obb_vectors_open3d_euler.npy")
scale_coefficients = np.load("mean_curvatures.npy")

min_dim = obb_vectors[:, 3:].min(axis=0)
max_dim = obb_vectors[:, 3:].max(axis=0)

def preprocess_obb_vectors(obb_vectors):
    angles_normalized = obb_vectors[:, :3]  # Assuming quaternion angles are already normalized
    dimensions_normalized = (obb_vectors[:, 3:] - min_dim) / (max_dim - min_dim)
    obb_normalized = np.concatenate([angles_normalized, dimensions_normalized], axis=1)
    return obb_normalized
min_angle = obb_vectors[:, :3].min(axis=0)
max_angle = obb_vectors[:, :3].max(axis=0)
min_dim = obb_vectors[:, 3:].min(axis=0)
max_dim = obb_vectors[:, 3:].max(axis=0)

labels = obb_vectors


_class = labels[1]
print("_class", _class)
#_class = preprocess_obb_vectors(_class)







import string

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R
import torch

# Function to plot the oriented bounding box (OBB)
def plot_obb(ax, obb_vector):
    # Extract OBB properties
    euler_angles = obb_vector[:3]
    extent = obb_vector[3:6]

    # Convert Euler angles to rotation matrix
    r = R.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()

    # Define half-dimensions
    half_extent = extent / 2

    # Define the corners of the OBB in the local coordinate system
    local_corners = np.array([
        [-half_extent[0], -half_extent[1], -half_extent[2]],
        [-half_extent[0], -half_extent[1],  half_extent[2]],
        [-half_extent[0],  half_extent[1], -half_extent[2]],
        [-half_extent[0],  half_extent[1],  half_extent[2]],
        [ half_extent[0], -half_extent[1], -half_extent[2]],
        [ half_extent[0], -half_extent[1],  half_extent[2]],
        [ half_extent[0],  half_extent[1], -half_extent[2]],
        [ half_extent[0],  half_extent[1],  half_extent[2]]
    ])

    # Transform the corners to the global coordinate system
    global_corners = local_corners @ rotation_matrix.T

    # Define the edges of the OBB
    edges = [
        [global_corners[0], global_corners[1]], [global_corners[0], global_corners[2]], [global_corners[0], global_corners[4]],
        [global_corners[1], global_corners[3]], [global_corners[1], global_corners[5]],
        [global_corners[2], global_corners[3]], [global_corners[2], global_corners[6]],
        [global_corners[3], global_corners[7]],
        [global_corners[4], global_corners[5]], [global_corners[4], global_corners[6]],
        [global_corners[5], global_corners[7]],
        [global_corners[6], global_corners[7]]
    ]

    # Plot OBB as a wireframe
    for edge in edges:
        ax.plot3D(*zip(*edge), color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z',labelpad=5)
    #ax.set_title('Oriented Bounding Box', fontsize=10)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string

def plot_encoded_feature_and_decoded_pc(fig, gs, encoded_feature, decoded_pc, obb_vector, row_idx=0):
    # LEFT: Plot OBB
    ax1 = fig.add_subplot(gs[row_idx, 0], projection='3d')
    plot_obb(ax1, obb_vector)
    ax1.set_xlabel('X', fontsize=20, labelpad=10, weight='bold')
    ax1.set_ylabel('Y', fontsize=20, labelpad=10, weight='bold')
    ax1.set_zlabel('Z', fontsize=20, labelpad=10, weight='bold')
    ax1.tick_params(labelsize=16)
    ax1.set_box_aspect([1, 1, 1])

    # Row label (a, b, c)
    ax1.text2D(-0.15, 0.5, f"({string.ascii_lowercase[row_idx]})", transform=ax1.transAxes,
               fontsize=22, weight='bold', ha='center', va='center')

    # CENTER: Encoded feature image with proper grid and centered colorbar
    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.axis('off')  # Hide outer axes

    # Inset for the encoded feature image
    inset_ax = inset_axes(ax2, width="80%", height="80%", loc='center')
    h, w = encoded_feature.shape
    shrink_ratio = 0.8  # Controls visual size inside the inset

    # Calculate grid edges for pcolormesh
    x0 = (1 - shrink_ratio) * w / 2
    x1 = x0 + shrink_ratio * w
    y0 = (1 - shrink_ratio) * h / 2
    y1 = y0 + shrink_ratio * h
    x = np.linspace(x0, x1, w + 1)
    y = np.linspace(y0, y1, h + 1)

    # Use pcolormesh for aligned grid and visible cell outlines
    img = inset_ax.pcolormesh(x, y, encoded_feature, cmap='coolwarm',
                              edgecolors='black', linewidth=0.2, shading='flat')
    inset_ax.set_xlim(x0, x1)
    inset_ax.set_ylim(y0, y1)
    inset_ax.set_aspect('equal')
    inset_ax.axis('off')

    # Colorbar, centered and truncated beneath the image
    cbar = fig.colorbar(img, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)
    vmin, vmax = encoded_feature.min(), encoded_feature.max()
    vmid = (vmin + vmax) / 2
    cbar.set_ticks([vmin, vmid, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmid:.2f}", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=14)

    # Truncate and center the colorbar (80% width)
    pos = cbar.ax.get_position()
    new_width = pos.width * 0.8
    new_x = pos.x0 + (pos.width - new_width) / 2
    cbar.ax.set_position([new_x, pos.y0, new_width, pos.height])


    # RIGHT: Reconstructed point cloud
    ax3 = fig.add_subplot(gs[row_idx, 2], projection='3d')
    ax3.scatter(decoded_pc[:, 0], decoded_pc[:, 1], decoded_pc[:, 2], s=4, c='darkorange')
    plot_obb(ax3, obb_vector)
    ax3.set_xlabel('X', fontsize=20, labelpad=10, weight='bold')
    ax3.set_ylabel('Y', fontsize=20, labelpad=10, weight='bold')
    ax3.set_zlabel('Z', fontsize=20, labelpad=10, weight='bold')
    ax3.tick_params(labelsize=16)
    ax3.set_box_aspect([1, 1, 1])


# === MAIN BLOCK ===
INDEX_VALUES = [4000, 100, 0]
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(len(INDEX_VALUES), 3, height_ratios=[1]*len(INDEX_VALUES))

# Column Titles (once, across top row)
column_titles = ["Conditioning Input (OBB)", "Denoised Latent", "Generated Point Cloud"]
for col, title in enumerate(column_titles):
    ax_title = fig.add_subplot(gs[0, col])
    ax_title.set_title(title, fontsize=20, weight='bold', pad=20)
    ax_title.axis('off')

# Plot each row
for row_idx, index in enumerate(INDEX_VALUES):
    gifline = f'vector_saved_models_EULER_NODE_new_node_gif_images_{index}'
    encoded_feature_vector = np.load(gifline + "/image_999.npy")

    # Ensure it's 2D
    if encoded_feature_vector.ndim == 3:
        encoded_feature_vector = encoded_feature_vector[0]
    elif encoded_feature_vector.ndim != 2:
        raise ValueError(f"Unexpected shape: {encoded_feature_vector.shape}")

    obb_vectors = np.load("obb_vectors_open3d_euler.npy")
    obb_vector = obb_vectors[index]

    # Decode point cloud
    encoded_tensor = torch.tensor(encoded_feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        decoded_output = model.decoder(encoded_tensor)
    reconstructed_pc = decoded_output.detach().view(-1, model.point_size, 6).cpu().numpy()[0, :, :3]

    # Plot row
    plot_encoded_feature_and_decoded_pc(fig, gs, encoded_feature_vector, reconstructed_pc, obb_vector, row_idx=row_idx)

# Final layout
# Final layout: add manual spacing, no tight_layout to avoid Z-axis clipping
plt.subplots_adjust(
    left=0.07, right=0.94,    # extra room for 3D axis labels
    top=0.94, bottom=0.06,
    wspace=0.25, hspace=0.01   # more spacing between subplots
)

plt.savefig("Encoded_OBB_Reconstruction.png", dpi=300)
plt.close(fig)



import os, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def plot_pc_with_obb(ax, pc, obb_vec):
    """(Unused now) Scatter + OBB wireframe."""
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=2, c='darkorange')
    _draw_obb(ax, obb_vec*0.8)
    _set_ax_off_and_limits(ax)

def plot_pc_only(ax, pc):
    """Scatter the point‐cloud only."""
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=2, c='darkorange')
    _set_ax_off_and_limits(ax)

def plot_obb_only(ax, obb_vec):
    """Draw only the OBB wireframe."""
    _draw_obb(ax, obb_vec)
    _set_ax_off_and_limits(ax)

def _draw_obb(ax, obb_vec):
    """Helper: compute & draw the 12 edges of an OBB."""
    euler  = np.asarray(obb_vec[:3], float)
    extent = 0.9*np.asarray(obb_vec[3:6], float)
    Rmat = R.from_euler('xyz', euler, degrees=True).as_matrix()
    half = extent / 2
    local = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]) * half
    corners = local @ Rmat.T
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
             (3,7),(4,5),(4,6),(5,7),(6,7)]
    for i,j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color='r')

def _set_ax_off_and_limits(ax):
    ax.set_axis_off()
    ax.set_xlim(-0.9,0.9)
    ax.set_ylim(-0.9,0.9)
    ax.set_zlim(-0.9,0.9)

# === data loading as before ===
vec_dir = "GGenerated_encoded_feature_vectors_CURVS2"
fns = [f for f in os.listdir(vec_dir)
       if f.startswith("encoded_vector_idx") and f.endswith(".npy")]

file_map = {}
pattern = re.compile(r"encoded_vector_idx(\d+)_curvs([0-9.]+)\.npy")
for fn in fns:
    m = pattern.match(fn)
    if not m: continue
    idx  = int(m.group(1))
    curv = float(m.group(2))
    file_map.setdefault(idx, {})[curv] = fn

selected_idxs = sorted(file_map.keys())[:4]
curv_list      = [0.0, 0.50, 0.75, 1.0]

# === 4×5 GRID ===
fig = plt.figure(figsize=(15, 12))
for row_i, idx in enumerate(selected_idxs):
    # get OBB params
    obb = labels[idx, :-2].flatten()

    # 1) First column: OBB only
    ax = fig.add_subplot(4, 5, row_i*5 + 1, projection='3d')
    plot_obb_only(ax, obb)
    ax.set_title(f"idx {idx}", fontsize=8)

    # 2) Next 4 cols: point‐cloud only for each curv
    for col_i, curv in enumerate(curv_list):
        ax = fig.add_subplot(4, 5, row_i*5 + 2 + col_i, projection='3d')
        fn = file_map[idx].get(curv)
        if fn:
            arr = np.load(os.path.join(vec_dir, fn))
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                dec = model.decoder(tensor).view(-1, model.point_size, 6).cpu().numpy()
            pc3 = dec[0,:,:3]
            plot_pc_only(ax, pc3)
            ax.set_title(f"curv {curv:.2f}", fontsize=8)
        else:
            ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[ ]:





# In[13]:


import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
vec_dir = "./DDPM_GEN_nodes_100"
save_dir = "./DECODED_POINTCLOUDS_100"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many plots per page (for visualization safety)
PLOTS_PER_PAGE = 25   # change if desired

# -------------------------------------------------
# LOAD ALL FILES
# -------------------------------------------------
graph_dirs = sorted([
    d for d in os.listdir(vec_dir)
    if os.path.isdir(os.path.join(vec_dir, d))
])

print(f"\nTotal graphs found: {len(graph_dirs)}")
print("Decoding all graphs...\n")

# -------------------------------------------------
# PROCESS ALL FILES
# -------------------------------------------------
for graph_name in graph_dirs:

    graph_path = os.path.join(vec_dir, graph_name)

    # mirror folder structure
    graph_save_path = os.path.join(save_dir, graph_name)
    os.makedirs(graph_save_path, exist_ok=True)

    node_files = sorted([
        f for f in os.listdir(graph_path)
        if f.endswith(".npy")
    ])

    total_files = len(node_files)

    print(f"Graph {graph_name} → {total_files} nodes")

    # optional plotting batching per graph
    for start_idx in range(0, total_files, PLOTS_PER_PAGE):

        batch_files = node_files[start_idx:start_idx + PLOTS_PER_PAGE]
        batch_size = len(batch_files)

        cols = math.ceil(math.sqrt(batch_size))
        rows = math.ceil(batch_size / cols)

        fig = plt.figure(figsize=(4*cols, 4*rows))

        for i, fname in enumerate(batch_files):

            # ---- Load latent ----
            encoded = np.load(os.path.join(graph_path, fname))
            encoded_tensor = torch.tensor(
                encoded,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # ---- Decode ----
            with torch.no_grad():
                decoded = model.decoder(encoded_tensor)
                decoded = decoded.view(-1, model.point_size, 6)

            pc = decoded[0, :, :3].cpu().numpy()

            # ---- Save decoded (same folder structure) ----
            base_name = os.path.splitext(fname)[0]
            np.save(
                os.path.join(graph_save_path, base_name + "_decoded.npy"),
                pc
            )

            # ---- Plot ----
            ax = fig.add_subplot(rows, cols, i+1, projection="3d")
            ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=2, c="darkorange")

            ax.set_title(base_name, fontsize=5)
            ax.set_axis_off()
            ax.set_xlim(-0.9,0.9)
            ax.set_ylim(-0.9,0.9)
            ax.set_zlim(-0.9,0.9)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

print(f"\n✅ Saved ALL decoded point clouds to: {save_dir}")


# In[14]:


pc_array = np.load("data/normalized_rotated_point_clouds6.npy")


# In[15]:


# -------------------------------------------------
# DATASET BOUNDING BOX
# -------------------------------------------------

xyz = pc_array[:, :, :3].reshape(-1, 3)

xmin, ymin, zmin = xyz.min(axis=0)
xmax, ymax, zmax = xyz.max(axis=0)

L = xmax - xmin
W = ymax - ymin
H = zmax - zmin

print("\n=== DATASET BOUNDING BOX ===")
print(f"x: [{xmin:.6f}, {xmax:.6f}]  -> Length  = {L:.6f}")
print(f"y: [{ymin:.6f}, {ymax:.6f}]  -> Width   = {W:.6f}")
print(f"z: [{zmin:.6f}, {zmax:.6f}]  -> Height  = {H:.6f}")


# In[16]:


# -------------------------------------------------
# SINGLE SAMPLE BOUNDING BOXMissing node 11 in graph 448
Missing node 12 in graph 448
Missing node 13 in graph 448
Missing node 14 in graph 448
Missing node 15 in graph 448
✓ Saved graph 448
Missing node 0 in graph 449
Missing node 1 in graph 449
Missing node 2 in graph 449
Missing node 3 in graph 449
Missing node 4 in graph 449
Missing node 5 in graph 449
Missing node 6 in graph 449
✓ Saved graph 449

✅ All graphs processed and saved.


# -------------------------------------------------

sample_xyz = single_sample[0, :, :3].cpu().numpy()

xmin, ymin, zmin = sample_xyz.min(axis=0)
xmax, ymax, zmax = sample_xyz.max(axis=0)

L = xmax - xmin
W = ymax - ymin
H = zmax - zmin

print("\n=== SINGLE SAMPLE BOUNDING BOX ===")
print(f"x: [{xmin:.6f}, {xmax:.6f}]  -> Length  = {L:.6f}")
print(f"y: [{ymin:.6f}, {ymax:.6f}]  -> Width   = {W:.6f}")
print(f"z: [{zmin:.6f}, {zmax:.6f}]  -> Height  = {H:.6f}")


# In[ ]:





# In[17]:


import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------------------------------------
# Load subgraphs
# -------------------------------------------------------

def load_graph_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    return data['subgraphs']

subgraphs = load_graph_from_npz("UNPERTURBED_center_subgraphs450_8.npz")

# -------------------------------------------------------
# Create output directory for saved features
# -------------------------------------------------------

SAVE_DIR = "saved_cluster_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------
# OBB Plotting
# -------------------------------------------------------

def plot_point_cloud_with_obb(obb_vector, scale, centroid, ax, color, alpha=0.25):

    if isinstance(obb_vector, np.ndarray) and obb_vector.size >= 6:
        euler_angles = obb_vector[:3]
        extent = obb_vector[3:6] * scale
    else:
        euler_angles = np.zeros(3)
        extent = np.ones(3) * scale

    r = R.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    half_extent = extent / 2

    local_corners = np.array([
        [-half_extent[0], -half_extent[1], -half_extent[2]],
        [-half_extent[0], -half_extent[1],  half_extent[2]],
        [-half_extent[0],  half_extent[1], -half_extent[2]],
        [-half_extent[0],  half_extent[1],  half_extent[2]],
        [ half_extent[0], -half_extent[1], -half_extent[2]],
        [ half_extent[0], -half_extent[1],  half_extent[2]],
        [ half_extent[0],  half_extent[1], -half_extent[2]],
        [ half_extent[0],  half_extent[1],  half_extent[2]]
    ])

    global_corners = (local_corners @ rotation_matrix.T) + centroid

    edges = [
        [0,1],[0,2],[0,4],[1,3],[1,5],
        [2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]
    ]

    for e in edges:
        ax.plot3D(*zip(global_corners[e[0]], global_corners[e[1]]), color=color, linewidth=1)

    faces = [
        [0,1,3,2],
        [4,5,7,6],
        [0,1,5,4],
        [2,3,7,6],
        [1,3,7,5],
        [0,2,6,4]
    ]

    face_vertices = [[global_corners[i] for i in face] for face in faces]

    poly = Poly3DCollection(
        face_vertices,
        facecolors=color,
        edgecolors=color,
        linewidths=0.5,
        alpha=alpha
    )

    ax.add_collection3d(poly)


# -------------------------------------------------------
# Plot + Save Features
# -------------------------------------------------------

def plot_and_save_subgraph(subgraph_index, ax):

    sg_data = subgraphs[subgraph_index]

    adj_matrix = sg_data['adj_matrix']
    positions = sg_data['positions']
    scales = sg_data['scales']
    obb_euler = sg_data['obb_euler']
    curvatures = sg_data.get('curvatures', None)

    # ---- Save node feature set ----
    save_path = os.path.join(SAVE_DIR, f"cluster_{subgraph_index:03d}.npz")

    np.savez(
        save_path,
        adj_matrix=adj_matrix,
        positions=positions,
        scales=scales,
        obb_euler=obb_euler,
        curvatures=curvatures
    )

    # ---- Build graph ----
    G = nx.from_numpy_array(adj_matrix)

    for i, pos in enumerate(positions):
        G.nodes[i]['centroid'] = pos
        G.nodes[i]['scale'] = scales[i]
        G.nodes[i]['obb_euler'] = obb_euler[i]

    nodes = list(G.nodes())

    # Clean axis
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    colors = cm.tab20(np.linspace(0, 1, len(nodes)))
    node_colors = {node: colors[i] for i, node in enumerate(nodes)}

    # Plot edges
    for edge in G.edges():
        p1 = G.nodes[edge[0]]['centroid']
        p2 = G.nodes[edge[1]]['centroid']
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color='gray',
            linewidth=1.2
        )

    # Plot centroids
    for node in nodes:
        c = G.nodes[node]['centroid']
        ax.scatter(c[0], c[1], c[2], color=node_colors[node], s=30)

    # Plot OBBs
    for node in nodes:
        plot_point_cloud_with_obb(
            G.nodes[node]['obb_euler'],
            G.nodes[node]['scale'],
            G.nodes[node]['centroid'],
            ax,
            node_colors[node],
            alpha=0.25
        )


# -------------------------------------------------------
# 4x8 Grid — FIRST 32 CLUSTERS
# -------------------------------------------------------

def plot_first_32_subgraphs():

    total = len(subgraphs)
    num = min(32, total)

    fig = plt.figure(figsize=(32, 16))

    for i in range(num):
        ax = fig.add_subplot(4, 8, i + 1, projection='3d')
        plot_and_save_subgraph(i, ax)
        ax.view_init(elev=25, azim=35)

    plt.tight_layout()
    plt.savefig("FIRST_32_SUBGRAPH_OVERLAY_GRID100.png", dpi=300)
    plt.show()


# -------------------------------------------------------
# Run
# -------------------------------------------------------


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


import numpy as np
import os
import networkx as nx
from scipy.spatial.transform import Rotation as R

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUBGRAPH_FILE = "UNPERTURBED_center_subgraphs450_8.npz"

# where normalized decoded nodes live
DECODED_DIR = "./DECODED_POINTCLOUDS_100"

# where placed point clouds will be saved
SAVE_DIR = "saved_cluster_features_100"

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------
# Load subgraphs
# -------------------------------------------------------

def load_graph_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    return data['subgraphs']


subgraphs = load_graph_from_npz(SUBGRAPH_FILE)

# -------------------------------------------------------
# Load + transform decoded node
# -------------------------------------------------------

def load_and_place_decoded_node(subgraph_index, node_index, centroid, scale, obb_euler):
    """
    Returns world-space point cloud for one particle node.
    """

    graph_dir = os.path.join(DECODED_DIR, f"graph_{subgraph_index:03d}")

    if not os.path.exists(graph_dir):
        return None

    prefix = f"node_{node_index:03d}"

    matches = [
        f for f in os.listdir(graph_dir)
        if f.startswith(prefix) and f.endswith(".npy")
    ]

    if len(matches) == 0:
        return None

    decoded_pc = np.load(os.path.join(graph_dir, matches[0]))

    # -------------------------------------------------
    # SCALE  (effective radius)
    # -------------------------------------------------
    placed_pc = decoded_pc * scale

    # -------------------------------------------------
    # ROTATE (OBB Euler)
    # -------------------------------------------------
    r = R.from_euler('xyz', obb_euler[:3], degrees=True)
    placed_pc = placed_pc @ r.as_matrix().T

    # -------------------------------------------------
    # TRANSLATE (centroid)
    # -------------------------------------------------
    placed_pc = placed_pc + centroid

    return placed_pc


# -------------------------------------------------------
# Process one subgraph
# -------------------------------------------------------

def process_subgraph(subgraph_index):

    sg_data = subgraphs[subgraph_index]

    adj_matrix = sg_data['adj_matrix']
    positions = sg_data['positions']
    scales = sg_data['scales']
    obb_euler = sg_data['obb_euler']
    curvatures = sg_data.get('curvatures', None)

    # -------------------------------------------------
    # Save metadata
    # -------------------------------------------------

    graph_save_dir = os.path.join(SAVE_DIR, f"graph_{subgraph_index:03d}")
    os.makedirs(graph_save_dir, exist_ok=True)

    meta_path = os.path.join(graph_save_dir, "graph_features.npz")

    np.savez(
        meta_path,
        adj_matrix=adj_matrix,
        positions=positions,
        scales=scales,
        obb_euler=obb_euler,
        curvatures=curvatures
    )

    # -------------------------------------------------
    # Build graph for convenience
    # -------------------------------------------------

    G = nx.from_numpy_array(adj_matrix)

    for i, pos in enumerate(positions):
        G.nodes[i]['centroid'] = pos
        G.nodes[i]['scale'] = scales[i]
        G.nodes[i]['obb_euler'] = obb_euler[i]

    nodes = list(G.nodes())

    # -------------------------------------------------
    # Export each node point cloud
    # -------------------------------------------------

    for node in nodes:

        centroid = G.nodes[node]['centroid']
        scale = G.nodes[node]['scale']
        obb = G.nodes[node]['obb_euler']

        placed_pc = load_and_place_decoded_node(
            subgraph_index,
            node,
            centroid,
            scale,
            obb
        )

        if placed_pc is None:
            print(f"Missing node {node} in graph {subgraph_index}")
            continue

        save_path = os.path.join(
            graph_save_dir,
            f"node_{node:03d}.npy"
        )

        np.save(save_path, placed_pc.astype(np.float32))

    print(f"✓ Saved graph {subgraph_index}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    total = len(subgraphs)

    for i in range(total):
        process_subgraph(i)

    print("\n✅ All graphs processed and saved.")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:


##Resolve Meshes


# In[3]:


import os
import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR  = "saved_cluster_features_100"
OUTPUT_DIR = "Resolved_Cluster_Mesh_1001"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RELAX_ITERS = 100
RELAX_ALPHA = 0.01      # movement strength
SAFETY_SCALE = 1.01    # enlarge radii slightly to ensure separation


# ============================================================
# UTILITIES
# ============================================================

def load_meshes(graph_dir):

    meshes = []
    names  = []

    for f in sorted(os.listdir(graph_dir)):
        if f.endswith(".stl"):
            path = os.path.join(graph_dir, f)
            mesh = o3d.io.read_triangle_mesh(path)
            
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                continue
            
            # ensure normals exist
            if not mesh.has_triangle_normals():
                mesh.compute_triangle_normals()
            
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            meshes.append(mesh)
            names.append(f)

    return meshes, names


def compute_centroids_and_radii(meshes):

    centroids = []
    radii     = []

    for mesh in meshes:

        verts = np.asarray(mesh.vertices)

        centroid = verts.mean(axis=0)

        # bounding sphere radius
        r = np.linalg.norm(verts - centroid, axis=1).max()

        centroids.append(centroid)
        radii.append(r * SAFETY_SCALE)

    return np.array(centroids), np.array(radii)


def resolve_overlaps(centroids, radii, iterations=8, alpha=0.5):

    C = centroids.copy()
    N = len(C)

    for _ in range(iterations):

        for i in range(N):
            for j in range(i + 1, N):

                d = C[j] - C[i]
                dist = np.linalg.norm(d)

                min_dist = radii[i] + radii[j]

                if dist < 1e-8:
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    move = min_dist * 0.5
                else:
                    overlap = min_dist - dist
                    if overlap <= 0:
                        continue
                    direction = d / dist
                    move = overlap * 0.5

                correction = direction * move * alpha

                C[i] -= correction
                C[j] += correction

    return C


def translate_mesh(mesh, delta):

    verts = np.asarray(mesh.vertices)
    verts = verts + delta
    mesh.vertices = o3d.utility.Vector3dVector(verts)


# ============================================================
# PROCESS ONE GRAPH
# ============================================================

def process_graph(graph_name):

    in_dir  = os.path.join(INPUT_DIR, graph_name)
    out_dir = os.path.join(OUTPUT_DIR, graph_name)

    os.makedirs(out_dir, exist_ok=True)

    meshes, names = load_meshes(in_dir)

    if len(meshes) == 0:
        print(f"⚠ No meshes in {graph_name}")
        return

    # --------------------------------------------------------
    # Compute centroids + radii
    # --------------------------------------------------------

    centroids, radii = compute_centroids_and_radii(meshes)

    # --------------------------------------------------------
    # Resolve overlaps
    # --------------------------------------------------------

    new_centroids = resolve_overlaps(
        centroids,
        radii,
        iterations=RELAX_ITERS,
        alpha=RELAX_ALPHA
    )

    # --------------------------------------------------------
    # Apply translations
    # --------------------------------------------------------

    for mesh, name, old_c, new_c in zip(meshes, names, centroids, new_centroids):

        delta = new_c - old_c
        translate_mesh(mesh, delta)
        
        # ---- REQUIRED FOR STL ----
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        
        out_path = os.path.join(out_dir, name)
        o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)

    print(f"✓ Resolved {graph_name}")


# ============================================================
# MAIN
# ============================================================

def main():

    graphs = sorted([
        g for g in os.listdir(INPUT_DIR)
        if g.startswith("graph_")
    ])

    print(f"Found {len(graphs)} graphs")

    for g in graphs:
        process_graph(g)

    print("\n✅ All graphs resolved and saved.")


if __name__ == "__main__":
    main()


# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =====================================================
# CONFIG
# =====================================================

SAVE_DIR = "saved_cluster_features_100"


# =====================================================
# LOAD GRAPH
# =====================================================

def load_graph(graph_index):

    graph_dir = os.path.join(SAVE_DIR, f"graph_{graph_index:03d}")

    if not os.path.exists(graph_dir):
        raise RuntimeError(f"Graph folder not found: {graph_dir}")

    meta_path = os.path.join(graph_dir, "graph_features.npz")

    if not os.path.exists(meta_path):
        raise RuntimeError("Missing graph metadata")

    data = np.load(meta_path)

    adj_matrix = data["adj_matrix"]
    positions  = data["positions"]

    node_files = sorted([
        f for f in os.listdir(graph_dir)
        if f.startswith("node_") and f.endswith(".npy")
    ])

    node_pointclouds = []

    for f in node_files:
        pc = np.load(os.path.join(graph_dir, f))
        node_pointclouds.append(pc)

    return adj_matrix, positions, node_pointclouds


# =====================================================
# PLOT
# =====================================================

def plot_graph(graph_index):

    adj_matrix, positions, node_pointclouds = load_graph(graph_index)

    G = nx.from_numpy_array(adj_matrix)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # -----------------------
    # Plot edges
    # -----------------------
    for u, v in G.edges():

        p1 = positions[u]
        p2 = positions[v]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="gray",
            linewidth=1.0,
            alpha=0.6
        )

    # -----------------------
    # Plot particles
    # -----------------------
    all_pts = []

    for pc in node_pointclouds:

        ax.scatter(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            s=2
        )

        all_pts.append(pc)

    # -----------------------
    # Equal bounds
    # -----------------------
    if all_pts:

        pts = np.vstack(all_pts)

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect([1, 1, 1])

    ax.set_title(f"Graph {graph_index}")
    ax.view_init(elev=25, azim=35)

    plt.show()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # ----------------------------
    # SELECT GRAPH HERE
    # ----------------------------
    GRAPH_ID = 1

    plot_graph(GRAPH_ID)


# In[15]:


if __name__ == "__main__":

    for GRAPH_ID in range(100):   # graphs 0 → 99

        print(f"Viewing graph {GRAPH_ID}")
        plot_graph(GRAPH_ID)


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:





# In[ ]:





# In[18]:


import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUBGRAPH_FILE = "UNPERTURBED_center_subgraphs450_8.npz"
DECODED_DIR = "./DECODED_POINTCLOUDS_100"
SAVE_DIR = "saved_cluster_features_100"

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------
# Load subgraphs
# -------------------------------------------------------

def load_graph_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    return data['subgraphs']

subgraphs = load_graph_from_npz(SUBGRAPH_FILE)

# -------------------------------------------------------
# Load and place decoded geometry
# -------------------------------------------------------

def load_and_place_decoded_node(subgraph_index, node_index, centroid, scale, obb_euler):

    prefix = f"g{subgraph_index:03d}_n{node_index:03d}"

    matches = [
        f for f in os.listdir(DECODED_DIR)
        if f.startswith(prefix) and f.endswith("_decoded.npy")
    ]

    if len(matches) == 0:
        return None

    decoded_pc = np.load(os.path.join(DECODED_DIR, matches[0]))

    # ---- scale ----
    decoded_pc = decoded_pc * scale

    # ---- rotate using OBB euler ----
    r = R.from_euler('xyz', obb_euler[:3], degrees=True)
    decoded_pc = decoded_pc @ r.as_matrix().T

    # ---- translate ----
    decoded_pc = decoded_pc + centroid

    return decoded_pc


# -------------------------------------------------------
# Plot + Save Features
# -------------------------------------------------------
import numpy as np
import matplotlib.cm as cm

c1 = cm.tab20(np.linspace(0,1,20))
c2 = cm.tab20b(np.linspace(0,1,20))
c3 = cm.tab20c(np.linspace(0,1,20))

COLOR_POOL = np.vstack([c1, c2, c3])  # up to 60 distinct colors

def plot_and_save_subgraph(subgraph_index, ax):

    sg_data = subgraphs[subgraph_index]

    adj_matrix = sg_data['adj_matrix']
    positions = sg_data['positions']
    scales = sg_data['scales']
    obb_euler = sg_data['obb_euler']
    curvatures = sg_data.get('curvatures', None)

    # ---- Save node feature set ----
    save_path = os.path.join(SAVE_DIR, f"cluster_{subgraph_index:03d}.npz")

    np.savez(
        save_path,
        adj_matrix=adj_matrix,
        positions=positions,
        scales=scales,
        obb_euler=obb_euler,
        curvatures=curvatures
    )

    # ---- Build graph ----
    G = nx.from_numpy_array(adj_matrix)

    for i, pos in enumerate(positions):
        G.nodes[i]['centroid'] = pos
        G.nodes[i]['scale'] = scales[i]
        G.nodes[i]['obb_euler'] = obb_euler[i]

    nodes = list(G.nodes())

    # Clean axis
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    num_nodes = len(nodes)
    
    if num_nodes <= len(COLOR_POOL):
        colors = COLOR_POOL[:num_nodes]
    else:
        repeats = int(np.ceil(num_nodes / len(COLOR_POOL)))
        colors = np.tile(COLOR_POOL, (repeats, 1))[:num_nodes]
    
    node_colors = {node: colors[i] for i, node in enumerate(nodes)}

    # ---- Plot edges ----
    for edge in G.edges():
        p1 = G.nodes[edge[0]]['centroid']
        p2 = G.nodes[edge[1]]['centroid']
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color='gray',
            linewidth=1.0,
            alpha=0.6
        )

    # ---- Insert decoded geometries ----
    for node in nodes:

        centroid = G.nodes[node]['centroid']
        scale = G.nodes[node]['scale']
        obb = G.nodes[node]['obb_euler']

        decoded_pc = load_and_place_decoded_node(
            subgraph_index,
            node,
            centroid,
            scale,
            obb
        )

        if decoded_pc is not None:
        
            # --- Softer directional lighting ---
            light_dir = np.array([0.3, 0.4, 1.0])
            light_dir = light_dir / np.linalg.norm(light_dir)
        
            centered = decoded_pc - decoded_pc.mean(axis=0)
            normals = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)
        
            # Lambertian term
            lambert = normals @ light_dir
            lambert = np.clip(lambert, 0, 1)
        
            # --- soften ---
            ambient = 0.55           # base illumination (0.5–0.7 works well)
            diffuse_strength = 0.6   # reduces contrast
            gamma = 0.7              # smooths transition
        
            intensity = ambient + diffuse_strength * lambert
            intensity = np.clip(intensity, 0, 1)
            intensity = intensity ** gamma
        
            base_color = np.array(node_colors[node][:3])
        
            shaded_colors = intensity[:, None] * base_color
            shaded_colors = np.clip(shaded_colors, 0, 1)
        
            ax.scatter(
                decoded_pc[:, 0],
                decoded_pc[:, 1],
                decoded_pc[:, 2],
                s=8,
                c=shaded_colors,
                depthshade=False
            )
        else:
            # fallback: plot centroid only
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                s=25,
                color=node_colors[node]
            )


from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

def _make_4x4(title, indices, filename):
    fig = plt.figure(figsize=(16, 16))

    axes = []
    for k, sg_idx in enumerate(indices):
        ax = fig.add_subplot(4, 4, k + 1, projection='3d')
        plot_and_save_subgraph(sg_idx, ax)
        ax.view_init(elev=25, azim=35)
        ax.set_proj_type('persp')
        ax.dist = 9
        axes.append(ax)

    # Leave tighter space for bracket + title
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.98])

    fig.canvas.draw()

    # Compute grid bounds
    top = max(ax.get_position().y1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    mid = 0.5 * (top + bottom)

    # Slight inward trim (shorter bracket)
    trim = 0.04
    y0 = bottom + trim
    y1 = top - trim

    # -----------------------------
    # Vertical title
    # -----------------------------
    fig.text(
        0.045,
        mid,
        title,
        fontsize=36,
        fontweight='bold',
        rotation=90,
        va='center',
        ha='center'
    )

    # -----------------------------
    # Right-angled bracket (flipped)
    # -----------------------------
    x_inner = 0.070   # vertical line (closer to plots)
    x_outer = 0.082   # horizontal stubs extend outward (away from plots)

    lw = 3.0
    color = '0.4'

    # Vertical segment
    fig.add_artist(Line2D(
        [x_inner, x_inner],
        [y0, y1],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))

    # Top horizontal segment
    fig.add_artist(Line2D(
        [x_inner, x_outer],
        [y1, y1],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))

    # Bottom horizontal segment
    fig.add_artist(Line2D(
        [x_inner, x_outer],
        [y0, y0],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))

    fig.savefig(filename, dpi=300)
    #fig.savefig(filename.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    return fig



def plot_first_32_subgraphs():

    total = len(subgraphs)
    num = min(32, total)

    generated_indices = []

    for i in range(num):
        col = i % 8
        generated_indices.append(i)

    _make_4x4("Generated", generated_indices, "Bracket_100_GENERATED.png")

    plt.show()


if __name__ == "__main__":
    plot_first_32_subgraphs()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUBGRAPH_FILE = "UNPERTURBED_center_subgraphs100_6.npz"
DECODED_DIR = "./DECODED_POINTCLOUDS_first32_100GPU"
SAVE_DIR = "saved_cluster_features100"

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------
# Load subgraphs
# -------------------------------------------------------

def load_graph_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    return data['subgraphs']

subgraphs = load_graph_from_npz(SUBGRAPH_FILE)

# -------------------------------------------------------
# Load and place decoded geometry
# -------------------------------------------------------

def load_and_place_decoded_node(subgraph_index, node_index, centroid, scale, obb_euler):

    prefix = f"g{subgraph_index:03d}_n{node_index:03d}"

    matches = [
        f for f in os.listdir(DECODED_DIR)
        if f.startswith(prefix) and f.endswith("_decoded.npy")
    ]

    if len(matches) == 0:
        return None

    decoded_pc = np.load(os.path.join(DECODED_DIR, matches[0]))

    # ---- scale ----
    decoded_pc = decoded_pc * scale

    # ---- rotate using OBB euler ----
    r = R.from_euler('xyz', obb_euler[:3], degrees=True)
    decoded_pc = decoded_pc @ r.as_matrix().T

    # ---- translate ----
    decoded_pc = decoded_pc + centroid

    return decoded_pc


# -------------------------------------------------------
# Plot + Save Features
# -------------------------------------------------------

def plot_and_save_subgraph(subgraph_index, ax):

    sg_data = subgraphs[subgraph_index]

    adj_matrix = sg_data['adj_matrix']
    positions = sg_data['positions']
    scales = sg_data['scales']
    obb_euler = sg_data['obb_euler']
    curvatures = sg_data.get('curvatures', None)

    # ---- Save node feature set ----
    save_path = os.path.join(SAVE_DIR, f"cluster_{subgraph_index:03d}.npz")

    np.savez(
        save_path,
        adj_matrix=adj_matrix,
        positions=positions,
        scales=scales,
        obb_euler=obb_euler,
        curvatures=curvatures
    )

    # ---- Build graph ----
    G = nx.from_numpy_array(adj_matrix)

    for i, pos in enumerate(positions):
        G.nodes[i]['centroid'] = pos
        G.nodes[i]['scale'] = scales[i]
        G.nodes[i]['obb_euler'] = obb_euler[i]

    nodes = list(G.nodes())

    # Clean axis
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    colors = cm.tab20(np.linspace(0, 1, len(nodes)))
    node_colors = {node: colors[i] for i, node in enumerate(nodes)}

    # ---- Plot edges ----
    for edge in G.edges():
        p1 = G.nodes[edge[0]]['centroid']
        p2 = G.nodes[edge[1]]['centroid']
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color='gray',
            linewidth=1.0,
            alpha=0.6
        )

    # ---- Insert decoded geometries ----
    for node in nodes:

        centroid = G.nodes[node]['centroid']
        scale = G.nodes[node]['scale']
        obb = G.nodes[node]['obb_euler']

        decoded_pc = load_and_place_decoded_node(
            subgraph_index,
            node,
            centroid,
            scale,
            obb
        )

        if decoded_pc is not None:
            ax.scatter(
                decoded_pc[:,0],
                decoded_pc[:,1],

                decoded_pc[:,2],
                s=2,
                color=node_colors[node],
                alpha=0.9
            )
        else:
            # fallback: plot centroid only
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                s=25,
                color=node_colors[node]
            )


from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

def _make_4x4(title, indices, filename):
    fig = plt.figure(figsize=(16, 16))

    axes = []
    for k, sg_idx in enumerate(indices):
        ax = fig.add_subplot(4, 4, k + 1, projection='3d')
        plot_and_save_subgraph(sg_idx, ax)
        ax.view_init(elev=25, azim=35)
        axes.append(ax)

    # Leave tighter space for bracket + title
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.98])

    fig.canvas.draw()

    # Compute grid bounds
    top = max(ax.get_position().y1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    mid = 0.5 * (top + bottom)

    # Slight inward trim (shorter bracket)
    trim = 0.04
    y0 = bottom + trim
    y1 = top - trim

    # -----------------------------
    # Vertical title
    # -----------------------------
    fig.text(
        0.045,
        mid,
        title,
        fontsize=36,
        fontweight='bold',
        rotation=90,
        va='center',
        ha='center'
    )

    # -----------------------------
    # Right-angled bracket (flipped)
    # -----------------------------
    x_inner = 0.070   # vertical line (closer to plots)
    x_outer = 0.082   # horizontal stubs extend outward (away from plots)

    lw = 3.0
    color = '0.4'

    # Vertical segment
    fig.add_artist(Line2D(
        [x_inner, x_inner],
        [y0, y1],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))

    # Top horizontal segment
    fig.add_artist(Line2D(
        [x_inner, x_outer],
        [y1, y1],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))

    # Bottom horizontal segment
    fig.add_artist(Line2D(
        [x_inner, x_outer],
        [y0, y0],
        transform=fig.transFigure,
        linewidth=lw,
        color=color
    ))


    fig.savefig(filename.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    return fig



def plot_first_32_subgraphs():

    total = len(subgraphs)
    num = min(32, total)

    generated_indices = []

    for i in range(num):
        col = i % 8
        generated_indices.append(i)
    _make_4x4("Generated", generated_indices, "BracketFIRST_16_100GENERATED.png")

    plt.show()


if __name__ == "__main__":
    plot_first_32_subgraphs()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


# -------------------------------------------------------
# PRINT DIMENSIONS — FIRST 32 CLUSTERS
# -------------------------------------------------------

def plot_first_32_subgraphs():

    # Load inside function to avoid NameError
    data = np.load("UNPERTURBED_center_subgraphs100_6.npz", allow_pickle=True)
    subgraphs = data['subgraphs']

    total = len(subgraphs)
    num = min(32, total)

    print("\n========== PARTICLE DIMENSIONS (WORLD COORDINATES) ==========\n")

    for sg_idx in range(num):

        sg_data = subgraphs[sg_idx]

        scales = sg_data['scales']               # (N,)
        obb_euler = sg_data['obb_euler']         # (N,6)

        base_extents = obb_euler[:, 3:6]         # raw box lengths
        dims = base_extents * scales[:, None]    # effective L,W,H

        print(f"\n--- Subgraph {sg_idx} ---")

        for p_idx, (L, W, H) in enumerate(dims):
            print(f"Particle {p_idx:03d}  "
                  f"L = {L:.6f},  "
                  f"W = {W:.6f},  "
                  f"H = {H:.6f}")

    print("\n=============================================================\n")

if __name__ == "__main__":
    plot_first_32_subgraphs()


# In[26]:


from scipy.spatial.transform import Rotation as R
import numpy as np

def compute_global_bounding_box():

    data = np.load("UNPERTURBED_center_subgraphs100_6.npz", allow_pickle=True)
    subgraphs = data['subgraphs']

    all_points = []

    for sg_data in subgraphs:

        positions = sg_data['positions']
        scales = sg_data['scales']
        obb_euler = sg_data['obb_euler']

        for centroid, scale, obb in zip(positions, scales, obb_euler):

            euler = obb[:3]
            extent = obb[3:6] * scale
            half = extent / 2

            local_corners = np.array([
                [-half[0], -half[1], -half[2]],
                [-half[0], -half[1],  half[2]],
                [-half[0],  half[1], -half[2]],
                [-half[0],  half[1],  half[2]],
                [ half[0], -half[1], -half[2]],
                [ half[0], -half[1],  half[2]],
                [ half[0],  half[1], -half[2]],
                [ half[0],  half[1],  half[2]]
            ])

            Rmat = R.from_euler('xyz', euler, degrees=True).as_matrix()
            global_corners = local_corners @ Rmat.T + centroid

            all_points.append(global_corners)

    all_points = np.vstack(all_points)

    xmin, ymin, zmin = all_points.min(axis=0)
    xmax, ymax, zmax = all_points.max(axis=0)

    L = xmax - xmin
    W = ymax - ymin
    H = zmax - zmin

    print("\n========== GLOBAL BOUNDING BOX ==========")
    print(f"x range: {xmin:.6f} to {xmax:.6f}")
    print(f"y range: {ymin:.6f} to {ymax:.6f}")
    print(f"z range: {zmin:.6f} to {zmax:.6f}")
    print("\nBox Dimensions:")
    print(f"Length  = {L:.6f}")
    print(f"Width   = {W:.6f}")
    print(f"Height  = {H:.6f}")
    print("=========================================\n")
compute_global_bounding_box()


# In[ ]:





# In[ ]:





# In[27]:





# In[33]:





# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:




