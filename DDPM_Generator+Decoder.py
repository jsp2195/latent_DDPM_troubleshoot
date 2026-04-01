#!/usr/bin/env python
# coding: utf-8

"""
Clean DDPM script split from the original notebook-style file.
- Keeps the same conditional U-Net architecture and DDPM implementation.
- Exposes two clean entry points:
  1) train_ddpm(...) for training
  2) run_ddpm_generation(...) for inference / generation

The point-cloud autoencoder workflow now lives in:
  pointcloud_autoencoder_pipeline.py
"""

import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ddpm_pipeline import configure_runtime, build_labels, Unet_conditional, DDPMPipeline


def build_datasets(encoded_features, labels, batch_size=128):
    n = min(11892, len(encoded_features), len(labels))
    x = encoded_features[:n]
    y = labels[:n]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    max_val = x.max() / 2.0

    def preprocess(xb, yb):
        xb = tf.cast(xb, tf.float32) / max_val - 1.0
        yb = tf.cast(yb, tf.float32)
        return xb, yb

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .cache()
        .shuffle(buffer_size=x_train.shape[0])
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, test_dataset


def build_unet(feature_size=32, init_dim=64):
    return Unet_conditional(
        dim=64,
        init_dim=init_dim,
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
        in_res=feature_size,
    )


def train_ddpm(epochs=300, patience=30, batch_size=128):
    configure_runtime()

    encoded_features = np.load("encoded_features_CURVATURE.npy")
    obb_vectors = np.load("obb_vectors_open3d_euler.npy")
    scale_coeffs = np.load("normalized_scale_coefficients6.npy")
    curv_coeffs = np.load("average_mean_curvatures.npy")

    labels = build_labels(obb_vectors, curv_coeffs, scale_coeffs)
    train_dataset, test_dataset = build_datasets(encoded_features, labels, batch_size=batch_size)

    feature_size = encoded_features.shape[1]
    init_dim = 64
    multend = 8
    timesteps = 1000
    endbeta = 0.025

    ddpm = DDPMPipeline(feature_size=feature_size, timesteps=timesteps, endbeta=endbeta, batch_size=batch_size)
    unet = build_unet(feature_size=feature_size, init_dim=init_dim)

    checkpoint_dir = f"./checkpoints_AMC1_vector_separate_euler_{endbeta}beta_{batch_size}batch_es_{init_dim}dim_{multend}mult_4"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(unet=unet)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    lr_schedule = ExponentialDecay(1e-4, decay_steps=10000, decay_rate=0.9, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    @tf.function
    def loss_fn(real, generated):
        return tf.reduce_mean(tf.math.squared_difference(real, tf.cast(generated, real.dtype)))

    @tf.function
    def train_step(batch, class_vec):
        bs = tf.shape(batch)[0]
        t = tf.random.uniform([bs], 0, timesteps, dtype=tf.int32)
        noised, noise = ddpm.forward_noise(batch, t)
        with tf.GradientTape() as tape:
            pred = unet(noised, t, class_vec, training=True)
            loss = loss_fn(noise, pred)
        grads = tape.gradient(loss, unet.trainable_variables)
        opt.apply_gradients(zip(grads, unet.trainable_variables))
        return loss

    @tf.function
    def validation_step(images, class_vec):
        bs = tf.shape(images)[0]
        t = tf.random.uniform([bs], 0, timesteps, dtype=tf.int32)
        noised, noise = ddpm.forward_noise(images, t)
        pred = unet(noised, t, class_vec, training=False)
        return loss_fn(noise, pred)

    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_losses = []
        for x_batch, class_batch in train_dataset:
            train_losses.append(float(train_step(x_batch, class_batch)))
        avg_train = float(np.mean(train_losses))

        val_losses = []
        for x_val, class_val in test_dataset:
            val_losses.append(float(validation_step(x_val, class_val)))
        avg_val = float(np.mean(val_losses))

        print(f"Epoch {epoch}/{epochs} | train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improvement = 0
            ckpt_manager.save(checkpoint_number=epoch)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping")
                break

    final_weights_path = os.path.join(checkpoint_dir, "unet_final.weights.h5")
    unet.save_weights(final_weights_path)
    return final_weights_path


def run_ddpm_generation(weights_path, num_samples=4):
    configure_runtime()
    encoded_features = np.load("encoded_features_CURVATURE.npy")
    obb_vectors = np.load("obb_vectors_open3d_euler.npy")
    scale_coeffs = np.load("normalized_scale_coefficients6.npy")
    curv_coeffs = np.load("average_mean_curvatures.npy")
    labels = build_labels(obb_vectors, curv_coeffs, scale_coeffs)

    feature_size = encoded_features.shape[1]
    timesteps = 1000
    endbeta = 0.025

    unet = build_unet(feature_size=feature_size, init_dim=64)
    _ = unet(tf.zeros((1, feature_size, feature_size, 1)), tf.zeros((1,), dtype=tf.int32), tf.zeros((1, 8), dtype=tf.float32))
    unet.load_weights(weights_path)

    beta = np.linspace(0.0001, endbeta, timesteps).astype(np.float32)
    alpha = (1.0 - beta).astype(np.float32)
    alpha_bar = np.cumprod(alpha).astype(np.float32)

    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    alpha_bar_tf = tf.constant(alpha_bar, dtype=tf.float32)
    beta_tf = tf.constant(beta, dtype=tf.float32)

    @tf.function
    def ddpm_step(x_t, pred_noise, t):
        pred_noise = tf.cast(pred_noise, x_t.dtype)
        t = tf.cast(tf.reshape(t, [-1]), tf.int32)
        a_t = tf.cast(tf.gather(alpha_tf, t), x_t.dtype)[:, None, None, None]
        a_bar_t = tf.cast(tf.gather(alpha_bar_tf, t), x_t.dtype)[:, None, None, None]
        b_t = tf.cast(tf.gather(beta_tf, t), x_t.dtype)[:, None, None, None]
        one = tf.constant(1.0, dtype=x_t.dtype)
        eps_coef = (one - a_t) / tf.sqrt(one - a_bar_t)
        mean = (one / tf.sqrt(a_t)) * (x_t - eps_coef * pred_noise)
        noise = tf.random.normal(tf.shape(x_t), dtype=x_t.dtype)
        return mean + tf.sqrt(b_t) * noise

    random_indices = random.sample(range(1, len(labels)), num_samples)
    out_dir = "./GGenerated_encoded_feature_vectors_CURVS2"
    os.makedirs(out_dir, exist_ok=True)

    for index in random_indices:
        class_vec = labels[index].reshape(1, -1).astype(np.float32)
        x = tf.random.normal((1, feature_size, feature_size, 1), dtype=tf.float32)
        for i in range(timesteps - 1):
            t = tf.constant([timesteps - i - 1], dtype=tf.int32)
            pred_noise = unet(x, t, class_vec, training=False)
            x = ddpm_step(x, pred_noise, t)

        np.save(os.path.join(out_dir, f"encoded_vector_idx{index}.npy"), np.squeeze(x.numpy(), axis=(0, -1)))


if __name__ == "__main__":
    # Train DDPM
    final_weights = train_ddpm()
    # Run DDPM generation with trained weights
    run_ddpm_generation(final_weights)
