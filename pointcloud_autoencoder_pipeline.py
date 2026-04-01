import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class PointCloudAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 3)),
            tf.keras.layers.Conv1D(64, 1, activation='relu'),
            tf.keras.layers.Conv1D(128, 1, activation='relu'),
            tf.keras.layers.Conv1D(256, 1, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(4096 * 3),
            tf.keras.layers.Reshape((4096, 3)),
        ])

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)


def chamfer_like_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def load_point_clouds(path="data/normalized_rotated_point_clouds6.npy"):
    pcs = np.load(path).astype(np.float32)
    if pcs.ndim != 3 or pcs.shape[-1] != 3:
        raise ValueError(f"Expected (N, P, 3), got {pcs.shape}")
    return pcs


def train_autoencoder(point_clouds_path="data/normalized_rotated_point_clouds6.npy", epochs=100, batch_size=32,
                      save_dir="./checkpoints_autoencoder"):
    pcs = load_point_clouds(point_clouds_path)
    x_train, x_val = train_test_split(pcs, test_size=0.2, random_state=42)

    model = PointCloudAutoencoder()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=chamfer_like_mse)
    history = model.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=epochs, batch_size=batch_size)

    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(os.path.join(save_dir, "pointcloud_autoencoder.weights.h5"))
    return model, history


def run_autoencoder(point_clouds_path="data/normalized_rotated_point_clouds6.npy", weights_path="./checkpoints_autoencoder/pointcloud_autoencoder.weights.h5",
                    out_encoded="encoded_features_CURVATURE.npy"):
    pcs = load_point_clouds(point_clouds_path)
    model = PointCloudAutoencoder()
    _ = model(tf.zeros((1, pcs.shape[1], 3), dtype=tf.float32))
    model.load_weights(weights_path)
    encoded = model.encoder.predict(pcs, verbose=1)
    np.save(out_encoded, encoded)
    return encoded


if __name__ == "__main__":
    model, _ = train_autoencoder()
    run_autoencoder()
