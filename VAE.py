#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:37:15 2025

@author: batdora
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers, metrics
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# https://keras.io/examples/generative/vae/

# Sampling layer using reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# VAE model with custom train_step for loss functions
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker          = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker             = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction       = self.decoder(z,    training=True)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.binary_crossentropy(data, reconstruction),
                    axis=(1,2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var
                             - tf.square(z_mean)
                             - tf.exp(z_log_var),
                             axis=1)
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # exactly the same computation, but no grads or optimizer
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction       = self.decoder(z,    training=False)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                losses.binary_crossentropy(data, reconstruction),
                axis=(1,2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var
                         - tf.square(z_mean)
                         - tf.exp(z_log_var),
                         axis=1)
        )
        total_loss = recon_loss + kl_loss

        # update the same trackers so val_loss etc appear in history
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
        }

# Prepare data
train_images = np.load('train_images.npy') / 255.0
test_images  = np.load('test_images.npy')  / 255.0
# reshape to (N,28,28,1) float32
train_images = np.expand_dims(train_images, -1).astype("float32")
test_images  = np.expand_dims(test_images,  -1).astype("float32")
all_images   = np.concatenate([train_images, test_images], axis=0)

latent_dims = [32]
kernel_sizes = [5]
activation_funcs = ["relu"]
results = []


for activation_func in activation_funcs:
    for kernel_size in kernel_sizes:
        for latent_dim in latent_dims:
            # Encoder
            inputs = layers.Input(shape=(28, 28))
            x = layers.LSTM(256, return_sequences=True)(inputs)
            x = layers.LSTM(256, return_sequences=True)(inputs)
            encoded = layers.LSTM(128, return_sequences=False)(x) 
            
            z_mean = layers.Dense(latent_dim, name="z_mean")(encoded)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoded)
            z = Sampling(name="z")([z_mean, z_log_var])
            encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
            encoder.summary()
            
            # Decoder
            latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
            x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
            x = layers.Reshape((7, 7, 64))(x)
            x = layers.Conv2DTranspose(64, kernel_size, strides=2, padding="same", activation=activation_func)(x)
            x = layers.Conv2DTranspose(32, kernel_size, strides=2, padding="same", activation=activation_func)(x)
            decoder_outputs = layers.Conv2DTranspose(1, kernel_size, padding="same", activation="sigmoid")(x)
            decoder = Model(latent_inputs, decoder_outputs, name="decoder")
            decoder.summary()
            
            # Instantiate, compile, and train VAE
            vae = VAE(encoder, decoder, name="vae")
            vae.compile(optimizer=optimizers.Adam())
            history = vae.fit(all_images,
                              epochs=10,
                              batch_size=100,
                              validation_split=0.1)
            
            results.append({
                'latent_dim': latent_dim,
                'kernel_size': kernel_size,
                'activation': activation_func,
                'reconstruction_loss': history.history['reconstruction_loss'][-1],
                'kl_loss': history.history['kl_loss'][-1],
                'total_loss': history.history['loss'][-1],
            })
       
            # Plot losses
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
            plt.plot(history.history['kl_loss'],             label='KL Loss')
            plt.plot(history.history['loss'],                label='Total Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.legend(); plt.title('VAE Training Losses')
            plt.figtext(0.5, -0.1, f'Latent Dim: {latent_dim}, Kernel Size: {kernel_size}, Activation Func: {activation_func}',
                        wrap=True, horizontalalignment='center', fontsize=10)
            plt.tight_layout()
            plt.show()
       
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_loss').reset_index(drop=True)

def get_feats_batched(imgs_uint8, model, batch_size=64):
    """
    imgs_uint8: numpy array (N,28,28,1), dtype uint8
    model: pretrained InceptionV3(include_top=False,pooling='avg')
    returns: numpy array (N,2048)
    """
    feats = []
    n = imgs_uint8.shape[0]
    for i in range(0, n, batch_size):
        batch = imgs_uint8[i : i+batch_size]
        batch_rgb = np.repeat(batch, 3, axis=-1)  # → (B,28,28,3)
        batch_rs  = tf.image.resize(batch_rgb, (299, 299))
        batch_inp = tf.keras.applications.inception_v3.preprocess_input(batch_rs)
        f = model.predict(batch_inp, verbose=0)
        feats.append(f)
    return np.vstack(feats)

inc_feats = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", pooling="avg"
)

real_uint8 = (all_images * 255).astype(np.uint8)
real_feats = get_feats_batched(real_uint8, inc_feats, batch_size=64)

N = 1000
z = np.random.normal(size=(N, latent_dim)).astype("float32")
gen = decoder.predict(z, batch_size=100)       # (N,28,28,1), [0,1]
gen_uint8 = (gen * 255).astype(np.uint8)
gen_feats  = get_feats_batched(gen_uint8, inc_feats, batch_size=64)

mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
mu_g, sigma_g = gen_feats.mean(axis=0),  np.cov(gen_feats,  rowvar=False)
covmean = sqrtm(sigma_r.dot(sigma_g))

if np.iscomplexobj(covmean):
    covmean = covmean.real
fid_value = np.sum((mu_r - mu_g)**2) + np.trace(sigma_r + sigma_g - 2 * covmean)
print(f"FID = {fid_value:.2f}")

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax in axes.flatten():
    idx = np.random.randint(0, N)
    ax.imshow(gen[idx].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

"""
# Extract Encoder for tSNE
encoder = models.Model(inputs=autoencoder.input, outputs=encoded)

# Get latent vector
latent_vectors = encoder.predict(test_images_binary)

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(latent_vectors)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    latent_2d[:, 0],
    latent_2d[:, 1],
    c=test_labels,
    cmap='nipy_spectral',
    s=5
)

# Discrete colorbar fix
cbar = plt.colorbar(scatter, boundaries=np.arange(6)-0.5, ticks=np.arange(5))
cbar.ax.set_yticklabels(['rabbit', 'yoga', 'snowman', 'hand', 'motorbike'])  # or whatever your labels are

plt.title('t-SNE of Latent Space (Test Set)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True)
plt.show()
"""

