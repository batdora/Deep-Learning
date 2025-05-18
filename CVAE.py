#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:12:17 2025

@author: batdora
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers, metrics
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import os

# https://keras.io/examples/generative/vae/
# 1) Sampling layer using reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE(Model):
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
        # If data is a tuple, unpack it
        if isinstance(data, tuple):
            imgs, labels = data
        else:
            raise ValueError("Expected data to be a (images, labels) tuple")
    
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([imgs, labels], training=True)
            reconstruction = self.decoder([z, labels], training=True)
    
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.binary_crossentropy(imgs, reconstruction),
                    axis=(1, 2)
                )
            )
    
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
    
            total_loss = recon_loss + kl_loss
    
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



    def test_step(self, data):
        # Handle (images, labels) format
        if isinstance(data, tuple):
            imgs, labels = data
        elif isinstance(data, dict):
            imgs = data.get("encoder_img")
            labels = data.get("encoder_label")
        else:
            raise ValueError("Expected data to be (images, labels) tuple or dict.")
    
        # Forward pass without training
        z_mean, z_log_var, z = self.encoder([imgs, labels], training=False)
        reconstruction = self.decoder([z, labels], training=False)
    
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                losses.binary_crossentropy(imgs, reconstruction),
                axis=(1, 2)
            )
        )
    
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
    
        total_loss = recon_loss + kl_loss
    
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

train_images = np.load('train_images.npy') / 255.0
test_images  = np.load('test_images.npy')  / 255.0

# reshape to (N,28,28,1) float32
train_images = np.expand_dims(train_images, -1).astype("float32")
test_images  = np.expand_dims(test_images,  -1).astype("float32")
all_images   = np.concatenate([train_images, test_images], axis=0)

test_labels = np.load('test_labels.npy')
train_labels = np.load('train_labels.npy')

all_labels   = np.concatenate([train_labels, test_labels])
num_classes = 5

all_labels_cat = tf.keras.utils.to_categorical(all_labels, num_classes)  # shape: (N,5)

latent_dims = [ 64]
kernel_sizes = [ 5]
activations = ['gelu']
for act in activations:
    for latent_dim in latent_dims:
        for kernel_size in kernel_sizes:
            
            # Encoder
            image_input = layers.Input(shape=(28, 28, 1), name="encoder_img")
            label_input = layers.Input(shape=(num_classes,), name="encoder_label")
            
            # Broadcast labels and concatenate
            label_broadcast = layers.Dense(28 * 28, activation='relu')(label_input)
            label_reshaped  = layers.Reshape((28, 28, 1))(label_broadcast)
            x = layers.Concatenate()([image_input, label_reshaped])  # shape (28,28,2)
            
            # Encoder layers as before
            x = layers.Conv2D(32, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2D(64, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2D(128, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(latent_dim, activation=act)(x)
            
            z_mean = layers.Dense(latent_dim)(x)
            z_log_var = layers.Dense(latent_dim)(x)
            z = Sampling()([z_mean, z_log_var])
            
            encoder = Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

            # Decoder
            latent_input = layers.Input(shape=(latent_dim,), name="decoder_z")
            label_input  = layers.Input(shape=(num_classes,), name="decoder_label")
            
            # Concatenate z + label
            x = layers.Concatenate()([latent_input, label_input])
            x = layers.Dense(4 * 4 * 128, activation=act)(x)
            x = layers.Reshape((4, 4, 128))(x)
            x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Cropping2D(((2,2),(2,2)))(x)
            output = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
            
            decoder = Model([latent_input, label_input], output, name="decoder")

            
            # 6) Instantiate, compile, and train VAE
            start_time = time.time()
            cvae = CVAE(encoder, decoder, name="vae")
            cvae.compile(optimizer=optimizers.Adam())
            
            history = cvae.fit(x=all_images,
                               y=all_labels_cat,
                            epochs=20,
                            batch_size=100,
                            validation_split=0.1)
            end_time = time.time()
            training_time = end_time - start_time
            log_string = (
                f"Activation func =  {act}"
                f"Kernel Size = {kernel_size}, Latent Dim = {latent_dim}\n"
                f"loss: {history.history['loss'][-1]:.4f} - "
                f"reconstruction_loss: {history.history['reconstruction_loss'][-1]:.4f} - "
                f"kl_loss: {history.history['kl_loss'][-1]:.4f} - "
                f"val_loss: {history.history['val_loss'][-1]:.4f} - "
                f"val_reconstruction_loss: {history.history['val_reconstruction_loss'][-1]:.4f} - "
                f"val_kl_loss: {history.history['val_kl_loss'][-1]:.4f}\n\n"
            )
           
### Show all losses in one plot ###                
   
plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
plt.plot(history.history['kl_loss'],             label='KL Loss')
plt.plot(history.history['loss'],                label='Total Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.title('VAE Training Losses'); plt.show()

### Generate Images ###
# Assume class indices: 0=rabbit, 1=yoga, 3=snowman
target_classes = [0, 1, 3]
samples_per_class = 5

def generate_specific_classes(decoder, latent_dim, target_classes, samples_per_class):
    images = []
    labels = []

    for class_id in target_classes:
        one_hot = np.eye(5)[[class_id] * samples_per_class]
        z = np.random.normal(size=(samples_per_class, latent_dim)).astype("float32")
        imgs = decoder.predict([z, one_hot], batch_size=8, verbose=0)
        images.append(imgs)
        labels.append(np.full((samples_per_class,), class_id))

    return np.vstack(images), np.concatenate(labels)

generated_imgs, true_labels = generate_specific_classes(decoder, latent_dim, target_classes, 5)

# Plot all generated samples: 3 classes × 5 images = 15
fig, axes = plt.subplots(len(target_classes), samples_per_class, figsize=(8, 5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i, class_id in enumerate(target_classes):
    for j in range(samples_per_class):
        idx = i * samples_per_class + j
        ax = axes[i, j]
        ax.imshow(generated_imgs[idx].squeeze(), cmap="gray")
        ax.set_title(f"Class {class_id}", fontsize=8)
        ax.axis("off")

plt.tight_layout()
plt.show()

# Create output directory
os.makedirs("generated_cvae", exist_ok=True)

# Save .npy files
np.save("generated_cvae/images.npy", generated_imgs.astype(np.float32))       
np.save("generated_cvae/labels.npy", true_labels.astype(np.int32))            
