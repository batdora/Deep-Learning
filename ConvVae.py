import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers, metrics
import matplotlib.pyplot as plt
import os
import time

# https://keras.io/examples/generative/vae/
# 1) Sampling layer using reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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
train_images = np.load('train_images.npy') / 255.0
test_images  = np.load('test_images.npy')  / 255.0

# reshape to (N,28,28,1) float32
train_images = np.expand_dims(train_images, -1).astype("float32")
test_images  = np.expand_dims(test_images,  -1).astype("float32")
all_images   = np.concatenate([train_images, test_images], axis=0)

latent_dims = [ 64]
kernel_sizes = [ 5]
# activations = [                       
#     layers.LeakyReLU(alpha=0.2),   
#     'selu',                        
#     # 'gelu'                         
# ]
activations = ['gelu']
os.makedirs("imagesVAE", exist_ok=True)
log_file = "outputVAE2.txt"
open(log_file, "a").close()  # clear log file
for act in activations:
    for latent_dim in latent_dims:
        for kernel_size in kernel_sizes:
            encoder_inputs = layers.Input(shape=(28, 28, 1), name="encoder_input")
            x = layers.Conv2D(32, 3, strides=2, padding="same", activation=act)(encoder_inputs)
            x = layers.Conv2D(64, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2D(128, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(latent_dim, activation=act)(x)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z = Sampling(name="z")([z_mean, z_log_var])
            encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

            # 3) Decoder
            latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
            x = layers.Dense(4 * 4 * 128, activation=act)(latent_inputs)
            x = layers.Reshape((4, 4, 128))(x)
            x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation=act)(x)
            x = layers.Cropping2D(cropping=((2,2),(2,2)), name='crop_to_28')(x)                                     # 32→28
            decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
            decoder = Model(latent_inputs, decoder_outputs, name="decoder")
            
            # 6) Instantiate, compile, and train VAE
            start_time = time.time()
            vae = VAE(encoder, decoder, name="vae")
            vae.compile(optimizer=optimizers.Adam())
            
            history = vae.fit(all_images,
                            epochs=40,
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
            with open(log_file, "a") as f:
                f.write(log_string)
                
                
### Show all losses in one plot ###
plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
plt.plot(history.history['kl_loss'],             label='KL Loss')
plt.plot(history.history['loss'],                label='Total Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.title('VAE Training Losses'); plt.show()

### Show all losses independently ###
"""
plt.figure()
plt.plot(history.history['loss'], label='Train Total Loss')
plt.plot(history.history['val_loss'], label='Val Total Loss')
plt.title(f'VAE Losses (kernel={kernel_size}, latent={latent_dim})')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig(f"imagesVAE/vae_kernel_{act}_{kernel_size}_latent{latent_dim}_LOSS.png")
plt.close()

plt.figure()
plt.plot(history.history['kl_loss'], label='Train KL Loss')
plt.plot(history.history['val_kl_loss'], label='Val KL Loss')
plt.title(f'KL Losses (kernel={kernel_size}, latent={latent_dim})')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig(f"imagesVAE/vae_kernel_{act}_{kernel_size}_latent{latent_dim}_KL.png")
plt.close()

plt.figure()
plt.plot(history.history['reconstruction_loss'], label='Train Reconstruction Loss')
plt.plot(history.history['val_reconstruction_loss'], label='Val Reconstruction Loss')
plt.title(f'Reconstruction Losses (kernel={kernel_size}, latent={latent_dim})')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig(f"imagesVAE/vae_kernel_{act}_{kernel_size}_latent{latent_dim}_Re.png")
plt.close()

latent_dim = decoder.input_shape[1]
"""

### Generate Data and Show them ###

"""
# sample 8 random points in latent space
z = np.random.normal(size=(8, latent_dim)).astype("float32")

# generate 8 images
samples = decoder.predict(z, batch_size=8)  # shape (8,28,28,1), values in [0,1]

# plot 2×4 grid
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

latent_dim = decoder.input_shape[1]

# sample 8 random points in latent space
z = np.random.normal(size=(8, latent_dim)).astype("float32")

# generate 8 images
samples = decoder.predict(z, batch_size=8)  # shape (8,28,28,1), values in [0,1]

# plot 2×4 grid
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

"""