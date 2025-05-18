#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 18:40:15 2025

@author: batdora
"""
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# Normalize images to the [0,1] range
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Binarize the normalized images
train_images_binary = (train_images_normalized > 0.3).astype(np.int32)
test_images_binary = (test_images_normalized > 0.3).astype(np.int32)

layer_encoder = [128,256]
add_layer_encoder = 256
add_layer_decoder = 256
layer_decoder = [128,256]
layer_bottleneck = [128,256]

total_parameters = len(layer_encoder)*len(layer_bottleneck)*len(layer_decoder)
counter = 0

log = []

for i in layer_decoder:
    for j in layer_encoder:
        for k in layer_bottleneck:
            counter +=1
            
            inputs = layers.Input(shape=(28, 28))
            x = layers.LSTM(i, return_sequences=True)(inputs)
            
            if add_layer_encoder:
                x = layers.LSTM(i, return_sequences=True)(x)
                
            encoded = layers.LSTM(k, return_sequences=False)(inputs) 
            
            # Decoder
            x = layers.RepeatVector(28)(encoded) # (batch, 28, k)                  
            x = layers.LSTM(j, return_sequences=True)(x)
            
            if add_layer_decoder:
                x = layers.LSTM(add_layer_decoder, return_sequences=True)(x)
                
            decoded = layers.TimeDistributed(layers.Dense(28))(x) # Reconstruct each row
            
            # Autoencoder model
            autoencoder = models.Model(inputs, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Fit the model
            history = autoencoder.fit(
                x=train_images_binary,
                y=train_images_binary,
                epochs=40,
                batch_size=100,
                validation_split=0.1
            )
            
            reconstructed = autoencoder.predict(test_images_binary)
            
            # Flatten both to shape (num_samples, 784)
            test_flat = test_images_binary.reshape(len(test_images_binary), -1)
            recon_flat = reconstructed.reshape(len(reconstructed), -1)
            
            # Compute MSE per image
            mse_per_image = np.mean((test_flat - recon_flat) ** 2, axis=1)
            
            # Average MSE over all images
            mean_mse = np.mean(mse_per_image)
            print(f"Test MSE: {mean_mse:.5f}")
            log.append([[j, add_layer_encoder, k, i, add_layer_decoder], mean_mse])
    
            
            
            ### PLOTTING ###
            
            # Plot Loss
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Training vs. Validation Loss')
            plt.legend()
            plt.grid(True)
            shape_str = f"Encoder: {j}, Add_Enc: {add_layer_encoder}, Bottleneck: {k}, Decoder: {i}, Add_Dec: {add_layer_decoder}"
            plt.figtext(0.5, -0.05, shape_str, wrap=True, horizontalalignment='center', fontsize=10)

            plt.show()
            
            print(f"{counter} of total runs {total_parameters}")


### tSNE ###
        
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
cbar.ax.set_yticklabels(['rabbit', 'yoga', 'snowman', 'hand', 'motorbike'])

plt.title('t-SNE of Latent Space (Test Set)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True)
plt.show()

