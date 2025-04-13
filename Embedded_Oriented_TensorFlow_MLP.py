#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 03:02:12 2025

@author: batdora
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

train_images = np.load('train_images.npy') / 255.0
train_labels = np.load('train_labels.npy')
test_images  = np.load('test_images.npy') / 255.0
test_labels  = np.load('test_labels.npy')


num_classes = 5
train_labels_enc = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels_enc  = tf.keras.utils.to_categorical(test_labels, num_classes)


X_train, X_val, y_train_enc, y_val_enc = train_test_split(
    train_images, train_labels_enc, test_size=0.15, random_state=42
)

# Flatten the images
X_train = X_train.reshape(-1, 784)
X_val   = X_val.reshape(-1, 784)
test_images = test_images.reshape(-1, 784)

hand_glove = np.array([
    0.088068, -0.42703, 0.21275, -0.46137, 0.88653, 0.31964, -0.0094923, 0.12259,
    -0.011234, -0.2113, -0.11769, 0.085932, -0.54004, 0.27666, -0.074244, 0.11298,
    -0.31362, -0.30666, 0.13833, -0.99789, -0.10509, 0.56499, 0.30105, -0.60911,
    0.21528, -1.9955, -0.23075, 0.36169, 0.36569, -0.83593, 3.1593, 0.38484,
    -0.58786, 0.30266, -0.080106, 0.7723, 0.14527, 0.54844, 0.13905, -0.15815,
    0.37559, 0.64325, -0.35815, 0.2687, 0.37035, -0.12839, 0.14046, -0.37389,
    -0.24085, -0.80756
])
snowman_glove = np.array([
    0.8478, 0.72781, -0.67417, -0.41539, -0.011032, 0.64983, -0.074828, -0.97899,
    -0.53372, -0.069041, 0.031371, 0.61443, 0.011017, 0.98334, 0.54671, 0.032902,
    -0.13441, 0.81386, -0.0011416, 0.27799, -0.40528, -0.23932, -0.26089, -0.54246,
    0.45309, 0.25588, -0.25099, 1.3033, 0.91843, -0.30506, -0.71782, 0.51889,
    -0.80543, 0.54186, -1.2563, 0.41131, 0.11828, -0.10867, -0.069171, 0.18632,
    -0.63774, -0.23173, -0.4134, -0.58999, 0.43614, -0.32327, 0.16359, -0.5004,
    0.20189, 0.050679
])
yoga_glove = np.array([
    -0.23044, -0.0051984, -1.4649, -0.022837, -0.39316, -1.1695, 0.95209, -1.0679,
    1.0116, 1.1636, 0.42308, 0.14469, -0.46647, -0.25521, -0.28783, 0.3255,
    -0.52067, 1.1222, 0.48441, 0.24881, -0.078846, 1.7609, 0.24501, 0.78899,
    0.79384, 0.059622, -0.59283, -1.0519, -0.72533, 0.054916, 1.9151, -0.017871,
    -0.42783, 0.7601, 0.062296, 0.11187, -0.025858, 0.22016, 0.45561, 1.3235,
    -0.18446, -0.37049, -1.0038, 1.9645, 0.89571, -0.57719, 1.3445, -1.1053,
    -0.74618, 0.28625
])
rabbit_glove = np.array([
    0.53049, -0.63657, -0.53314, -0.37542, 0.28821, 1.2374, -0.47467, -1.2037,
    0.58209, -0.55149, -0.2719, 0.70193, 0.74694, 0.34327, 0.65301, 0.54077,
    0.66454, 0.47677, -1.0837, 0.12478, -0.15093, -0.66961, 0.55866, 0.60741,
    0.70239, -0.91675, -0.92081, 0.59262, 0.0070694, -0.95443, 0.69853, -0.13292,
    -0.061585, 1.206, -0.58842, 0.43482, -0.19392, -0.19351, -0.07301, -0.85527,
    0.32885, 0.57285, -0.57111, 0.10893, 1.0902, -0.028394, 0.78458, -0.97332,
    0.36124, -0.056677
])
motorbike_glove = np.array([
    0.14362, -1.1402, 0.39368, 0.18135, -0.094088, 0.67473, -0.52618, 0.21466,
    0.62416, -0.17217, 0.67109, -1.1389, -0.84819, 0.085305, 0.20975, -0.59836,
    -0.78554, 1.21, -0.90412, -1.009, 0.42731, 0.39614, -1.0663, 0.66758,
    0.54771, -0.93963, -0.31805, 0.14893, 0.4489, -0.1986, 0.20147, 0.47226,
    -0.31627, 0.83248, 0.84036, 0.40339, 0.24902, -0.034884, -0.11794, 0.89527,
    -0.33927, 0.13761, -0.037933, -0.26963, 0.85965, -1.174, 0.31216, -0.62433,
    1.4447, -1.0968
])

labels_glove = np.array([rabbit_glove, yoga_glove, hand_glove, snowman_glove, motorbike_glove])


image_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32)  # Output: 32-dim embedding (unbounded, linear)
])

glove_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50,)),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(32)  # Output: 32-dim embedding
])


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


def predict_accuracy(images, labels_onehot):
    """
    Compute classification accuracy: for each image, compute its embedding using image_model,
    then compute glove embeddings for each class using glove_model. Use Euclidean distance to assign the class.
    """
    # Get image embeddings from image_model
    img_embeds = image_model(images, training=False)  # shape: (N, 32)
    # Compute glove embeddings for all classes (fixed glove vectors)
    glove_embeds = glove_model(labels_glove)  # shape: (5, 32)
    
    # Convert to numpy arrays
    img_embeds_np = img_embeds.numpy()
    glove_embeds_np = glove_embeds.numpy()
    
    # Compute pairwise Euclidean distances between image embeds and glove embeds
    dists = np.sqrt(np.sum((img_embeds_np[:, np.newaxis, :] - glove_embeds_np[np.newaxis, :, :]) ** 2, axis=2))
    pred_classes = np.argmin(dists, axis=1)
    true_classes = np.argmax(labels_onehot, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    return accuracy


epochs = 40
margin = 0.01

train_loss_log = []
val_loss_log = []
train_acc_log  = []
val_acc_log    = []

# Training loop
for epoch in range(epochs):

    # Use GradientTape to compute gradients and update weights
    with tf.GradientTape() as tape:
        # Forward pass on training images
        anchor_output = image_model(X_train, training=True)  # shape: (N_train, 32)
        # For each training sample, find the positive glove embedding (using true labels)
        pos_indices = np.argmax(y_train_enc, axis=1)
        pos_embeddings = glove_model(labels_glove[pos_indices])  # shape: (N_train, 32)
        # For each training sample, sample a negative glove (different from true label)
        neg_indices = np.array([
            np.random.choice([j for j in range(num_classes) if j != i])
            for i in pos_indices
        ])
        neg_embeddings = glove_model(labels_glove[neg_indices])  # shape: (N_train, 32)
        
        # Compute squared Euclidean distances
        pos_dist = tf.reduce_sum(tf.square(anchor_output - pos_embeddings), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor_output - neg_embeddings), axis=1)
        # Triplet loss: max( pos_dist - neg_dist + margin, 0 )
        loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    # Apply gradients
    grads = tape.gradient(loss, image_model.trainable_variables + glove_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, image_model.trainable_variables + glove_model.trainable_variables))
    train_loss_log.append(loss.numpy())
    
    # Compute training classification accuracy
    train_acc = predict_accuracy(X_train, y_train_enc)
    train_acc_log.append(train_acc)
    

    val_anchor = image_model(X_val, training=False)
    val_pos_idx = np.argmax(y_val_enc, axis=1)
    val_pos_embed = glove_model(labels_glove[val_pos_idx])
    val_neg_idx = np.array([
        np.random.choice([j for j in range(num_classes) if j != i])
        for i in val_pos_idx
    ])
    val_neg_embed = glove_model(labels_glove[val_neg_idx])
    
    val_pos_dist = tf.reduce_sum(tf.square(val_anchor - val_pos_embed), axis=1)
    val_neg_dist = tf.reduce_sum(tf.square(val_anchor - val_neg_embed), axis=1)
    val_loss = tf.reduce_mean(tf.maximum(val_pos_dist - val_neg_dist + margin, 0.0))
    val_loss_log.append(val_loss.numpy())
    
    val_acc = predict_accuracy(X_val, y_val_enc)
    val_acc_log.append(val_acc)
    
    print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {loss.numpy():.4f}, Train Acc: {train_acc:.4f}, " +
          f"Val Loss: {val_loss.numpy():.4f}, Val Acc: {val_acc:.4f}")


test_acc = predict_accuracy(test_images, test_labels_enc)
print(f"\nTest Set Accuracy: {test_acc:.4f}")


epochs_range = np.arange(1, epochs + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_log, label='Train Loss', marker='o')
plt.plot(epochs_range, val_loss_log, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc_log, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_acc_log, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

