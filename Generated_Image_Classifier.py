#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:42:53 2025

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

### Assignment 1 Classifier ###
train_images = np.load('train_images.npy') / 255.0
test_images  = np.load('test_images.npy')  / 255.0
test_labels = np.load('test_labels.npy')
train_labels = np.load('train_labels.npy')
generated_imgs = np.load('generated_cvae/images.npy')
generated_labels = np.load('generated_cvae/labels.npy')

# reshape to (N,28,28,1) float32
train_images = np.expand_dims(train_images, -1).astype("float32")
test_images  = np.expand_dims(test_images,  -1).astype("float32")

# Flatten images
generated_imgs_flat = generated_imgs.reshape(generated_imgs.shape[0], -1)
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

encoded_train_labels = to_categorical(train_labels, num_classes=5)
encoded_test_labels = to_categorical(test_labels, num_classes=5)

# Define model
model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation="sigmoid"),
    Dense(5, activation="softmax")
])


# Compile model
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Train model
history = model.fit(
    train_images_flat,
    encoded_train_labels,
    batch_size=100,
    epochs=50,
    shuffle=True,
    verbose=2,
    validation_split=0.1
)


target_classes = [0, 1, 3]

# Predict on generated images
pred_probs_gen = model.predict(generated_imgs_flat)
pred_classes_gen = np.argmax(pred_probs_gen, axis=1)

# Report: predictions on generated images
print("\n--- CLASSIFIER ON GENERATED SAMPLES (5 per class) ---")
for i in range(len(pred_classes_gen)):
    true_label = int(generated_labels[i])
    pred_label = int(pred_classes_gen[i])
    confidence = np.max(pred_probs_gen[i])
    print(f"True: {true_label} | Pred: {pred_label} | Confidence: {confidence:.4f}")

# Sample real test data from the same classes
def sample_test_data_per_class(test_images, test_labels, target_classes, num_per_class=5):
    samples = []
    labels = []

    for class_id in target_classes:
        idxs = np.where(test_labels == class_id)[0]
        chosen = np.random.choice(idxs, size=num_per_class, replace=False)
        samples.append(test_images[chosen])
        labels.append(test_labels[chosen])

    return np.vstack(samples), np.concatenate(labels)

test_samples, test_sample_labels = sample_test_data_per_class(
    test_images, test_labels, target_classes, num_per_class=5
)

# Flatten and predict on sampled real test images
test_samples_flat = test_samples.reshape(test_samples.shape[0], -1)
test_sample_probs = model.predict(test_samples_flat)
test_sample_preds = np.argmax(test_sample_probs, axis=1)

print("\n--- CLASSIFIER ON REAL TEST SAMPLES (5 per class) ---")
for i in range(len(test_sample_preds)):
    true_label = int(test_sample_labels[i])
    pred_label = int(test_sample_preds[i])
    confidence = np.max(test_sample_probs[i])
    print(f"True: {true_label} | Pred: {pred_label} | Confidence: {confidence:.4f}")
