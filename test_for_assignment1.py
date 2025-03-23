#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:39:55 2025

@author: batdora
"""

import numpy as np

train_images = np.load('train_images.npy')/255
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')/255
test_labels = np.load('test_labels.npy')


## Code for Activation Functions and Derivatives

def ReLu (vector):
    return np.maximum(0,np.array(vector))

def Sigmoid (vector):
    return 1/(1+ np.exp(-1*np.array(vector)))

def Tanh (vector):
    return np.tanh(np.array(vector))

def Softmax(vector):
    
    x_shifted = vector - np.max(vector, axis=-1, keepdims=True)
    
    exps = np.exp(x_shifted)

    softmax_values = exps / np.sum(exps, axis=-1, keepdims=True)
    
    return softmax_values

def derReLu (vector):
    return np.where(np.array(vector)>0, 1 ,0)

def derSigmoid (vector):
    return Sigmoid(vector)*(1-Sigmoid(vector))

def derTanh (vector):
    return (1-(Tanh(vector)**2))


#This is combined derivative for using with SoftMax
def derCrossEntropy(y_true, y_pred):
    return y_pred-y_true


##Batch Creator

def create_batches(X, X_labels, batch_size, seed=42):
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Take the number of total
    num_samples = X.shape[0]
    
    # Flatten the images: from (5000, 28, 28) to (5000, 784)
    X_flat = X.reshape(num_samples, -1)
    
    # Create a random permutation of indices and shuffle X_flat accordingly
    #indices = np.random.permutation(num_samples)
    #X_shuffled = X_flat[indices]
    
    batches = []
    batches_labels = []
    
    for i in range(0, num_samples, batch_size):
        #For data
        batch = X_flat[i:i+batch_size]
        batches.append(batch)
        
        #For label
        label = X_labels[i:i+batch_size]
        batches_labels.append(label)
    
    return batches, batches_labels


def he_initialize(vector, seed=42):

    weights = []
    biases = []
    np.random.seed(seed)
    for i in range(len(vector) - 1):
        fan_in = vector[i]
        scale = np.sqrt(2.0 / fan_in)
        weight = np.random.randn(vector[i], vector[i+1]) * scale
        weights.append(weight)
        bias = np.zeros(vector[i+1])
        biases.append(bias)
    
    # After initializing weights and biases:
    v_weights = [np.zeros_like(W) for W in weights]
    v_biases = [np.zeros_like(b) for b in biases]

    return weights, biases, v_weights, v_biases


def forward_pass_network(batch, weights, biases, batch_size, activation=ReLu, seed=42):

    outputs = []  # Final output for each batch
    caches = []   # Cache for each batch (used later for backprop, if needed)
    
    
    a = batch  # 'a' is the input to the network; shape: (batch_size, 784)
    
    # Loop over layers. There are len(weights) layers.
    for i in range(len(weights)):
        # Compute the linear combination: z = a*W + b
        # biases[i] (a 1D array) broadcasts along the batch dimension.
        z = np.dot(a, weights[i]) + biases[i]
        
        # For the output layer (last layer), use Softmax; otherwise use the given activation.
        if i == len(weights) - 1:
            a = Softmax(z)  
        else:
            a = activation(z)
            
        
        # Save the tuple (a, z) for potential use in backpropagation.
        caches.append((a, z))
        outputs.append(a)
      
    
    return outputs, weights, biases, caches


def categorical_crossentropy_loss(y_true, y_pred):
   
    # Small value to avoid log(0)
    epsilon = 1e-9
    # Compute the loss for each sample and then average
    sample_losses = -np.sum(y_true * np.log(y_pred + epsilon), axis=1)
    return np.mean(sample_losses)


def encoder(labels):
    
    vector = np.zeros([len(labels),5])
    
    for i, k in enumerate(labels):
        vector[i][k]= 1
        
    return np.array(vector)
  
def backpropagation(batch, caches, output, labels, weights, biases, v_weights, v_biases, learning_rate = 0.01, momentum= 0.9, activation = ReLu, actvation_last = Softmax):

    derivatives = []
    batch_size = batch.shape[0]
    
    #For Softmax
    cross = derCrossEntropy(labels,output)
    dW4 = (1/batch_size)*((caches[2][0].T) @ cross)
    db4 = (1/batch_size)*(np.sum(cross, axis=0, keepdims=True))
    derivatives.append([dW4,db4])
    
 
    # Can be for looped
    dLayer3 = cross @  weights[3].T
    dReLu3 =  dLayer3 * derReLu(caches[2][1])
    dW3 = (1/batch_size)*(caches[1][0].T @ dReLu3)
    db3 = (1/batch_size)*(np.sum(dReLu3, axis=0, keepdims=True))
    derivatives.append([dW3,db3])
    
    dLayer2 = dReLu3 @ weights[2].T
    dReLu2 = dLayer2 * derReLu(caches[1][1])
    dW2 = (1/batch_size)*(caches[0][0].T @ dReLu2)
    db2 = (1/batch_size)*(np.sum(dReLu2, axis=0, keepdims=True))
    derivatives.append([dW2,db2])
    
    dLayer1 = dReLu2 @ weights[1].T
    dReLu1 = dLayer1 * derReLu(caches[0][1])
    dW1 = (1/batch_size)*(batch.T @ dReLu1)
    db1 = (1/batch_size)*(np.sum(dReLu1, axis=0, keepdims=True))
    derivatives.append([dW1,db1])

    
    """
    # Can be for looped
    dLayer3 = cross @  weights[3].T
    dTanh3 =  dLayer3 * derTanh(caches[2][1])
    dW3 = caches[1][0].T @ dTanh3
    db3 = np.sum(dTanh3, axis=0, keepdims=True)
    derivatives.append([dW3,db3])
    
    dLayer2 = dTanh3 @ weights[2].T
    dTanh2 = dLayer2 * derTanh(caches[1][1])
    dW2 = caches[0][0].T @ dTanh2
    db2 = np.sum(dTanh2, axis=0, keepdims=True)
    derivatives.append([dW2,db2])
    
    dLayer1 = dTanh2 @ weights[1].T
    dTanh1 = dLayer1 * derTanh(caches[0][1])
    dW1 = batch.T @ dTanh1
    db1 = np.sum(dTanh1, axis=0, keepdims=True)
    derivatives.append([dW1,db1])
    """
    
    
    for i in range(len(derivatives)):
        # Update velocity for weights: v = momentum * v + learning_rate * gradient
        v_weights[-(i+1)] = momentum * v_weights[-(i+1)] + learning_rate * derivatives[i][0]
        # Update weight: W = W - v
        weights[-(i+1)] -= v_weights[-(i+1)]
        
        # Update velocity for biases similarly:
        # Convert derivative for bias to proper shape if needed.
        bias_grad = (derivatives[i][1].T).squeeze()
        v_biases[-(i+1)] = momentum * v_biases[-(i+1)] + learning_rate * bias_grad
        biases[-(i+1)] -= v_biases[-(i+1)]
    
    
    
def predict(X, weights, biases, batch_size=20):
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    predictions = []
    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        outputs, _, _, _ = forward_pass_network(batch, weights, biases, batch_size)
        # The final output is in outputs[-1]; take argmax along the class axis.
        preds = np.argmax(outputs[-1], axis=1)
        predictions.extend(preds)
    return np.array(predictions)



encoded_train_labels = encoder(train_labels)
encoded_test_labels = encoder(test_labels)


# Initialize weights and biases for a network with architecture: [784,256,64,16,5]

# Create randomized batches (this also flattens the data to (num_samples, 784))
batches, batches_labels = create_batches(train_images, encoded_train_labels, 20)

weights, biases, v_weights, v_biases = he_initialize([784, 256, 64, 16, 5])


epochs = 40

for epoch in range(epochs):
    
    total_loss = 0
    
    
    for i in range(len(batches)):
        
        output, weights, biases, caches = forward_pass_network(batches[i], weights, biases, batch_size= 20, activation=ReLu)
          
        loss = categorical_crossentropy_loss(batches_labels[i], output[-1])
        
        total_loss += loss
        
        backpropagation(batches[i], caches, output[-1], batches_labels[i], weights, biases, v_weights, v_biases)



    # Print epoch statistics
    avg_loss = total_loss / len(batches)
    if epoch % 1 == 0:  # Print every epoch
        # Evaluate on test set
        test_preds = predict(test_images, weights, biases, batch_size=20)
        accuracy = np.mean(test_preds == test_labels)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")



import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation  
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

# Define the model with the architecture [784, 256, 64, 16, 5]
model = Sequential([
    Dense(256, input_shape=(784,), activation="relu"),  # First hidden layer
    Dense(64, activation="relu"),                       # Second hidden layer
    Dense(16, activation="relu"),                       # Third hidden layer
    Dense(5, activation="softmax")                      # Output layer with 5 classes
])

# Display model summary
model.summary()

# Compile with SGD optimizer using momentum
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    optimizer=sgd_optimizer, 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Make sure labels are properly one-hot encoded
# train_labels_onehot = to_categorical(train_labels, num_classes=5)

# Train the model
# Assuming train_images is already flattened and normalized
train_images_flat = train_images.reshape(train_images.shape[0], -1)  # Flatten 28x28 to 784
history = model.fit(
    train_images_flat,
    encoded_train_labels,  # Using your encoder function output
    batch_size=20,
    epochs=10,
    shuffle=True,
    verbose=2,
    validation_split=0.1  # Optional: use 10% of training data for validation
)

# Evaluate on test set
test_images_flat = test_images.reshape(test_images.shape[0], -1)  # Flatten 28x28 to 784
test_loss, test_acc = model.evaluate(test_images_flat, encoded_test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")





