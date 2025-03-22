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

#print(train_images.shape) # (20000, 28, 28)
#print(test_images.shape) # (5000, 28, 28)


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



def initialize(vector, seed=42):
    
    """ 
    
    Vector input is the number of nuerons per layer structured as a vector
    So for a network with 784x64x16x4, input vector is [784,64,16,4]
    
    """
    weights = []
    biases = []
    
    np.random.seed(seed)
    
    for i in range(len(vector) - 1):
        # Weight shape: (current layer size, next layer size)
        weight = np.random.randn(vector[i], vector[i+1])*0.01
        weights.append(weight)
        # Bias shape: (next layer size,)
        bias = np.random.randn(vector[i+1])*0.01
        biases.append(bias)


    # After initializing weights and biases:
    v_weights = [np.zeros_like(W) for W in weights]
    v_biases = [np.zeros_like(b) for b in biases]

    return weights, biases, v_weights, v_biases



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
  
def backpropagation_corrected(batch, caches, output, labels, weights, biases, v_weights, v_biases, learning_rate=0.01, momentum=0.9, activation=ReLu):
    batch_size = batch.shape[0]
    num_layers = len(weights)
    
    # Initialize gradients
    dW = [None] * num_layers
    db = [None] * num_layers
    
    # Compute gradient for output layer (using cross-entropy derivative with softmax)
    dZ = output - labels  # This is the combined derivative of softmax + cross-entropy
    dW[num_layers-1] = (1/batch_size) * np.dot(caches[num_layers-2][0].T, dZ)
    db[num_layers-1] = (1/batch_size) * np.sum(dZ, axis=0)
    
    # Backpropagate through hidden layers
    dA_prev = dZ
    
    for l in reversed(range(num_layers-1)):
        dA = np.dot(dA_prev, weights[l+1].T)
        
        # Select the appropriate activation derivative
        if activation == ReLu:
            der_activation = derReLu
        elif activation == Sigmoid:
            der_activation = derSigmoid
        elif activation == Tanh:
            der_activation = derTanh
        
        # Get Z from the cache for the current layer
        Z = caches[l][1]
        
        # Compute dZ using the derivative of the activation function
        dZ = dA * der_activation(Z)
        
        # Compute input for this layer (either from previous layer cache or the original input)
        if l > 0:
            A_prev = caches[l-1][0]
        else:
            A_prev = batch
            
        # Compute gradients
        dW[l] = (1/batch_size) * np.dot(A_prev.T, dZ)
        db[l] = (1/batch_size) * np.sum(dZ, axis=0)
        
        # Set dA_prev for next iteration
        dA_prev = dZ
    
    # Update weights and biases using momentum
    for l in range(num_layers):
        v_weights[l] = momentum * v_weights[l] + learning_rate * dW[l]
        weights[l] -= v_weights[l]
        
        v_biases[l] = momentum * v_biases[l] + learning_rate * db[l]
        biases[l] -= v_biases[l]
    
    return weights, biases, v_weights, v_biases

    
    
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

# Updated training loop
def train_network(train_images, train_labels, test_images, test_labels, 
                  architecture=[784, 256, 64, 16, 5], 
                  batch_size=20, 
                  epochs=10, 
                  learning_rate=0.01, 
                  momentum=0.9,
                  activation=ReLu):
    
    # Encode labels
    encoded_train_labels = encoder(train_labels)
    encoded_test_labels = encoder(test_labels)
    
    # Create batches
    batches, batches_labels = create_batches(train_images, encoded_train_labels, batch_size)
    
    # Initialize weights and biases
    weights, biases, v_weights, v_biases = he_initialize(architecture)  # Using He initialization
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(batches)):
            # Forward pass
            outputs, weights, biases, caches = forward_pass_network(
                batches[i], weights, biases, batch_size, activation)
            
            # Compute loss
            loss = categorical_crossentropy_loss(batches_labels[i], outputs[-1])
            total_loss += loss
            
            # Backpropagation
            weights, biases, v_weights, v_biases = backpropagation_corrected(
                batches[i], caches, outputs[-1], batches_labels[i], 
                weights, biases, v_weights, v_biases,
                learning_rate, momentum, activation)
        
        # Print epoch statistics
        avg_loss = total_loss / len(batches)
        if epoch % 1 == 0:  # Print every epoch
            # Evaluate on test set
            test_preds = predict(test_images, weights, biases, batch_size)
            accuracy = np.mean(test_preds == test_labels)
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return weights, biases

# To use this function
weights, biases = train_network(train_images, train_labels, test_images, test_labels, epochs=10)
    




  

