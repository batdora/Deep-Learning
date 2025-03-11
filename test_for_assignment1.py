#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:39:55 2025

@author: batdora
"""

import numpy as np

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
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


def derSoftmax(vector):
    s = Softmax(vector)
    jacobian = np.diag(s) - np.outer(s, s)
    return jacobian

def derReLu (vector):
    return np.where(np.array(vector)>0, 1 ,0)

def derSigmoid (vector):
    return Sigmoid(vector)*(1-Sigmoid(vector))

def derTanh (vector):
    return (1-(Tanh(vector)**2))


##Batch Creator

def create_batches(X, batch_size, seed=42):
    
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
    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        batches.append(batch)
    
    return batches


def initialize(vector, seed=42):
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

    return weights, biases


def forward_pass_network(X, batch_size, activation=ReLu, seed=42):
  
    # Create randomized batches (this also flattens the data to (num_samples, 784))
    batches = create_batches(X, batch_size, seed)
    
    outputs = []  # Final output for each batch
    caches = []   # Cache for each batch (used later for backprop, if needed)
    
    # Initialize weights and biases for a network with architecture: [784,256,64,16,5]
    weights, biases = initialize([784, 256, 64, 16, 5])
    
    for batch in batches:
        a = batch  # 'a' is the input to the network; shape: (batch_size, 784)
        cache_layers = []  # Cache for the current batch
        
        # Loop over layers. There are len(weights) layers.
        for i in range(len(weights)):
            # Compute the linear combination: z = a*W + b
            # biases[i] (a 1D array) broadcasts along the batch dimension.
            z = np.dot(a, weights[i]) + biases[i]
            
            # For the output layer (last layer), use Softmax; otherwise use the given activation.
            if i == len(weights) - 1:
                print(a)
                a = Softmax(z)
                print(a)
            else:
                a = activation(z)
                
            
            # Save the tuple (a, z) for potential use in backpropagation.
            cache_layers.append((a, z))
        
        outputs.append(a)
        caches.append(cache_layers)
    
    return outputs, caches

