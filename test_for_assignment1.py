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
  
def backpropagation(batch, caches, output, labels, weights, biases, v_weights, v_biases, learning_rate = 0.001, momentum= 0.9, activation = ReLu, actvation_last = Softmax):
    
    derivatives = []
 
    
    #For Softmax
    cross = derCrossEntropy(labels,output[-1])
    dW4 = (caches[2][0].T) @ cross
    db4 = np.sum(cross, axis=0, keepdims=True)
    derivatives.append([dW4,db4])
    
 
    # Can be for looped
    dLayer3 = cross @  weights[3].T
    dReLu3 =  dLayer3 * derReLu(caches[2][1])
    dW3 = caches[1][0].T @ dReLu3
    db3 = np.sum(dReLu3, axis=0, keepdims=True)
    derivatives.append([dW3,db3])
    
    dLayer2 = dReLu3 @ weights[2].T
    dReLu2 = dLayer2 * derReLu(caches[1][1])
    dW2 = caches[0][0].T @ dReLu2
    db2 = np.sum(dReLu2, axis=0, keepdims=True)
    derivatives.append([dW2,db2])
    
    dLayer1 = dReLu2 @ weights[1].T
    dReLu1 = dLayer1 * derReLu(caches[0][1])
    dW1 = batch.T @ dReLu1
    db1 = np.sum(dReLu1, axis=0, keepdims=True)
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
    # Create batches without labels (we don't need to shuffle here if order doesn't matter)
    # We'll reuse create_batches but ignore labels.
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    batches = []
    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        batches.append(batch)
    
    predictions = []
    for batch in batches:
        outputs, _, _, _ = forward_pass_network(batch, weights, biases, batch_size)
        # The final output is in outputs[-1]; take the argmax along the class axis.
        preds = np.argmax(outputs[-1], axis=1)
        predictions.extend(preds)
    return np.array(predictions)


        

encoded_train_labels = encoder(train_labels)
encoded_test_labels = encoder(test_labels)


# Initialize weights and biases for a network with architecture: [784,256,64,16,5]

# Create randomized batches (this also flattens the data to (num_samples, 784))
batches, batches_labels = create_batches(train_images, encoded_train_labels, 20)
  
weights, biases, v_weights, v_biases = initialize([784, 256, 128, 64, 5])

epochs = 1

for j in range(epochs):
    for i in range(len(batches)):
        
        output, weights, biases, caches = forward_pass_network(batches[i], weights, biases, batch_size= 20, activation=ReLu)
          
        loss = categorical_crossentropy_loss(batches_labels[i], output[-1])
        
        print(loss)
        
        backpropagation(batches[i], caches, output[-1], batches_labels[i], weights, biases, v_weights, v_biases)
    

    

 

# After training, for example:
test_preds = predict(test_images, weights, biases, batch_size=20)
accuracy = np.mean(test_preds == test_labels)
print("Test Accuracy:", accuracy)
  

