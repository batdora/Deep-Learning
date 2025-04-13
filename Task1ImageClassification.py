#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:39:55 2025

@author: batdora
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
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
    
    # Subtract largest element for stability in exponentiation
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
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
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

def xavier_initialize(layers, seed=42):
    
    weights = []
    biases = []
    np.random.seed(seed)
    for i in range(len(layers) - 1):
        fan_in = layers[i]
        fan_out = layers[i+1]

        stddev = np.sqrt(2 / (fan_in + fan_out))
        weight = np.random.randn(fan_in, fan_out) * stddev

        weights.append(weight)
        biases.append(np.zeros(fan_out))
    
    v_weights = [np.zeros_like(w) for w in weights]
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
    
    v_weights = [np.zeros_like(W) for W in weights]
    v_biases = [np.zeros_like(b) for b in biases]

    return weights, biases, v_weights, v_biases


def forward_pass_network(batch, weights, biases, batch_size, activation=ReLu, seed=42):

    outputs = []  
    caches = []  # before and after activation.
    
    
    a = batch  # 'a' is the input to the network; shape: (batch_size, 784)
    
    # Loop over layers. There are len(weights) layers.
    for i in range(len(weights)):
        # Compute the linear combination: z = a*W + b
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
    #One hot encoder for labels. for example 2 -> 0 0 1 0 0 or 3 -> 0 0 0 1 0
    vector = np.zeros([len(labels),5])
    for i, k in enumerate(labels):
        vector[i][k]= 1
    return np.array(vector)

def backpropagation_auto(batch, caches, output, labels, weights, biases, v_weights, v_biases,
                    learning_rate=0.01, momentum=0.9, activation_deriv=derReLu):

    derivatives = []
    batch_size = batch.shape[0]

    # Output layer 
    dZ = derCrossEntropy(labels, output)  # shape (batch, num_classes)
    A_prev = caches[-2][0] if len(caches) > 1 else batch

    dW = (1 / batch_size) * A_prev.T @ dZ
    db = (1 / batch_size) * np.sum(dZ, axis=0, keepdims=True)
    derivatives.append([dW, db])
    dA = dZ @ weights[-1].T  # propagate to previous layer

    # --- Hidden layers ---
    for i in reversed(range(len(weights) - 1)):
        A_prev = batch if i == 0 else caches[i - 1][0]
        Z = caches[i][1]

        dZ = dA * activation_deriv(Z)  # element-wise product
        dW = (1 / batch_size) * A_prev.T @ dZ
        db = (1 / batch_size) * np.sum(dZ, axis=0, keepdims=True)
        derivatives.append([dW, db])
        dA = dZ @ weights[i].T

    # --- Apply momentum updates in correct order ---
    for i in range(len(weights)):
        idx = -(i + 1)  # reverse order (last appended = output layer)

        v_weights[idx] = momentum * v_weights[idx] + learning_rate * derivatives[i][0]
        weights[idx] -= v_weights[idx]

        v_biases[idx] = momentum * v_biases[idx] + learning_rate * derivatives[i][1].squeeze()
        biases[idx] -= v_biases[idx]


def backpropagation(batch, caches, output, labels,
                    weights, biases, v_weights, 
                    v_biases, learning_rate = 0.01, 
                    momentum= 0.9, activation = ReLu, actvation_last = Softmax):

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
    
    for i in range(len(derivatives)):
        # Update velocity for weights: v = momentum * v + learning_rate * gradient
        v_weights[-(i+1)] = momentum * v_weights[-(i+1)] + learning_rate * derivatives[i][0]
        # Update weight: W = W - v
        weights[-(i+1)] -= v_weights[-(i+1)]
        
        # Update velocity for biases similarly:
        bias_grad = (derivatives[i][1].T).squeeze()
        v_biases[-(i+1)] = momentum * v_biases[-(i+1)] + learning_rate * bias_grad
        biases[-(i+1)] -= v_biases[-(i+1)]
    
    
    
def predict(X, weights, biases, batch_size=20,activation=ReLu):
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    predictions = []
    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        outputs, _, _, _ = forward_pass_network(batch, weights, biases, batch_size,activation=activation)
        preds = np.argmax(outputs[-1], axis=1)
        predictions.extend(preds)
    return np.array(predictions)


# This code here used for training the dataset between train_images and test_images.
# Create randomized batches (this also flattens the data to (num_samples, 784))
encoded_train_labels = encoder(train_labels)
encoded_test_labels = encoder(test_labels)
batches, batches_labels = create_batches(train_images, encoded_train_labels, 20)
momentums = [0.5, 0.7, 0.9, 0.95]
activations = [Tanh, Sigmoid, ReLu]
derAct = [derReLu, derTanh, derSigmoid]
architectures = [[784,512,128,5]]
epochs = 40
weights, biases, v_weights, v_biases = xavier_initialize([784,128,5])

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(batches)):
        output, weights, biases, caches = forward_pass_network(batches[i], weights, biases, batch_size= 20, activation=Sigmoid)
        loss = categorical_crossentropy_loss(batches_labels[i], output[-1])
        total_loss += loss
        backpropagation_auto(batches[i], caches, output[-1], batches_labels[i], weights, biases, v_weights, v_biases,momentum=0.9,activation_deriv=derSigmoid)



    # Print epoch statistics
    avg_loss = total_loss / len(batches)
    if epoch % 1 == 0:  # Print every epoch
        test_preds = predict(test_images, weights, biases, batch_size=20, activation=Sigmoid)
        accuracy = np.mean(test_preds == test_labels)
        # Open file in append mode
        with open("resultsArc.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Momentum: 0.9, Activation: Sigmoid, Model Architecture: {modelarc}\n")



# Validation Set and Train Test Split with 0.15 split.
num_total = train_images.shape[0]
val_size = int(0.15 * num_total)
indices = np.random.permutation(num_total)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

X_train = train_images[train_indices]
y_train = encoded_train_labels[train_indices]
X_val = train_images[val_indices]
y_val = encoded_train_labels[val_indices]


batches, batches_labels = create_batches(X_train, y_train, 20)
train_losses = []
train_accuracies = []
val_accuracies = []


# This code sample here re-initalizes the weights and momentum due to the fact that this is a separate training based on validation and training data. 
weights, biases, v_weights, v_biases = xavier_initialize([784,128,5])
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(batches)):
        output, _, _, caches = forward_pass_network(batches[i], weights, biases, batch_size=20, activation=Sigmoid)
        loss = categorical_crossentropy_loss(batches_labels[i], output[-1])
        total_loss += loss
        backpropagation_auto(batches[i], caches, output[-1], batches_labels[i],
                             weights, biases, v_weights, v_biases,
                             momentum=0.9, activation_deriv=derSigmoid)

    avg_loss = total_loss / len(batches)
    train_losses.append(avg_loss)

    # Training accuracy
    train_preds = predict(X_train, weights, biases, batch_size=20, activation=Sigmoid)
    train_true = np.argmax(y_train, axis=1)
    train_acc = np.mean(train_preds == train_true)
    train_accuracies.append(train_acc)

    # Validation accuracy
    val_preds = predict(X_val, weights, biases, batch_size=20, activation=Sigmoid)
    val_true = np.argmax(y_val, axis=1)
    val_acc = np.mean(val_preds == val_true)
    val_accuracies.append(val_acc)

    # Log to file
    with open("resultsArc2.txt", "a") as f:
        f.write(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, "
                f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}, "
                f"Momentum: 0.9, Activation: Sigmoid, Model Architecture: [784,128,5]\n")

# Evaluation on test set
test_preds = predict(test_images, weights, biases, batch_size=20, activation=Sigmoid)
test_true = test_labels
test_onehot = encoder(test_labels)

def predict_proba(X, weights, biases):
    X_flat = X.reshape(X.shape[0], -1)
    outputs, _, _, _ = forward_pass_network(X_flat, weights, biases, batch_size=20, activation=Sigmoid)
    return outputs[-1]

test_probs = predict_proba(test_images, weights, biases)

# Metrics
report = classification_report(test_true, test_preds, output_dict=False)
roc_auc = roc_auc_score(test_onehot, test_probs, multi_class='ovr')
pr_auc_scores = []

for i in range(5):
    precision, recall, _ = precision_recall_curve(test_onehot[:, i], test_probs[:, i])
    pr_auc_scores.append(auc(recall, precision))

# Save metrics
with open("final_test_metrics.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nMulticlass ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Average Precision-Recall AUC: {np.mean(pr_auc_scores):.4f}\n")

# Plotting
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("loss_accuracy_plot.png")
plt.show()











