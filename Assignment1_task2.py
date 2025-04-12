#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:49:38 2025

@author: batdora
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:39:55 2025

@author: batdora
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


train_images = np.load('train_images.npy')/255
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')/255
test_labels = np.load('test_labels.npy')

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


## Code for Activation Functions and Derivatives

def ReLu (vector):
    return np.maximum(0,np.array(vector))

def Sigmoid (vector):
    return 1/(1+ np.exp(-1*np.array(vector)))

def Tanh (vector):
    return np.tanh(np.array(vector))


def triplet_loss(a, p, n, margin=1.0):
    # Compute L2 distances per sample (shape: (batch,))
    positive = np.sum((a - p) ** 2, axis=1)
    negative = np.sum((a - n) ** 2, axis=1)

    # Triplet loss for each sample
    losses = positive - negative + margin
    mask = (losses > 0).astype(np.float32)

    # Final scalar loss to minimize
    loss = np.mean(np.maximum(losses, 0))

    # Compute gradients only for triplets violating the margin
    # shape: (batch_size, embed_dim)
    
    grad_a = 2 * ((n - p)) * mask[:, np.newaxis]  
    grad_p = -2 * (a - p) * mask[:, np.newaxis]
    grad_n = 2 * (a - n) * mask[:, np.newaxis]

    return loss, grad_a, grad_p, grad_n

    
def get_triplet_batch(image_batch, labels, glove_vectors):
    label_indices = np.argmax(labels, axis=1)  # format back from one hot
    positive_gloves = glove_vectors[label_indices]

    # Generate negative indices (mismatched labels)
    negative_indices = []
    for label in label_indices:
        choices = [i for i in range(glove_vectors.shape[0]) if i != label]
        negative_indices.append(np.random.choice(choices))
    negative_gloves = glove_vectors[negative_indices]

    return positive_gloves, negative_gloves

    

def derReLu (vector):
    return np.where(np.array(vector)>0, 1 ,0)

def derSigmoid (vector):
    return Sigmoid(vector)*(1-Sigmoid(vector))

def derTanh (vector):
    return (1-(Tanh(vector)**2))


##Batch Creator

def create_batches(X, X_labels, batch_size, seed=42):
    np.random.seed(seed)
    
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    
    # Create a random permutation of indices and shuffle both X and labels
    indices = np.random.permutation(num_samples)
    X_shuffled = X_flat[indices]
    labels_shuffled = X_labels[indices]
    
    batches = []
    batches_labels = []
    
    for i in range(0, num_samples, batch_size):
        batches.append(X_shuffled[i:i+batch_size])
        batches_labels.append(labels_shuffled[i:i+batch_size])
    
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

# Gaussian Xavier for Sigmoid and Tanh
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



def forward_pass(x, weights, biases, activation=ReLu, output_activation=None):
    a = x
    caches = []

    for i in range(len(weights)):
        z = np.dot(a, weights[i]) + biases[i]
        
        # If the final layer do this
        if i == len(weights) - 1:
            if output_activation:
                a = output_activation(z)
            else:
                a = z  # for the two MLP model
        
        # Else do activation function
        else:
            a = activation(z)

        caches.append((a, z))

    return a, caches




def encoder(labels):
    
    vector = np.zeros([len(labels),5])
    
    for i, k in enumerate(labels):
        vector[i][k]= 1
        
    return np.array(vector)
  
def backprop_generic(x, caches, dA, weights, biases, v_weights, v_biases,
                     activation_fn=ReLu, activation_deriv=derReLu,
                     lr=0.01, momentum=0.9):

    grads_w = []
    grads_b = []

    for i in reversed(range(len(weights))):
        A_prev = x if i == 0 else caches[i-1][0]
        Z = caches[i][1]

        dW = (1 / x.shape[0]) * A_prev.T.dot(dA)
        db = (1 / x.shape[0]) * np.sum(dA, axis=0, keepdims=True)

        grads_w.insert(0, dW)
        grads_b.insert(0, db)

        if i != 0:
            dZ = dA.dot(weights[i].T)
            dA = dZ * activation_deriv(caches[i-1][1])

    for i in range(len(weights)):
        v_weights[i] = momentum * v_weights[i] + lr * grads_w[i]
        weights[i] -= v_weights[i]

        v_biases[i] = momentum * v_biases[i] + lr * grads_b[i].squeeze()
        biases[i] -= v_biases[i]
 
    
def predict(X, img_weights, img_biases, glove_weights, glove_biases, labels_glove, activation, batch_size=20):
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    predictions = []

    # Vectorized glove embedding encoding
    glove_embeds, _ = forward_pass(labels_glove, glove_weights, glove_biases, activation)  # shape: (C, D)

    for i in range(0, num_samples, batch_size):
        batch = X_flat[i:i+batch_size]
        img_embed, _ = forward_pass(batch, img_weights, img_biases, activation)  # shape: (B, D)

        # Compute pairwise Euclidean distances: (B, C)
        dists = np.sqrt(
            np.sum((img_embed[:, np.newaxis, :] - glove_embeds[np.newaxis, :, :]) ** 2, axis=2)
        )  # shape: (batch_size, num_classes)

        preds = np.argmin(dists, axis=1)  # smallest distance = most similar
        predictions.extend(preds)

    return np.array(predictions)

#Plotting Functions

def get_glove_embeddings(glove_vectors, glove_weights, glove_biases):
    embeddings, _ = forward_pass(glove_vectors, glove_weights, glove_biases, activation=ReLu)
    return embeddings  # shape: (num_classes, d)

def plot_embeddings(embeddings, labels, method='tsne', title="Glove Embeddings in Shared Space"):
    if embeddings.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(3, len(embeddings) - 1), random_state=42)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        reduced = reducer.fit_transform(embeddings)
    else:
        reduced = embeddings  # already 2D

    plt.figure(figsize=(6, 6))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y, label=label)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()




# Using the MODELS

encoded_train_labels = encoder(train_labels)
encoded_test_labels = encoder(test_labels)


# Initialize weights and biases 

# Common embedding space dimensionality
embedding_dims = [3,5,16,32,64]


epochs = 40
batch_size = 20
activationfuncs = [ReLu, Sigmoid, Tanh]
derfuncs = [derReLu, derSigmoid, derTanh]
margin = 0.01
category_names = ["rabbit", "yoga", "hand", "snowman", "motorbike"]


batches, batches_labels = create_batches(train_images, encoded_train_labels, batch_size)
for embedding_dim in embedding_dims:
    
    # Model Architecture
    image_model_arch = [784, 256, 128, embedding_dim]
    glove_model_arch = [50, 128, embedding_dim]
    
    for r in range(3):
        
        # He initialize for ReLu and Xavier for Sigmoid and Tanh
        if activationfuncs[r] == ReLu:
            # Image encoder
            img_weights, img_biases, img_vw, img_vb = he_initialize(image_model_arch)
    
            # Glove encoder
            glove_weights, glove_biases, glove_vw, glove_vb = he_initialize(glove_model_arch)
        else:
            # Image encoder
            img_weights, img_biases, img_vw, img_vb = xavier_initialize(image_model_arch)
    
            # Glove encoder
            glove_weights, glove_biases, glove_vw, glove_vb = xavier_initialize(glove_model_arch)
        
   
        print(f"The working activation function is {activationfuncs[r].__name__} and embedding dim is {embedding_dim}")
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(batches)):
                
                # Image forward
                img_output, img_caches = forward_pass(batches[i], img_weights, img_biases, activation=activationfuncs[r])
                
                # Word forward
                
                postive_glove, negative_glove = get_triplet_batch(batches[i], batches_labels[i], labels_glove)
                    #pos
                pos_output, pos_caches = forward_pass(postive_glove, glove_weights, glove_biases, activation=activationfuncs[r])
                    #neg
                neg_output, neg_caches = forward_pass(negative_glove, glove_weights, glove_biases, activation=activationfuncs[r])
                
                loss, grad_a, grad_p, grad_n = triplet_loss(img_output, pos_output, neg_output, margin)
                total_loss += loss
        
        
                # Image Back Propagation
                backprop_generic(batches[i], img_caches, grad_a, img_weights, img_biases, img_vw, img_vb,
                                     activation_fn=activationfuncs[r], activation_deriv=derfuncs[r],
                                     lr=0.01, momentum=0.9)
        
                
                
                # Word Back Propagation (sequential)
                    #pos
                backprop_generic(postive_glove, pos_caches, grad_p, glove_weights, glove_biases, glove_vw, glove_vb,
                                         activation_fn=activationfuncs[r], activation_deriv=derfuncs[r],
                                         lr=0.01, momentum=0.9)
                    #neg
                backprop_generic(negative_glove, neg_caches, grad_n, glove_weights, glove_biases, glove_vw, glove_vb,
                                         activation_fn=activationfuncs[r], activation_deriv=derfuncs[r],
                                         lr=0.01, momentum=0.9)
            
            
            # Print epoch statistics
            avg_loss = total_loss / len(batches)
            
            if epoch % 1 == 0:  # Print every epoch
                test_preds = predict(test_images, img_weights, img_biases,
                                     glove_weights, glove_biases, labels_glove, activation= activationfuncs[r],
                                     batch_size=batch_size)
            
                accuracy = np.mean(test_preds == test_labels)
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
                
        
        glove_embeds = get_glove_embeddings(labels_glove, glove_weights, glove_biases)
        plot_embeddings(glove_embeds, category_names, method="pca")
        
        
        
        
        
        
        
