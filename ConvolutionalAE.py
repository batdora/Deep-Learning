import os
import numpy as np
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib import cm
import time
# ————————————————————————————
# 1) Load & normalize
train_images = np.load('train_images.npy')[..., None] / 255.0   # shape=(N_train,28,28,1)
test_images  = np.load('test_images.npy')[...,  None] / 255.0   # shape=(N_test,28,28,1)
train_labels = np.load('train_labels.npy')                      # shape=(N_train,)
test_labels  = np.load('test_labels.npy')                       # shape=(N_test,)
train_images= (train_images > 0.3).astype(np.float32)
test_images = (test_images > 0.3).astype(np.float32)

# latend_dim_list = [16, 32, 64, 128]
latend_dim_list = [128]
# kernel_size_list = [3, 5, 7]
kernel_size_list = [3]
# activations = ['relu','selu','layers.LeakyReLU(alpha=0.2)','elu','gelu']
activations = ['selu']
# ————————————————————————————
os.makedirs("images", exist_ok=True)
for act in activations:
    for kernel_size in kernel_size_list:
        for latent_dim in latend_dim_list:
            encoder_input = layers.Input(shape=(28,28,1), name='enc_input')
            x = layers.Conv2D(32, kernel_size, strides=2, padding='same', activation=act, name='enc_conv1')(encoder_input)  # 28→14
            x = layers.Conv2D(64, kernel_size, strides=2, padding='same', activation=act, name='enc_conv2')(x)              # 14→7
            # x = layers.Conv2D(128,kernel_size, strides=2, padding='same', activation=act, name='enc_conv3')(x)             # 7→4
            x = layers.Flatten(name='enc_flatten')(x)
            bottleneck = layers.Dense(latent_dim, activation=None, name='bottleneck')(x)

            encoder = Model(encoder_input, bottleneck, name='encoder')

            # ————————————————————————————
            # 3) Decoder with Cropping
            decoder_input = layers.Input(shape=(latent_dim,), name='dec_input')
            # Make the 7*7 below 4,4 if you use 3 conv hidden layer and uncomment cropping, 
            # 14,14 if you use 1 conv layer and keep cropping commented
            x = layers.Dense(7*7*64, activation=act, name='dec_dense')(decoder_input)
            x = layers.Reshape((7,7,64), name='dec_reshape')(x)
            # x = layers.Conv2DTranspose(128, kernel_size, strides=2, padding='same', activation=act, name='dec_deconv1')(x)  # 4→8
            x = layers.Conv2DTranspose(64, kernel_size, strides=2, padding='same', activation=act, name='dec_deconv2')(x)  # 8→16
            x = layers.Conv2DTranspose(32, kernel_size, strides=2, padding='same', activation=act, name='dec_deconv3')(x)  # 16→32
            # x = layers.Cropping2D(cropping=((2,2),(2,2)), name='crop_to_28')(x)                                     # 32→28
            decoder_output = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid', name='dec_output')(x)

            decoder = Model(decoder_input, decoder_output, name='decoder')

            # ————————————————————————————
            # 4) Autoencoder = Encoder + Decoder
            ae_input      = encoder_input
            ae_bottleneck = encoder(ae_input)
            ae_output     = decoder(ae_bottleneck)
            conv_autoencoder = Model(ae_input, ae_output, name='conv_autoencoder')
            conv_autoencoder.compile(optimizer='adam', loss='mse')

            start_time = time.time()
            history = conv_autoencoder.fit(
                train_images, train_images,
                validation_split=0.2,
                epochs=20,
                batch_size=128
            )
            end_time = time.time()
            training_time = end_time - start_time

            final_train_loss = history.history['loss'][-1]
            final_val_loss   = history.history['val_loss'][-1]

            with open("output.txt", "a") as f:
                f.write(f"Activation Func = {act}")
                f.write(f"Kernel Size = {kernel_size}, Latent Dim = {latent_dim}\n")
                f.write(f"Final Training MSE: {final_train_loss:.5f}\n")
                f.write(f"Final Validation MSE: {final_val_loss:.5f}\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n")
                f.write("\n")  

        plt.figure()
        plt.plot(history.history['loss'], label='Train MSE')
        plt.plot(history.history['val_loss'], label='Val MSE')
        plt.title(f'Conv AE MSE (k={kernel_size}, latent={latent_dim})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plot_filename = f'images/loss_k{kernel_size}_l{latent_dim}_{act}.png'
        plt.savefig(plot_filename)
        plt.close()  


        
        encoder_model = encoder  
        test_embeddings = encoder_model.predict(test_images)  

        # 7) t-SNE → 2D
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        embeddings_2d = tsne.fit_transform(test_embeddings)  
        cmap = ListedColormap(cm.tab10.colors[:5])
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(
            embeddings_2d[:,0],
            embeddings_2d[:,1],
            c= test_labels.astype(int),
            cmap=cmap,
            s=5
        )
        plt.title(f't-SNE (k={kernel_size}, latent={latent_dim})')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.tight_layout()

        tsne_filename = f'images/tsne_k{kernel_size}_l{latent_dim}_{act}.png'
        plt.savefig(tsne_filename)
        plt.close()
