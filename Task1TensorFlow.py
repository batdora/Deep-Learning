import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Load and normalize data
train_images = np.load('train_images.npy') / 255
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy') / 255
test_labels = np.load('test_labels.npy')

# One-hot encode labels
encoded_train_labels = to_categorical(train_labels, num_classes=5)
encoded_test_labels = to_categorical(test_labels, num_classes=5)

# Flatten images
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Define model
model = Sequential([
    Dense(128, input_shape=(784,), activation="sigmoid"),
    Dense(5, activation="softmax")
])

# Compile model
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_images_flat,
    encoded_train_labels,
    batch_size=20,
    epochs=40,
    shuffle=True,
    verbose=2,
    validation_split=0.2
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images_flat, encoded_test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Predictions for metrics
pred_probs = model.predict(test_images_flat)
pred_classes = np.argmax(pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, pred_classes))


# Plot loss and accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("keras_loss_accuracy_plot.png")
plt.show()

