import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import random

# Load dataset
def load_images(input_dir):
    images, labels = [], []
    for i in range(10):  # Digits 0-9
        path = os.path.join(input_dir, str(i))
        if os.path.exists(path):
            img_files = glob.glob(os.path.join(path, "*.jpg"))
            for img_path in img_files:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                img = cv2.resize(img, (28, 28))  # Resize to 28x28
                images.append(img)
                labels.append(i)
    return np.array(images), np.array(labels)

# Define dataset paths
input_dir_train = r"F:\Projects\NN_projects\Reduced_Trainging_data"
input_dir_testing = r"F:\Projects\NN_projects\Reduced_Testing_data"

X_train, y_train = load_images(input_dir_train)
X_test, y_test = load_images(input_dir_testing)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Define CNN model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout to reduce overfitting
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train CNN model
cnn_model = build_model()
cnn_model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=15, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
# Evaluate
test_loss, test_acc = cnn_model.evaluate(X_train, y_train)
print(f"Final Train Accuracy: {test_acc * 100:.2f}%")
