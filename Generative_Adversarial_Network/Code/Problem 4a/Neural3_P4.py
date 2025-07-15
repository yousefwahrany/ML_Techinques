import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Concatenate

# Function to create the ReducedMNIST dataset (1000 examples per class for training, 200 for testing)
def create_reduced_mnist(random_seed = 1005):
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0  # Normalize pixel values to [0, 1]

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    reduced_X_train = []
    reduced_y_train = []
    reduced_X_test = []
    reduced_y_test = []
    
    # For each digit (0-9)
    for digit in range(10):
        digit_indices = np.where(y.astype(int) == digit)[0]
        
        # Randomly select 1000 for training and 200 for testing
        selected_indices = np.random.choice(digit_indices, 1200, replace=False)
        train_indices = selected_indices[:1000]
        test_indices = selected_indices[1000:1200]
        
        # Add to our reduced datasets
        reduced_X_train.append(X.iloc[train_indices])
        reduced_y_train.append(y.iloc[train_indices])
        reduced_X_test.append(X.iloc[test_indices])
        reduced_y_test.append(y.iloc[test_indices])
    
    # Concatenate all digits
    reduced_X_train = np.vstack(reduced_X_train)
    reduced_y_train = np.concatenate(reduced_y_train).astype(int)
    reduced_X_test = np.vstack(reduced_X_test)
    reduced_y_test = np.concatenate(reduced_y_test).astype(int)
    
    return reduced_X_train, reduced_y_train, reduced_X_test, reduced_y_test


# Define a basic CNN model
def create_basic_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=3, keepdims=True)
        max_pool = tf.reduce_max(x, axis=3, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=3)
        attention_map = self.conv(concat)
        return x * attention_map


# Define a CNN model with spatial attention
def create_attention_cnn():
    inputs = keras.Input(shape=(28, 28, 1))
    
    # First convolutional block
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Apply spatial attention after first block
    x = SpatialAttention()(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Apply spatial attention after second block
    x = SpatialAttention()(x)
   
    # Third convolutional block
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    
    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train and evaluate a model
def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...")

    start_time = time.time()
    # Train the model
    history = model.fit( X_train, y_train_cat, batch_size=128, epochs=10, validation_split=0.1, verbose=1)
    training_time = time.time() - start_time
    
    # Evaluate the model
    print(f"\nEvaluating {model_name}...")
    start_time = time.time()
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    testing_time = time.time() - start_time

    print(f"{model_name} test accuracy: {test_acc:.4f}")
    print(f"{model_name} training time: {training_time:.2f} seconds")
    print(f"{model_name} testing time:  {testing_time:.2f} seconds")
    
    # Generate predictions and confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    return test_acc, training_time, history, testing_time

# //////////////////////////////////////////
# ------------ Main execution ------------ #
# //////////////////////////////////////////

# Create ReducedMNIST dataset
X_train, y_train, X_test, y_test = create_reduced_mnist()

# Reshape the data for CNN input (adding channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create and train both models
basic_cnn = create_basic_cnn()

attention_cnn = create_attention_cnn()

# Print model summaries
print("\nBasic CNN Model Summary:")
basic_cnn.summary()

print("\nAttention CNN Model Summary:")
attention_cnn.summary()

# Train and evaluate models
basic_acc, basic_train_time, basic_history, basic_test_time = train_and_evaluate(basic_cnn, "Basic CNN")
attention_acc, attention_train_time, attention_history, attention_test_time = train_and_evaluate(attention_cnn, "Attention CNN")

# Compare the models
print("\n--- Model Comparison ---")
print(f"Basic CNN accuracy: {basic_acc:.4f}, training time: {basic_train_time:.2f} seconds, testing time: {basic_test_time:.2f} seconds")
print(f"Attention CNN accuracy: {attention_acc:.4f}, training time: {attention_train_time:.2f} seconds, testing time: {attention_test_time:.2f} seconds")
print(f"Accuracy difference: {(attention_acc - basic_acc):.4f}")
print(f"Training time difference: {(attention_train_time - basic_train_time):.2f} seconds")
print(f"Testing time difference:  {(attention_test_time - basic_test_time):.2f} seconds")
