import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Function to create the ReducedMNIST dataset (1000 examples per class for training, 200 for testing)
def create_reduced_mnist(random_seed = 1000):
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

def extract_dct_features(images, n_coefficients=225):
    """
    Extract DCT features from images.
    
    Parameters:
    images: array of shape (n_samples, height*width) - flattened images
    n_coefficients: number of DCT coefficients to keep (should be a square number)
    
    Returns:
    dct_features: array of shape (n_samples, n_coefficients)
    """
    n_samples = images.shape[0]
    img_size = int(np.sqrt(images.shape[1]))  # Assuming square images
    side_length = int(np.sqrt(n_coefficients))
    
    # Reshape to square images
    square_images = images.reshape(n_samples, img_size, img_size)
    
    # Initialize DCT feature array
    dct_features = np.zeros((n_samples, n_coefficients))
    
    for i in range(n_samples):
        # Apply 2D DCT
        img_dct = dct(dct(square_images[i].T, norm='ortho').T, norm='ortho')
        
        # Keep only the top-left n_coefficients (lower frequencies)
        dct_features[i] = img_dct[:side_length, :side_length].flatten()
    
    return dct_features

# Define function to build MLP model
def build_mlp_model(input_shape, num_hidden_layers=1, hidden_units=128):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Add hidden layers
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(hidden_units, activation='relu'))
    
    # Output layer for 10 classes
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define function to train MLP model and evaluate testing accuracy and training and testing time
def train_model(model, train_data, test_data, train_label, test_label, num_hidden_layers):
    print(f"MLP({num_hidden_layers} hidden layers)")
    start_time = time.time()
    # Train the Model
    history = model.fit(train_data, train_label, epochs=20, batch_size=128)
    # Training Time
    training_time = time.time() - start_time

    start_time = time.time()
    # Testing Accuracy
    test_loss, test_acc = model.evaluate(test_data, test_label, batch_size=128, verbose=0)
    # Testing Time
    testing_time = time.time() - start_time

    print(model.summary())
    print(f"Training time: {training_time} seconds")
    print(f"Testing time: {testing_time} seconds")
    print("Test accuracy:", test_acc * 100)

# Define function to preform the Model for all number of layers
def MLP_train_evaluate(train_data, test_data, train_label, test_label, hidden_layers):
    for num_hidden_layers in hidden_layers:
        mlp_model = build_mlp_model(input_shape=(train_data.shape[1],), num_hidden_layers=num_hidden_layers)
        train_model(mlp_model, train_data, test_data, train_label, test_label, num_hidden_layers)
        print()

# Main execution
# Create ReducedMNIST dataset
X_train, y_train, X_test, y_test = create_reduced_mnist()

# Save Data for later use
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print(f"Training set: {X_train.shape[0]} examples")
print(f"Test set: {X_test.shape[0]} examples")

# Extract DCT features
print("Extracting DCT features...")
dct_train_features = extract_dct_features(X_train)
dct_test_features = extract_dct_features(X_test)

print(f"DCT features shape - Train: {dct_train_features.shape}, Test: {dct_test_features.shape}")
print("DCT features extraction completed and saved.")

# Save features for later use
np.save('dct_train_features.npy', dct_train_features)
np.save('dct_test_features.npy', dct_test_features)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Convert labels to categorical (one-hot encoding for 10 classes)
labels_training = keras.utils.to_categorical(y_train, num_classes=10)
labels_test = keras.utils.to_categorical(y_test, num_classes=10)

# Perform MLP for DCT Features
MLP_train_evaluate(dct_train_features, dct_test_features, labels_training, labels_test, [1, 3, 5])