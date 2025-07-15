import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
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

def extract_pca_features(X_train, X_test, variance_threshold=0.95):
    """
    Extract PCA features from images.
    
    Parameters:
    X_train: Training images array of shape (n_samples, n_features)
    X_test: Test images array of shape (n_samples, n_features)
    variance_threshold: Amount of variance to preserve (between 0 and 1)
    
    Returns:
    pca_train: PCA-transformed training data
    pca_test: PCA-transformed test data
    n_components: Number of components used
    explained_variance_ratio: Explained variance ratio for each component
    """
    # Initialize PCA without specifying number of components yet
    pca = PCA(n_components=None, random_state=42)
    
    # Fit PCA on training data
    pca.fit(X_train)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components needed to explain variance_threshold of variance
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Number of components needed for {variance_threshold*100}% variance: {n_components}")
    
    # Re-initialize PCA with the determined number of components
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit and transform the data
    pca_train = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)
    
    return pca_train, pca_test, n_components, pca.explained_variance_ratio_


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

# Extract PCA features
print("Extracting PCA features...")
pca_train, pca_test, n_components, explained_variance = extract_pca_features(X_train, X_test)

print(f"PCA features shape - Train: {pca_train.shape}, Test: {pca_test.shape}")
print("PCA features extraction completed and saved.")

# Save features for later use
np.save('pca_train_features.npy', pca_train)
np.save('pca_test_features.npy', pca_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Convert labels to categorical (one-hot encoding for 10 classes)
labels_training = keras.utils.to_categorical(y_train, num_classes=10)
labels_test = keras.utils.to_categorical(y_test, num_classes=10)

# Perform MLP for DCT Features
MLP_train_evaluate(pca_train, pca_test, labels_training, labels_test, [1, 3, 5])