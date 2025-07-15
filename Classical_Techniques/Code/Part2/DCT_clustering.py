import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

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

# Training using DCT features
n_clusters_list = [1, 4, 16, 32]
results = []

print("Training Using DCT features")
for n_clusters_per_class in n_clusters_list:
    print(f"\nTraining with {n_clusters_per_class} clusters per class...")
    
    # Measure training time
    train_start_time = time.time()
    
    # Initialize centroids array and labels array for each centroid
    centroids = []
    centroid_labels = []
    
    for digit in range(10):
        # Get features for this digit
        digit_features = dct_train_features[y_train == digit]
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters_per_class, random_state=0, n_init="auto").fit(digit_features)
        
        # Store centroids and their corresponding digit label
        centroids.append(kmeans.cluster_centers_)
        centroid_labels.extend([digit] * n_clusters_per_class)
    
    # Combine all centroids
    centroids = np.vstack(centroids)
    centroid_labels = np.array(centroid_labels)
    
    train_time = time.time() - train_start_time
    
    # Measure testing time
    test_start_time = time.time()
    
    # Vectorized approach for classification
    predictions = []
    for test_im in dct_test_features:
        # Calculate Euclidean distances efficiently
        distances = np.sqrt(np.sum((centroids - test_im)**2, axis=1))
        # Find the closest centroid
        closest_centroid_idx = np.argmin(distances)
        # Get the label of the closest centroid
        predicted_label = centroid_labels[closest_centroid_idx]
        predictions.append(predicted_label)
    
    predictions = np.array(predictions)
    test_time = time.time() - test_start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    
    results.append({
        'n_clusters': n_clusters_per_class,
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time,
        'total_time': train_time + test_time,
        'predictions': predictions
    })
    
    print(f"For {n_clusters_per_class} clusters per class:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Testing time: {test_time:.2f} seconds")
    print(f"  Total processing time: {train_time + test_time:.2f} seconds")

# Find best model based on accuracy
best_model = max(results, key=lambda x: x['accuracy'])
print(f"\nBest model: {best_model['n_clusters']} clusters per class with {best_model['accuracy']:.2f}% accuracy")

# Replace the single confusion matrix plot with a grid of confusion matrices for all models
plt.figure(figsize=(20, 16))

for i, result in enumerate(results):
    # Calculate confusion matrix for this model
    cm = confusion_matrix(y_test, result['predictions'])
    
    # Create subplot
    plt.subplot(2, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (DCT, K-means with {result["n_clusters"]} clusters)\nAccuracy: {result["accuracy"]:.2f}%')

plt.tight_layout()
plt.savefig('confusion_matrices_dct_kmeans_all.png')
plt.show()

# Display results in a table format
print("\nResults Summary (DCT Features):")
print("--------------------------------------------------------")
print("| Clusters | Accuracy (%) | Processing Time (seconds) |")
print("--------------------------------------------------------")
for result in results:
    print(f"| {result['n_clusters']:8d} | {result['accuracy']:11.2f} | {result['total_time']:25.2f} |")
print("--------------------------------------------------------")