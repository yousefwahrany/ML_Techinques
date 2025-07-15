import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn import svm

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

# Training using PCA features with different SVM kernels
svm_kernels = ["linear", "rbf"]
results = []

print("\nTraining Using PCA features with SVM classifiers")
for kernel in svm_kernels:
    print(f"\nTraining with {kernel} kernel...")
    
    # Measure training time
    train_start_time = time.time()
    
    # Create and train SVM classifier
    clf = svm.SVC(kernel=kernel)
    clf.fit(pca_train, y_train)
    
    train_time = time.time() - train_start_time
    
    # Measure testing time
    test_start_time = time.time()
    
    # Make predictions
    predictions = clf.predict(pca_test)
    
    test_time = time.time() - test_start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    
    results.append({
        'kernel': kernel,
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time,
        'total_time': train_time + test_time,
        'predictions': predictions
    })
    
    print(f"For {kernel} kernel SVM:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Testing time: {test_time:.2f} seconds")
    print(f"  Total processing time: {train_time + test_time:.2f} seconds")

# Find best model based on accuracy
best_model = max(results, key=lambda x: x['accuracy'])
print(f"\nBest model: SVM with {best_model['kernel']} kernel - {best_model['accuracy']:.2f}% accuracy")

# Display results in a table format
print("\nResults Summary (PCA Features with SVM):")
print("-------------------------------------------------------------")
print("| Kernel   | Accuracy (%) | Processing Time (seconds) |")
print("-------------------------------------------------------------")
for result in results:
    print(f"| {result['kernel']:<8} | {result['accuracy']:11.2f} | {result['total_time']:25.2f} |")
print("-------------------------------------------------------------")

# Plot confusion matrices for all SVM models
plt.figure(figsize=(20, 16))

for i, result in enumerate(results):
    # Calculate confusion matrix for this model
    cm = confusion_matrix(y_test, result['predictions'])
    
    # Create subplot
    plt.subplot(2, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (PCA, SVM with {result["kernel"]} kernel)\nAccuracy: {result["accuracy"]:.2f}%')

plt.tight_layout()
plt.savefig('confusion_matrices_pca_svm_all.png')
plt.show()