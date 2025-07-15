import os
import glob
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from PIL import Image
import random
import time

start_time = time.time()
# Load dataset (same as Pipeline 1)
def load_images(input_dir):
    images, labels = [], []
    for i in range(10):
        path = os.path.join(input_dir, str(i))
        if os.path.exists(path):
            img_files = glob.glob(os.path.join(path, "*.jpg"))
            images.extend(img_files)
            labels.extend([i] * len(img_files))
    return np.array(images), np.array(labels)

# Define dataset paths
input_dir_train = r"F:\Projects\NN_projects\Reduced_Trainging_data"
input_dir_testing = r"F:\Projects\NN_projects\Reduced_Testing_data"

X_train_paths, y_train = load_images(input_dir_train)
X_test_paths, y_test = load_images(input_dir_testing)

# Convert images to grayscale arrays
def preprocess_images(image_paths):
    return np.array([np.array(Image.open(fname).convert("L")) for fname in image_paths])

X_train = preprocess_images(X_train_paths)
X_test = preprocess_images(X_test_paths)

# Apply PCA for dimensionality reduction
#pca = PCA(n_components=50)
#X_train_flatten = pca.fit_transform(X_train.reshape(len(X_train), -1))
#X_test_flatten = pca.transform(X_test.reshape(len(X_test), -1))

#Flatten images from (28,28) to (784,)
X_train_flatten = X_train.reshape(len(X_train), -1)   # Shape: (num_images, 784)
X_test_flatten = X_test.reshape(len(X_test), -1)   # Shape: (num_images, 784)

# Step 1: Select a small labeled subset for initial training
initial_sample_size = 20  # Number of labeled samples per class
labeled_indices = []

for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    labeled_indices.extend(np.random.choice(digit_indices, initial_sample_size, replace=False))

X_labeled = X_train_flatten[labeled_indices]
y_labeled = y_train[labeled_indices]

# Step 2: Train an initial weak SVM classifier
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_labeled, y_labeled)

# Step 3: Active Learning Loop
num_iterations = 5 # Number of active learning cycles
num_samples_per_iteration = 50

total_labeled = len(labeled_indices)

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}:")
    
    # Find uncertain samples
    unlabeled_indices = np.setdiff1d(np.arange(len(y_train)), labeled_indices)
    X_unlabeled = X_train_flatten[unlabeled_indices]
    
    def select_uncertain_samples(svm, X_unlabeled, unlabeled_indices, num_samples=50):
        probs = svm.predict_proba(X_unlabeled)
        uncertainty = np.max(probs, axis=1)  # Low confidence means high uncertainty
        uncertain_samples = np.argsort(uncertainty)[:num_samples]  # Pick most uncertain samples
        return unlabeled_indices[uncertain_samples]
    
    uncertain_indices = select_uncertain_samples(svm, X_unlabeled, unlabeled_indices, num_samples_per_iteration)
    y_uncertain = y_train[uncertain_indices]  # In real active learning, human annotation happens here
    
    # Add new labeled samples
    X_labeled = np.vstack((X_labeled, X_train_flatten[uncertain_indices]))
    y_labeled = np.concatenate((y_labeled, y_uncertain))
    labeled_indices = np.concatenate((labeled_indices, uncertain_indices))
    total_labeled += len(uncertain_indices)
    
    # Retrain SVM
    svm.fit(X_labeled, y_labeled)
    
    # Evaluate
    y_pred = svm.predict(X_train_flatten)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Train Accuracy after iteration {iteration + 1}: {accuracy * 100:.2f}%")
    y_pred1 = svm.predict(X_test_flatten)
    accuracy = accuracy_score(y_test, y_pred1)
    print(f"Test Accuracy after iteration {iteration + 1}: {accuracy * 100:.2f}%")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    start_time = time.time()
# Final Evaluation

print(f"Total labeled samples at the end: {total_labeled - 10 * initial_sample_size}")
