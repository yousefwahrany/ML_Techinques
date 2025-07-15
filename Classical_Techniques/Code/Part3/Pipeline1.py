import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import re
import matplotlib.pyplot as plt
import random
import sys
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time

start_time = time.time()


# Define file paths
input_dir_train = r"Reduced_Trainging_data"
input_dir_testing = r"Reduced_Testing_data"

# Ensure directories exist
if not os.path.exists(input_dir_train) or not os.path.exists(input_dir_testing):
    raise FileNotFoundError("One or both dataset directories not found!")

# Load image file paths
images_train = []
images_test = []

for i in range(10):  # Digits 0-9
    train_path = os.path.join(input_dir_train, str(i))
    test_path = os.path.join(input_dir_testing, str(i))

    # Ensure subdirectories exist
    if os.path.exists(train_path):
        images_train.extend(glob.glob(os.path.join(train_path, "*.jpg")))
    if os.path.exists(test_path):
        images_test.extend(glob.glob(os.path.join(test_path, "*.jpg")))

print("Number of training images:", len(images_train))
print("Number of testing images:", len(images_test))

# Ensure we have images before converting
if not images_train or not images_test:
    raise ValueError("No images found in dataset!")

# Load images as NumPy arrays (Grayscale conversion)
X_train = np.array([np.array(Image.open(fname).convert("L")) for fname in images_train])
X_test = np.array([np.array(Image.open(fname).convert("L")) for fname in images_test])

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Extract labels from parent folder (which represents the digit class)
y_train = np.array([int(os.path.basename(os.path.dirname(fname))) for fname in images_train])
y_test = np.array([int(os.path.basename(os.path.dirname(fname))) for fname in images_test])



#Flatten images from (28,28) to (784,)
X_train_flattened = X_train.reshape(len(X_train), -1)   # Shape: (num_images, 784)
X_test_flattened = X_test.reshape(len(X_test), -1)   # Shape: (num_images, 784)



#pca = PCA(n_components=200)  # Reduce to 50 principal components
#X_train_flattened = pca.fit_transform(X_train.reshape(len(X_train), -1))
#X_test_flattened = pca.transform(X_test.reshape(len(X_test), -1))  


print("Shape of X_train_flattened:", X_train_flattened.shape)

# Apply KMeans clustering
num_clusters = 100  # We want to cluster into 100 groups
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_flattened)



"""
def label_clusters(cluster_assignments, image_paths):
"""
"""
    Display 5 random images from each cluster and let the user assign a label.
    Ensure the input box stays active without clicking the terminal.

    Parameters:
        cluster_assignments (list or np.array): Cluster IDs for each image.
        image_paths (list): Paths to the corresponding images.

    Returns:
        dict: A dictionary mapping cluster IDs to user-assigned labels.
    """
"""
    clusters_assigned_labels = {}  # Store user-assigned labels
    clustered_images = {}  # Group image paths by cluster
    
    # Group image paths by their assigned cluster
    for img_path, cluster_id in zip(image_paths, cluster_assignments):
        if cluster_id not in clustered_images:
            clustered_images[cluster_id] = []
        clustered_images[cluster_id].append(img_path)

    # Iterate through clusters and display images
    for cluster_id, images in clustered_images.items():
        num_images = len(images)
        selected_images = random.sample(images, min(5, num_images))  # Pick 5 or all if fewer

        # Show selected images
        fig, axes = plt.subplots(1, len(selected_images), figsize=(10, 3))
        for ax, img_path in zip(axes, selected_images):
            img = plt.imread(img_path)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.suptitle(f"Cluster {cluster_id}: Assign a digit")
        
        # **Ensure input appears first and keeps focus**
        print(f"\nEnter the digit for cluster {cluster_id}: ", end="", flush=True)
        sys.stdout.flush()  # Force the message to appear before images
        plt.draw()
        plt.pause(0.2)  # Keep the figure open
        
        # Read input without losing focus
        label = sys.stdin.read(2).strip()  # Reads 2 characters (digit + newline)
        while not label.isdigit() or int(label) not in range(10):  # Validate input
            print("Invalid input! Enter a digit (0-9): ", end="", flush=True)
            sys.stdout.flush()
            label = sys.stdin.read(2).strip()
        
        clusters_assigned_labels[cluster_id] = int(label)
        plt.close(fig)  # Close the figure after entering the label

    return clusters_assigned_labels  # Dictionary {cluster_id: assigned_label}
    """

# Example usage:
#clusters_assigned_labels = label_clusters(cluster_labels, images_train)

# Assign labels based on cluster assignments
#y_train_clusters_predicted = np.array([clusters_assigned_labels[cluster] for cluster in cluster_labels])
#print("Sample predicted labels:", y_train_clusters_predicted[:10])

# Assign labels to clusters based on a few samples
cluster_to_label = {}


for cluster in range(num_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]  # Find images in cluster
    
    if len(cluster_indices) > 5:
        sampled_indices = np.random.choice(cluster_indices, 5, replace=False)  # Random 5 indices
    else:
        sampled_indices = cluster_indices  # Use all if fewer than 5 exist
    
    sampled_labels = y_train[sampled_indices]  # Get corresponding labels
    majority_label = np.bincount(sampled_labels).argmax()  # Find most common label
    cluster_to_label[cluster] = majority_label  # Assign to cluster


# Assign labels to all images based on their cluster
y_train_clusters_predicted = np.array([cluster_to_label[cluster] for cluster in cluster_labels])
# Compute accuracy
accuracy = accuracy_score(y_train, y_train_clusters_predicted)
print(f"Labeling Accuracy: {accuracy * 100:.2f}%")


# Train an SVM classifier
svm_classifier = SVC(kernel="rbf", random_state=42)
svm_classifier.fit(X_train_flattened, y_train_clusters_predicted)  # Train on labeled images

# Predict on test data
y_train_svm_pred = svm_classifier.predict(X_train_flattened)
y_test_svm_pred = svm_classifier.predict(X_test_flattened)

# Compute accuracy on training data
svm_train_accuracy = accuracy_score(y_train, y_train_svm_pred)
print(f"SVM Training Accuracy: {svm_train_accuracy * 100:.2f}%")
svm_test_accuracy = accuracy_score(y_test, y_test_svm_pred)
print(f"SVM Test Accuracy: {svm_test_accuracy * 100:.2f}%")



changed_labels = np.sum(y_train_svm_pred != y_train_clusters_predicted)
total_images = len(cluster_labels)

# Percentage of changed clusters
#change_percentage = (changed_labels / total_images) * 100
#print(f"Percentage of images that changed clusters: {change_percentage:.2f}%")





# Step 1: Identify uncertain images (where SVM disagrees with clustering)
#uncertain_indices = np.where(y_train_svm_pred != y_train_clusters_predicted)[0]  # Indices of mismatched images

#print(f"Found {len(uncertain_indices)} uncertain images.")

"""
def label_uncertain_images(cluster_assignments, image_paths):
    """
"""
    Display 5 random images from each cluster and let the user assign a label.
    Ensures input box stays active without requiring a terminal click.

    Parameters:
        cluster_assignments (list or np.array): Cluster IDs for each image.
        image_paths (list): Paths to the corresponding images.

    Returns:
        dict: A dictionary mapping cluster IDs to user-assigned labels.
"""
"""
    user_corrected_labels = {}  # Store user-assigned labels
    clustered_images = {}  # Group image paths by cluster
    
    # Group image paths by their assigned cluster
    for img_path, cluster_id in zip(image_paths, cluster_assignments):
        clustered_images.setdefault(cluster_id, []).append(img_path)

    # Iterate through clusters and display images
    for cluster_id, images in clustered_images.items():
        selected_images = random.sample(images, min(5, len(images)))  # Pick 5 or all if fewer

        # Show selected images
        fig, axes = plt.subplots(1, len(selected_images), figsize=(10, 3))
        for ax, img_path in zip(axes, selected_images):
            img = plt.imread(img_path)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.suptitle(f"Cluster {cluster_id}: Assign a digit")
        
        # **Ensure input appears first and remains active**
        print(f"\nEnter the digit for cluster {cluster_id}: ", end="", flush=True)

        plt.show(block=True)  # Block execution until images are closed

        # Read input **AFTER** images are closed
        label = input().strip()  
        while not label.isdigit() or int(label) not in range(10):  # Validate input
            label = input("Invalid input! Enter a digit (0-9): ").strip()

        user_corrected_labels[cluster_id] = int(label)

    return user_corrected_labels  # Dictionary {cluster_id: assigned_label}

# Step 2: Show uncertain images to user and get new labels
user_corrected_labels = label_uncertain_images(uncertain_indices, X_train)

# Step 3: Update labels with user input
for idx, correct_label in user_corrected_labels.items():
    y_train_clusters_predicted[idx] = correct_label  # Update with user-corrected label
"""
end_time = time.time()
execution_time = end_time - start_time
print(f"Total Execution Time: {execution_time:.2f} seconds")

for i in range(5):
    start_time = time.time()
    # Automatically update uncertain labels using ground truth from y_train
    #for idx in uncertain_indices:
    #    y_train_clusters_predicted[idx] = y_train[idx]  # Use actual labels from y_train


    #print("Updated uncertain labels. Now retraining SVM...")
    print("Now retraining SVM...")

    # Step 4: Retrain SVM with corrected labels
    svm_classifier.fit(X_train_flattened, y_train_clusters_predicted)
    # Step 5: Predict again with the retrained SVM
    y_train_svm_pred = svm_classifier.predict(X_train_flattened)
    y_test_svm_pred = svm_classifier.predict(X_test_flattened)
    # Compute new accuracy
    new_accuracy = accuracy_score(y_train, y_train_svm_pred)
    print(f"Updated SVM Training Accuracy: {new_accuracy * 100:.2f}%")
    new_accuracy = accuracy_score(y_test, y_test_svm_pred)
    print(f"Updated SVM Test Accuracy: {new_accuracy * 100:.2f}%")


    #changed_labels = np.sum(y_train_svm_pred != y_train_clusters_predicted)
    #total_images = len(cluster_labels)
    # Percentage of changed clusters
    #change_percentage = (changed_labels / total_images) * 100
    #print(f"Percentage of images that changed labels: {change_percentage:.2f}%")


    # Step 1: Identify uncertain images (where SVM disagrees with clustering)
    #uncertain_indices = np.where(y_train_svm_pred != y_train_clusters_predicted)[0]  # Indices of mismatched images
    #print(f"Found {len(uncertain_indices)} uncertain images.")

    y_train_clusters_predicted = y_train_svm_pred

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Execution Time: {execution_time:.2f} seconds")
