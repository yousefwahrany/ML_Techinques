import torch
import time
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

# Load the dataset (images and labels)
images, labels, images_tst, labels_tst = create_reduced_mnist()

# Number of samples to select per class
samples_per_class = 40  # Adjust as needed

# Dictionary to store selected samples
selected_images = []
selected_labels = []

# Loop through all 10 classes
for class_id in range(10):
    # Find indices of all images belonging to this class
    class_indices = np.where(labels == class_id)[0].tolist()

    # Randomly select `samples_per_class` images
    selected_indices = random.sample(class_indices, samples_per_class)

    # Store the selected images and labels
    selected_images.extend(images[selected_indices])
    selected_labels.extend(labels[selected_indices])

selected_images = selected_images

# Define augmentation transformations
rotate_5_right = transforms.RandomRotation(degrees=(0, 10))
rotate_5_left = transforms.RandomRotation(degrees=(-10, 0))
shift_y = transforms.RandomAffine(degrees=0, translate=(0, 0.1))
shift_x = transforms.RandomAffine(degrees=0, translate=(0.1, 0))


# Function to add Gaussian noise
def add_gaussian_noise(img, mean=0, std=0.01):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0, 1)  # Ensure pixel values stay in range [0,255]

# Augmented data lists
augmented_images = []
augmented_labels = []

# Apply augmentations
for img, label in zip(selected_images, selected_labels):
    img_tensor = torch.tensor(img, dtype=torch.float32).reshape(1, 28, 28).clone()  # Ensure we don’t modify the original list data

    augmented_images.extend([
        img_tensor,  # Original
        rotate_5_right(img_tensor),
        rotate_5_left(img_tensor),
        rotate_5_right(img_tensor),
        rotate_5_left(img_tensor),
        shift_y(img_tensor),
        shift_x(img_tensor),
        shift_y(img_tensor),
        shift_x(img_tensor),
        shift_x(img_tensor),
        shift_y(img_tensor),
        #add_gaussian_noise(img_tensor),
    ])
    augmented_labels.extend([label] * 11)  # Each transformation keeps the same label


augmented_images = np.array(augmented_images)
total_train_set = np.array(images)
images_tst = np.array(images_tst)
#augmented_images = augmented_images.reshape(augmented_images.shape[0], 28, 28)

augmented_images = augmented_images.reshape(augmented_images.shape[0], -1)
total_train_set = total_train_set.reshape(total_train_set.shape[0], -1)
tst_set = images_tst.reshape(images_tst.shape[0], -1)

print("Initial Model Training:")
start_time = time.time()
svm_model = svm.SVC()
svm_model.fit(augmented_images, augmented_labels)

labels_pred = svm_model.predict(total_train_set)
accuracy = accuracy_score(labels, labels_pred)
print(f"- Iteration 0 Accuracy: {accuracy*100}%")
print(f"Iteration 0 Time: {time.time() - start_time}s")

print("Iterative Refinement:")
for i in range(5):
    start_time = time.time()
    svm_model.fit(total_train_set, labels_pred)
    labels_pred = svm_model.predict(total_train_set)
    accuracy = accuracy_score(labels, labels_pred)
    print(f"- Iteration {str(i+1)} Accuracy: {accuracy*100}%")
    print(f"Iteration {str(i+1)} Time: {time.time() - start_time}s")

start_time = time.time()
labels_pred = svm_model.predict(tst_set)
accuracy = accuracy_score(labels_tst, labels_pred)
print(f"Accuracy based on Test Set: {accuracy*100}%")
print(f"Testing Time: {time.time() - start_time}s")

svm_model.fit(total_train_set, labels)
labels_pred = svm_model.predict(tst_set)
accuracy = accuracy_score(labels_tst, labels_pred)
print(f"Test set Accuracy after training on 10,000 labeled set: {accuracy*100}%")