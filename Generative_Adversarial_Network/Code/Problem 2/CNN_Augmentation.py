import torch
import time
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


# Function to create the ReducedMNIST dataset (1000 examples per class for training, 200 for testing)
def create_reduced_mnist(random_seed=1000, save_path="mnist_reduced.npz"):
    # Check if dataset already exists
    if os.path.exists(save_path):
        print("Loading MNIST dataset from local storage...")
        data = np.load(save_path)
        return (data['X_train'], data['y_train'], data['X_test'], data['y_test'])
    
    # Otherwise, download the dataset
    print("Downloading MNIST dataset...")
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
        reduced_X_train.append(X.iloc[train_indices].to_numpy())
        reduced_y_train.append(y.iloc[train_indices].to_numpy().astype(int))
        reduced_X_test.append(X.iloc[test_indices].to_numpy())
        reduced_y_test.append(y.iloc[test_indices].to_numpy().astype(int))
    
    # Concatenate all digits
    reduced_X_train = np.vstack(reduced_X_train).reshape(-1, 28, 28)
    reduced_y_train = np.concatenate(reduced_y_train)
    reduced_X_test = np.vstack(reduced_X_test).reshape(-1, 28, 28)
    reduced_y_test = np.concatenate(reduced_y_test)

    # Save to disk for future use
    np.savez_compressed(save_path, X_train=reduced_X_train, y_train=reduced_y_train, 
                         X_test=reduced_X_test, y_test=reduced_y_test)
    print(f"Dataset saved to {save_path}")
    
    return reduced_X_train, reduced_y_train, reduced_X_test, reduced_y_test

# Load the dataset (images and labels)
images, labels, images_tst, labels_tst = create_reduced_mnist()

# Define data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

# Convert dataset to PyTorch tensors
X_train_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
X_test_tensor = torch.tensor(images_tst, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(labels, dtype=torch.long)
y_test_tensor = torch.tensor(labels_tst, dtype=torch.long)

augmented_images = []
augmented_labels = []

for img, label in zip(X_train_tensor, y_train_tensor):
    pil_image = transforms.ToPILImage()(img)

    augmented_images.extend([
        img,  # Original
        transform(pil_image),
        transform(pil_image),
        transform(pil_image),
    ])
    augmented_labels.extend([label] * 4)  # Each transformation keeps the same label


augmented_images = torch.stack(augmented_images)
augmented_labels = torch.stack(augmented_labels)

# Define Modified LeNet-5 for 28x28 Input
class LeNet5(nn.Module):
    def __init__(model):
        super().__init__()
        model.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 28x28x1 -> 28x28x6
        model.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28x6 -> 14x14x6
        model.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14x14x6 -> 10x10x16
        model.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10x16 -> 5x5x16
    
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  # 16*5*5 as per the image
            nn.ReLU(),
            nn.Linear(120, 84),  
            nn.ReLU(),
            nn.Linear(84, 10),  
            #nn.Tanh(),
        )

    def forward(model, x):
        x = model.conv1(x)
        x = model.pool1(x)
        x = model.conv2(x)
        x = model.pool2(x)
        x = model.fc(x)
        return x

# Training function
def train_model(model, loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}, Acc: {100*correct/total:.2f}%")

# Evaluation function
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Define the combinations of augmented and real data
augmentedarray = np.array([0, 1000, 2000, 3000])  # Number of augmented samples
realarray = np.array([300, 700, 1000])  # Number of real samples

# Dictionary to store results
results = {}

# Loop through the 12 datasets
for real in realarray:  # Outer loop for real samples
    for augmented in augmentedarray:  # Inner loop for augmented samples
        print(f"\nTraining on dataset with {real} real samples and {augmented} augmented samples...")

        # Initialize lists to store data for all digits
        X_real_list, y_real_list = [], []
        X_augmented_list, y_augmented_list = [], []

        # Process each digit (0-9) separately
        for digit in range(10):
            # Select real data for the current digit
            digit_indices = (y_train_tensor == digit).nonzero(as_tuple=True)[0]
            real_indices = np.random.choice(digit_indices.cpu().numpy(), real, replace=False)
            X_real_list.append(X_train_tensor[real_indices])
            y_real_list.append(y_train_tensor[real_indices])

            # Generate augmented data for the current digit
            if augmented > 0:
                digit_aug_indices = (augmented_labels == digit).nonzero(as_tuple=True)[0]
                aug_indices = np.random.choice(digit_aug_indices.cpu().numpy(), augmented, replace=False)
                X_augmented_list.append(augmented_images[aug_indices])
                y_augmented_list.append(augmented_labels[aug_indices])

        # Combine real and augmented data for all digits
        X_real = torch.cat(X_real_list, dim=0)
        y_real = torch.cat(y_real_list, dim=0)
        if augmented > 0:
            X_augmented = torch.cat(X_augmented_list, dim=0)
            y_augmented = torch.cat(y_augmented_list, dim=0)
        else:
            X_augmented = torch.empty((0, *X_train_tensor.shape[1:]), dtype=torch.float32)
            y_augmented = torch.empty((0,), dtype=torch.long)

        X_train_final = torch.cat([X_real, X_augmented], dim=0)
        y_train_final = torch.cat([y_real, y_augmented], dim=0)

        # Create dataset and data loader
        dataset_train = TensorDataset(X_train_final, y_train_final)
        dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

        # Initialize model, loss, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet5().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        # Train & Evaluate
        start_time = time.time()
        train_model(model, train_loader, epochs=10)
        train_accuracy = evaluate(train_loader)
        print(f"Final Train Acc: {train_accuracy:.2f}%")
        end_time = time.time()
        print(f"Time taken to train LeNet-5: {round((end_time - start_time) * 100, 1)/100} seconds")
        
        start_time = time.time()
        test_accuracy = evaluate(test_loader)
        results[(real, augmented)] = test_accuracy  
        print(f"Final Test Acc: {test_accuracy:.2f}%")
        end_time = time.time()
        print(f"Time taken to test LeNet-5: {round((end_time - start_time) * 100, 1)/100} seconds")

# Print final results table
print("\nFinal Accuracy Table:")
print(f"{'Aug\\Real':>10} | {'300':>8} | {'700':>8} | {'1000':>8}")
print("-" * 40)
for augmented in augmentedarray:
    row = f"{augmented:>10} |"
    for real in realarray:
        acc = results.get((real, augmented), 0.0)  # Access results using (real, augmented)
        row += f" {acc:>7.2f}% |"
    print(row)