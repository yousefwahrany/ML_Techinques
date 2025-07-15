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

# Define arrays for real and generated sample sizes
generatedarray = np.array([0, 1000, 2000, 3000])  # Generated samples per digit
sizearray = np.array([300, 700, 1000])  # Real samples per digit

# Dictionary to store results
results = {}

# Loop through the combinations of real and generated data
for size in sizearray:
    for generate in generatedarray:
        print(f"\nTraining on dataset with {size} real samples and {generate} generated samples per digit...")

        # Select real data dynamically based on the size
        selected_images = []
        selected_labels = []
        for i in range(10):  # For each digit (0-9)
            digit_indices = np.where(labels == i)[0]
            real_indices = np.random.choice(digit_indices, size, replace=False)
            selected_images.append(images[real_indices])
            selected_labels.append(labels[real_indices])

        # Concatenate the selected samples into one array
        selected_images = np.concatenate(selected_images, axis=0)
        selected_labels = np.concatenate(selected_labels, axis=0)

        if generate > 0:
            # Load generated data
            save_path = f"generated_data/gen_image_{size}_{generate}.npz"
            data = np.load(save_path)
            generated_images = data['arr_0']
            generated_images = np.delete(generated_images, 0, axis=0)
            generated_labels = np.repeat(np.arange(10), generate)

            # Combine real and generated data
            augmented_images = np.concatenate((generated_images, selected_images), axis=0)
            augmented_labels = np.concatenate((generated_labels, selected_labels), axis=0)
        else:
            # Use only real data when generate == 0
            augmented_images = selected_images
            augmented_labels = selected_labels

        # Convert dataset to PyTorch tensors
        augmented_images = torch.tensor(augmented_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        X_test_tensor = torch.tensor(images_tst, dtype=torch.float32).unsqueeze(1)
        augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
        y_test_tensor = torch.tensor(labels_tst, dtype=torch.long)

        # Create dataset and data loader
        dataset_train = TensorDataset(augmented_images, augmented_labels)
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
        results[(size, generate)] = test_accuracy
        print(f"Final Test Acc: {test_accuracy:.2f}%")
        end_time = time.time()
        print(f"Time taken to test LeNet-5: {round((end_time - start_time) * 100, 1)/100} seconds")

# Print final results table
print("\nFinal Accuracy Table:")
print(f"{'Gen\\Real':>10} | {'300':>8} | {'700':>8} | {'1000':>8}")
print("-" * 40)
for generate in generatedarray:
    row = f"{generate:>10} |"
    for size in sizearray:
        acc = results.get((size, generate), 0.0)
        row += f" {acc:>7.2f}% |"
    print(row)