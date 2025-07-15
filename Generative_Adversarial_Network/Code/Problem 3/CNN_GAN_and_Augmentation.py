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

# Use 300 real samples only in the main loop
real = 300  # Fixed number of real samples

# Define arrays for generated and augmented sample sizes
generatedarray = np.array([0, 1000, 2000, 3000])  # Generated samples per digit
augmentedarray = np.array([0, 1000, 2000, 3000])  # Augmented samples per digit

# Dictionary to store results
results = {}

# Loop through the combinations of generated and augmented data
for generate in generatedarray:
    for augment in augmentedarray:
        print(f"\nTraining on dataset with {real} real samples, {generate} generated samples, and {augment} augmented samples per digit...")

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
            if augment > 0:
                digit_aug_indices = (augmented_labels == digit).nonzero(as_tuple=True)[0]
                aug_indices = np.random.choice(digit_aug_indices.cpu().numpy(), augment, replace=False)
                X_augmented_list.append(augmented_images[aug_indices])
                y_augmented_list.append(augmented_labels[aug_indices])

        # Combine real and augmented data for all digits
        X_real = torch.cat(X_real_list, dim=0)
        y_real = torch.cat(y_real_list, dim=0)
        if augment > 0:
            X_augmented = torch.cat(X_augmented_list, dim=0)
            y_augmented = torch.cat(y_augmented_list, dim=0)
        else:
            X_augmented = torch.empty((0, *X_train_tensor.shape[1:]), dtype=torch.float32)
            y_augmented = torch.empty((0,), dtype=torch.long)

        # Load generated data if applicable
        if generate > 0:
            save_path = f"generated_data/gen_image_{real}_{generate}.npz"
            data = np.load(save_path)
            generated_images = data['arr_0']
            generated_images = np.delete(generated_images, 0, axis=0)
            generated_labels = np.repeat(np.arange(10), generate)
            X_generated = torch.tensor(generated_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
            y_generated = torch.tensor(generated_labels, dtype=torch.long)
        else:
            X_generated = torch.empty((0, *X_train_tensor.shape[1:]), dtype=torch.float32)
            y_generated = torch.empty((0,), dtype=torch.long)

        # Combine real, generated, and augmented data
        X_train_final = torch.cat([X_real, X_augmented, X_generated], dim=0)
        y_train_final = torch.cat([y_real, y_augmented, y_generated], dim=0)

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
        results[(generate, augment)] = test_accuracy
        print(f"Final Test Acc: {test_accuracy:.2f}%")
        end_time = time.time()
        print(f"Time taken to test LeNet-5: {round((end_time - start_time) * 100, 1)/100} seconds")

# Evaluate using 1000 real samples only
print("\nEvaluating with 1000 real samples only...")
X_real_list, y_real_list = [], []
for digit in range(10):
    digit_indices = (y_train_tensor == digit).nonzero(as_tuple=True)[0]
    real_indices = np.random.choice(digit_indices.cpu().numpy(), 1000, replace=False)
    X_real_list.append(X_train_tensor[real_indices])
    y_real_list.append(y_train_tensor[real_indices])

X_real = torch.cat(X_real_list, dim=0)
y_real = torch.cat(y_real_list, dim=0)

# Create dataset and data loader
dataset_train = TensorDataset(X_real, y_real)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

# Train & Evaluate
start_time = time.time()
train_model(model, train_loader, epochs=10)
train_accuracy = evaluate(train_loader)
print(f"Final Train Acc (1000 real): {train_accuracy:.2f}%")
end_time = time.time()
print(f"Time taken to train LeNet-5 (1000 real): {round((end_time - start_time) * 100, 1)/100} seconds")

start_time = time.time()
test_accuracy = evaluate(test_loader)
results["1000_real_only"] = test_accuracy
print(f"Final Test Acc (1000 real): {test_accuracy:.2f}%")
end_time = time.time()
print(f"Time taken to test LeNet-5 (1000 real): {round((end_time - start_time) * 100, 1)/100} seconds")

# Print final results table
print("\nFinal Accuracy Table:")
print(f"{'Gen\\Aug':>10} | {'0':>8} | {'1000':>8} | {'2000':>8} | {'3000':>8}")
print("-" * 50)
for generate in generatedarray:
    row = f"{generate:>10} |"
    for augment in augmentedarray:
        acc = results.get((generate, augment), 0.0)  # Access results using (generate, augment)
        row += f" {acc:>7.2f}% |"
    print(row)

# Add the result for 1000 real samples only
print(f"\n{'1000_real_only':>10} | {results['1000_real_only']:>7.2f}%")