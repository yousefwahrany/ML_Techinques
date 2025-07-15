import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import time  # Import time module for timing

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply channel attention
        channel_attn = self.channel_attention(x)
        x = x * channel_attn
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(x)
        x = x * spatial_attn
        
        return x

class AudioMNISTCNN(nn.Module):
    def __init__(self, use_attention=False):
        super(AudioMNISTCNN, self).__init__()
        self.use_attention = use_attention
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Attention modules
        if use_attention:
            self.attn1 = AttentionModule(32)
            self.attn2 = AttentionModule(64)
            self.attn3 = AttentionModule(128)
            self.attn4 = AttentionModule(256)
        
        # Classification layers
        # Input image is 28x28, after 3 max pooling layers: 28/2/2/2 = 3.5 -> 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_attention:
            x = self.attn1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_attention:
            x = self.attn2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_attention:
            x = self.attn3(x)
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        if self.use_attention:
            x = self.attn4(x)
        
        # Global pooling and classification
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.digits = []
        self.digit_indices = {}
        
        # Load all image paths and labels
        for digit in range(10):
            digit_dir = os.path.join(self.root_dir, str(digit))
            if os.path.exists(digit_dir):
                digit_samples = []
                for file_name in os.listdir(digit_dir):
                    if file_name.endswith('_spectrogram.png'):
                        image_path = os.path.join(digit_dir, file_name)
                        digit_samples.append((image_path, digit))
                
                self.samples.extend(digit_samples)
                
                # Store indices for each digit
                start_idx = len(self.samples) - len(digit_samples)
                self.digit_indices[digit] = list(range(start_idx, len(self.samples)))
                self.digits.append(digit)
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_balanced_subset(self, samples_per_digit, seed=42):
        """
        Create a subset with a fixed number of samples for each digit.
        
        Args:
            samples_per_digit: Number of samples to include per digit
            seed: Random seed for reproducibility
            
        Returns:
            Subset of the dataset with balanced samples
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        selected_indices = []
        for digit in range(10):
            if digit in self.digit_indices:
                digit_indices = self.digit_indices[digit]
                if len(digit_indices) > samples_per_digit:
                    # Randomly select samples_per_digit indices
                    selected = random.sample(digit_indices, samples_per_digit)
                else:
                    # If not enough samples, take all and oversample if needed
                    selected = digit_indices
                    if samples_per_digit > len(digit_indices):
                        # Oversample with replacement to reach samples_per_digit
                        additional = random.choices(digit_indices, k=samples_per_digit - len(digit_indices))
                        selected.extend(additional)
                selected_indices.extend(selected)
        
        return Subset(self, selected_indices)

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # For timing
    train_times = []
    eval_times = []
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        train_times.append(train_time)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluation phase
        eval_start_time = time.time()
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time
        eval_times.append(eval_time)
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Calculate total epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
              f'Train Time: {train_time:.2f}s, Eval Time: {eval_time:.2f}s, '
              f'Total Epoch Time: {epoch_time:.2f}s')
    
    timing_data = {
        'train_times': train_times,
        'eval_times': eval_times,
        'epoch_times': epoch_times
    }
    
    return train_losses, test_losses, train_accuracies, test_accuracies, timing_data

def plot_performance(train_losses, test_losses, train_accuracies, test_accuracies, timing_data):
    plt.figure(figsize=(16, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Plot timing information
    plt.subplot(2, 2, 3)
    plt.plot(timing_data['train_times'], label='Training Time')
    plt.plot(timing_data['eval_times'], label='Evaluation Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Training and Evaluation Time per Epoch')
    
    # Plot total epoch time
    plt.subplot(2, 2, 4)
    plt.plot(timing_data['epoch_times'], label='Total Epoch Time', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Total Time per Epoch')
    
    plt.tight_layout()
    plt.savefig('training_performance_with_timing.png')
    plt.show()
    
    # Also create and save a separate figure just for timing data
    plt.figure(figsize=(12, 8))
    
    # Stacked bar chart for timing breakdown
    bar_width = 0.8
    epochs = range(1, len(timing_data['train_times']) + 1)
    
    plt.bar(epochs, timing_data['train_times'], bar_width, 
            label='Training Time', color='#3498db')
    plt.bar(epochs, timing_data['eval_times'], bar_width,
            bottom=timing_data['train_times'], label='Evaluation Time', color='#e74c3c')
    
    # Calculate other time (overhead)
    other_times = []
    for i in range(len(timing_data['epoch_times'])):
        other_time = timing_data['epoch_times'][i] - timing_data['train_times'][i] - timing_data['eval_times'][i]
        other_times.append(other_time)
    
    plt.bar(epochs, other_times, bar_width,
            bottom=[sum(x) for x in zip(timing_data['train_times'], timing_data['eval_times'])],
            label='Overhead Time', color='#2ecc71')
    
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Time Breakdown per Epoch')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total time values on top of each bar
    for i, epoch in enumerate(epochs):
        plt.text(epoch, timing_data['epoch_times'][i] + 0.5, 
                 f"{timing_data['epoch_times'][i]:.1f}s", 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('timing_breakdown.png')
    plt.show()

def main(use_attention=False, samples_per_digit=None, seed=42, learning_rate=0.001, epochs=20):
    """
    Main function to train the model
    
    Args:
        use_attention: Whether to use attention mechanisms
        samples_per_digit: Number of samples per digit to use (None for all samples)
        seed: Random seed for reproducibility
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
    """
    # Track overall execution time
    start_time = time.time()
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Track dataset loading time
    data_load_start = time.time()
    
    # Create datasets
    train_dataset = AudioMNISTDataset(
        root_dir='AudioMNIST_spectrograms', 
        split='train', 
        transform=transform
    )
    
    test_dataset = AudioMNISTDataset(
        root_dir='AudioMNIST_spectrograms', 
        split='test', 
        transform=transform
    )
    
    # Create balanced subsets if samples_per_digit is specified
    if samples_per_digit is not None:
        train_dataset = train_dataset.get_balanced_subset(samples_per_digit, seed)
        # For test set, we might want to keep all samples to get a proper evaluation
        # But you can also limit test samples if needed:
        # test_dataset = test_dataset.get_balanced_subset(samples_per_digit, seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    data_load_time = time.time() - data_load_start
    
    # Print dataset information
    print(f"Training with {'attention' if use_attention else 'no attention'}")
    print(f"Samples per digit: {samples_per_digit if samples_per_digit else 'all'}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Dataset loading time: {data_load_time:.2f} seconds")
    
    # Create model
    model_creation_start = time.time()
    model = AudioMNISTCNN(use_attention=use_attention)
    model_creation_time = time.time() - model_creation_start
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    # Train model
    training_start = time.time()
    train_losses, test_losses, train_accuracies, test_accuracies, timing_data = train_model(
        model, train_loader, test_loader, epochs=epochs, learning_rate=learning_rate
    )
    total_training_time = time.time() - training_start
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    # Plot and save performance
    plot_performance(train_losses, test_losses, train_accuracies, test_accuracies, timing_data)
    
    # Save the model
    model_save_start = time.time()
    attention_str = "with_attention" if use_attention else "no_attention"
    samples_str = f"_{samples_per_digit}samples" if samples_per_digit else ""
    lr_str = f"_lr{learning_rate:.6f}".replace(".", "p")
    model_name = f"audiomnist_cnn_{attention_str}{samples_str}_seed{seed}{lr_str}.pth"
    torch.save(model.state_dict(), model_name)
    model_save_time = time.time() - model_save_start
    print(f"Model saved as {model_name}")
    print(f"Model saving time: {model_save_time:.2f} seconds")
    
    # Print overall execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Save timing summary to file
    with open('timing_summary.txt', 'w') as f:
        f.write(f"Timing Summary for AudioMNIST CNN Training\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Configuration: {attention_str}, {samples_per_digit if samples_per_digit else 'all'} samples per digit\n")
        f.write(f"Dataset loading time: {data_load_time:.2f} seconds\n")
        f.write(f"Model creation time: {model_creation_time:.2f} seconds\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        f.write(f"Model saving time: {model_save_time:.2f} seconds\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        f.write(f"Epoch-by-Epoch Timing Breakdown\n")
        f.write(f"-----------------------------\n")
        f.write(f"{'Epoch':<6} {'Train Time (s)':<15} {'Eval Time (s)':<15} {'Total Time (s)':<15}\n")
        for i in range(len(timing_data['train_times'])):
            f.write(f"{i+1:<6} {timing_data['train_times'][i]:<15.2f} {timing_data['eval_times'][i]:<15.2f} {timing_data['epoch_times'][i]:<15.2f}\n")

if __name__ == "__main__":
    # Example usage:
    # Set your parameters here:
    use_attention = True       # Toggle attention mechanism
    samples_per_digit = 50     # Number of samples per digit (None = use all)
    seed = 42                  # Random seed for reproducibility
    learning_rate = 0.0001     # Learning rate for the optimizer
    epochs = 20                # Number of training epochs
    
    main(
        use_attention=use_attention, 
        samples_per_digit=samples_per_digit, 
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs
    )