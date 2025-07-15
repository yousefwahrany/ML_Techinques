import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from datetime import timedelta

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AudioMNISTNet(nn.Module):
    def __init__(self, input_channels):
        super(AudioMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        
        # Dynamically calculate the flattened size after convolutions
        def conv2d_output_size(input_size, kernel_size=5, stride=1, padding=0):
            return (input_size - kernel_size + 2*padding) // stride + 1
        
        # Assuming 32x32 input size
        input_size = 32
        after_conv1 = conv2d_output_size(input_size)
        after_pool1 = after_conv1 // 2
        after_conv2 = conv2d_output_size(after_pool1)
        after_pool2 = after_conv2 // 2
        
        self.fc1 = nn.Linear(16 * after_pool2 * after_pool2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImageAugmentation:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def squeeze(self, image, factor=0.95):
        """Squeeze image horizontally."""
        width, height = image.size
        new_width = int(width * factor)
        resized = image.resize((new_width, height), Image.LANCZOS)
        
        # Create a new blank image and paste the resized image
        new_image = Image.new(image.mode, (width, height), color='black')
        paste_x = (width - new_width) // 2
        new_image.paste(resized, (paste_x, 0))
        
        return new_image
    
    def expand(self, image, factor=1.05):
        """Expand image horizontally."""
        width, height = image.size
        new_width = int(width * factor)
        resized = image.resize((new_width, height), Image.LANCZOS)
        
        # Create a new blank image and paste the resized image
        new_image = Image.new(image.mode, (width, height), color='black')
        paste_x = (new_width - width) // 2
        new_image.paste(resized, (-paste_x, 0))
        
        return new_image
    
    def add_noise(self, image):
        """Add Gaussian noise to the image."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1)
        
        # Convert back to PIL Image
        return Image.fromarray((noisy_img * 255).astype(np.uint8))

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_images=False, augment_audio=False, samples_per_digit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augment_images = augment_images
        self.augment_audio = augment_audio
        self.data = []
        self.labels = []
        self.augmenter = ImageAugmentation()
        
        # Path for audio augmented spectrograms
        self.augmented_root_dir = root_dir.replace('AudioMNIST_spectrograms', 'AudioMNIST_augmented_spectrograms')
        
        # Iterate through digit folders
        for digit in range(10):
            digit_path = os.path.join(root_dir, str(digit))
            if not os.path.exists(digit_path):
                continue
                
            digit_images = [
                os.path.join(digit_path, img_name) 
                for img_name in os.listdir(digit_path) 
                if img_name.endswith('_spectrogram.png')
            ]
            
            # Limit samples if specified
            if samples_per_digit is not None:
                digit_images = digit_images[:samples_per_digit]
            
            for img_path in digit_images:
                # Original image
                self.data.append(img_path)
                self.labels.append(digit)
                
                # Image-based augmentation if enabled
                if augment_images:
                    # Load image
                    img = Image.open(img_path)
                    
                    # Squeeze image
                    squeezed_img = self.augmenter.squeeze(img)
                    self.data.append(squeezed_img)
                    self.labels.append(digit)
                    
                    # Expand image
                    expanded_img = self.augmenter.expand(img)
                    self.data.append(expanded_img)
                    self.labels.append(digit)
                    
                    # Noisy image
                    noisy_img = self.augmenter.add_noise(img)
                    self.data.append(noisy_img)
                    self.labels.append(digit)
                
                # Audio-based augmentation if enabled
                if augment_audio:
                    # Get base filename without spectrogram suffix
                    base_filename = os.path.basename(img_path).replace('_spectrogram.png', '')
                    augmented_digit_path = os.path.join(self.augmented_root_dir, str(digit))
                    
                    # Only proceed if the augmented directory exists
                    if os.path.exists(augmented_digit_path):
                        # Look for the three audio augmentation types
                        aug_types = ["speed_up", "slow_down", "add_noise"]
                        
                        for aug_type in aug_types:
                            aug_file = f"{base_filename}_{aug_type}_spectrogram.png"
                            aug_path = os.path.join(augmented_digit_path, aug_file)
                            
                            # Check if augmented file exists
                            if os.path.exists(aug_path):
                                self.data.append(aug_path)
                                self.labels.append(digit)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Check if item is a path or already a PIL Image
        if isinstance(self.data[idx], str):
            image = Image.open(self.data[idx])
        else:
            image = self.data[idx]
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def visualize_augmentations(original_dataset, augmented_image_dataset=None, augmented_audio_dataset=None, num_samples=3):
    """
    Visualize original and augmented images (both image-based and audio-based augmentations).
    
    Args:
    original_dataset: Dataset containing original images
    augmented_image_dataset: Dataset containing image-based augmentations
    augmented_audio_dataset: Dataset containing audio-based augmentations
    num_samples: Number of samples to visualize
    """
    # Find some original images
    original_images = []
    original_labels = []
    
    for i in range(len(original_dataset)):
        if len(original_images) >= num_samples:
            break
        
        if isinstance(original_dataset.data[i], str):
            img_path = original_dataset.data[i]
            label = original_dataset.labels[i]
            img = Image.open(img_path)
            
            # Extract base filename for matching with augmented versions
            base_filename = os.path.basename(img_path).split('_spectrogram')[0]
            
            original_images.append((img, base_filename))
            original_labels.append(label)
    
    # Set up the visualization grid
    num_rows = len(original_images)
    
    # Count columns: 1 for original + 3 for image augs (if present) + 3 for audio augs (if present)
    num_cols = 1
    if augmented_image_dataset is not None:
        num_cols += 3
    if augmented_audio_dataset is not None:
        num_cols += 3
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    # If only one row, wrap axes in list for consistent indexing
    if num_rows == 1:
        axes = [axes]
    
    for i, ((img, base_filename), label) in enumerate(zip(original_images, original_labels)):
        # Plot original image
        col_idx = 0
        axes[i][col_idx].imshow(img, cmap='viridis')
        axes[i][col_idx].set_title(f'Digit {label} - Original')
        axes[i][col_idx].axis('off')
        col_idx += 1
        
        # Find and plot image-based augmentations if available
        if augmented_image_dataset is not None:
            aug_types = ['Squeezed', 'Expanded', 'Noisy']
            
            # Find the in-memory augmentations (not file paths)
            aug_counter = 0
            for j in range(len(augmented_image_dataset)):
                if not isinstance(augmented_image_dataset.data[j], str) and augmented_image_dataset.labels[j] == label:
                    if aug_counter < 3:  # We have 3 types of image augmentations
                        axes[i][col_idx].imshow(augmented_image_dataset.data[j], cmap='viridis')
                        axes[i][col_idx].set_title(f'Digit {label} - {aug_types[aug_counter]}')
                        axes[i][col_idx].axis('off')
                        col_idx += 1
                        aug_counter += 1
                    
                    if aug_counter >= 3:
                        break
        
        # Find and plot audio-based augmentations if available
        if augmented_audio_dataset is not None:
            aug_types = ['Speed Up', 'Slow Down', 'Add Noise']
            aug_counter = 0
            
            for j in range(len(augmented_audio_dataset)):
                if isinstance(augmented_audio_dataset.data[j], str):
                    aug_path = augmented_audio_dataset.data[j]
                    aug_filename = os.path.basename(aug_path)
                    
                    # Check if this augmentation belongs to the current original image
                    if base_filename in aug_filename and augmented_audio_dataset.labels[j] == label:
                        # Determine which augmentation type this is
                        aug_type_idx = None
                        if "speed_up" in aug_filename:
                            aug_type_idx = 0
                        elif "slow_down" in aug_filename:
                            aug_type_idx = 1
                        elif "add_noise" in aug_filename:
                            aug_type_idx = 2
                        
                        if aug_type_idx is not None:
                            img = Image.open(aug_path)
                            axes[i][col_idx].imshow(img, cmap='viridis')
                            axes[i][col_idx].set_title(f'Digit {label} - {aug_types[aug_type_idx]}')
                            axes[i][col_idx].axis('off')
                            col_idx += 1
                            aug_counter += 1
                        
                        if aug_counter >= 3:
                            break
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    epoch_times = []  # Track time for each epoch
    
    total_start_time = time.time()  # Start timing for the entire training process
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start timing for this epoch
        
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Record metrics
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Calculate and record epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Format time as minutes:seconds
        epoch_time_formatted = str(timedelta(seconds=int(epoch_time)))
        
        print(f'Epoch {epoch+1}/{num_epochs} - Time: {epoch_time_formatted}')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # Calculate total training time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    total_time_formatted = str(timedelta(seconds=int(total_time)))
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_epoch_time_formatted = str(timedelta(seconds=int(avg_epoch_time)))
    
    print(f'\nTotal training time: {total_time_formatted}')
    print(f'Average time per epoch: {avg_epoch_time_formatted}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies, epoch_times, total_time

def main(seed=42, samples_per_digit=1000, augment_images=False, augment_audio=False):
    start_time = time.time()  # Start timing the entire program execution
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Print augmentation settings
    print(f"Running with settings:")
    print(f"- Image-based augmentation: {'Enabled' if augment_images else 'Disabled'}")
    print(f"- Audio-based augmentation: {'Enabled' if augment_audio else 'Disabled'}")
    print(f"- Samples per digit: {samples_per_digit}")

    # First, determine the number of channels in the images
    sample_img_path = os.path.join('AudioMNIST_spectrograms/train/0', 
                                   os.listdir('AudioMNIST_spectrograms/train/0')[0])
    sample_img = Image.open(sample_img_path)
    
    # Convert RGBA to RGB if needed
    if sample_img.mode == 'RGBA':
        sample_img = sample_img.convert('RGB')
    
    # Debug prints
    print(f"Sample image shape: {np.array(sample_img).shape}")
    print(f"Image mode: {sample_img.mode}")
    
    input_channels = len(sample_img.getbands())
    print(f"Detected {input_channels} channel(s) in spectrograms")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match network input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*input_channels, std=[0.5]*input_channels)
    ])
    
    load_start_time = time.time()
    print("Loading datasets...")
    
    # Create separate datasets based on augmentation settings for visualization
    base_dataset = AudioMNISTDataset(
        'AudioMNIST_spectrograms/train', 
        transform=transform,
        augment_images=False,
        augment_audio=False,
        samples_per_digit=samples_per_digit
    )
    
    # Create the actual training dataset with all enabled augmentations
    train_dataset = AudioMNISTDataset(
        'AudioMNIST_spectrograms/train', 
        transform=transform,
        augment_images=augment_images,
        augment_audio=augment_audio,
        samples_per_digit=samples_per_digit
    )
    
    # For visualization purposes only - separate datasets with individual augmentation types
    img_aug_dataset = None
    audio_aug_dataset = None
    
    if augment_images:
        img_aug_dataset = AudioMNISTDataset(
            'AudioMNIST_spectrograms/train', 
            transform=transform,
            augment_images=True,
            augment_audio=False,
            samples_per_digit=5  # Just need a few for visualization
        )
    
    if augment_audio:
        audio_aug_dataset = AudioMNISTDataset(
            'AudioMNIST_spectrograms/train', 
            transform=transform,
            augment_images=False,
            augment_audio=True,
            samples_per_digit=5  # Just need a few for visualization
        )
    
    load_time = time.time() - load_start_time
    print(f"Datasets loaded in {timedelta(seconds=int(load_time))}")
    
    # Visualize augmentations if any are enabled
    if augment_images or augment_audio:
        print("Visualizing augmentations...")
        viz_start_time = time.time()
        visualize_augmentations(base_dataset, img_aug_dataset, audio_aug_dataset)
        viz_time = time.time() - viz_start_time
        print(f"Visualization completed in {timedelta(seconds=int(viz_time))}")
    
    # Print dataset sizes
    print(f"Training dataset contains {len(train_dataset)} samples")
    
    # Create validation dataset (no augmentation for validation)
    val_dataset = AudioMNISTDataset(
        'AudioMNIST_spectrograms/test', 
        transform=transform,
        augment_images=False,
        augment_audio=False
    )
    print(f"Validation dataset contains {len(val_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model, Loss, and Optimizer
    model = AudioMNISTNet(input_channels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nStarting model training...")
    train_losses, val_losses, train_accuracies, val_accuracies, epoch_times, total_training_time = train_model(
        train_loader, val_loader, model, criterion, optimizer, num_epochs=100
    )
    
    # Save the model
    model_filename = f"audio_mnist_model_img{int(augment_images)}_audio{int(augment_audio)}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,
        'samples_per_digit': samples_per_digit,
        'augment_images': augment_images,
        'augment_audio': augment_audio,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'total_training_time': total_training_time
    }, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Plot training and validation metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epoch_times)
    plt.title('Epoch Execution Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(2, 2, 4)
    # Calculate cumulative time
    cumulative_times = np.cumsum(epoch_times)
    plt.plot(cumulative_times)
    plt.title('Cumulative Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(f"training_curves_img{int(augment_images)}_audio{int(augment_audio)}.png")
    plt.show()
    
    # Calculate and print total program execution time
    end_time = time.time()
    total_exec_time = end_time - start_time
    print(f"\nTotal program execution time: {timedelta(seconds=int(total_exec_time))}")
    
    # Print timing summary
    print("\nTiming Summary:")
    print(f"- Data loading time: {timedelta(seconds=int(load_time))}")
    if augment_images or augment_audio:
        print(f"- Visualization time: {timedelta(seconds=int(viz_time))}")
    print(f"- Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"- Average time per epoch: {timedelta(seconds=int(sum(epoch_times)/len(epoch_times)))}")
    print(f"- Total execution time: {timedelta(seconds=int(total_exec_time))}")

if __name__ == '__main__':
    # You can now specify samples per digit and whether to use augmentation types
    main(seed=42, samples_per_digit=100, augment_images=True, augment_audio=True)