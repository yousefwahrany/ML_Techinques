import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
from tqdm import tqdm

# Configuration
class Config:
    # Data
    data_path = 'AudioMNIST/data/'
    sr = 16000  # Sample rate - AudioMNIST is 16kHz
    
    # Feature extraction
    n_mels = 64  # Number of mel bands
    frame_length_ms = 15  # Frame length in milliseconds
    hop_length_ms = 10  # Hop length in milliseconds
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    latent_dim = 64  # For autoencoder methods
    max_frames = 80  # Maximum number of frames per utterance
    
    # Splits
    test_size = 0.2
    random_state = 42
    
    # Model paths for saving
    model_paths = {
        'autoencoder_a': 'models/autoencoder_a.pt',
        'autoencoder_b': 'models/autoencoder_b.pt',
        'autoencoder_c': 'models/autoencoder_c.pt',
    }

config = Config()

# Convert ms to samples
config.frame_length = int(config.sr * config.frame_length_ms / 1000)
config.hop_length = int(config.sr * config.hop_length_ms / 1000)

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(list(config.model_paths.values())[0]), exist_ok=True)

# Data preprocessing functions
def load_audiomnist(data_path):
    """
    Load AudioMNIST dataset and return features and labels.
    """
    print("Loading AudioMNIST dataset...")
    features = []
    labels = []
    speaker_ids = []
    
    # Find all speaker directories
    speaker_dirs = sorted(glob.glob(os.path.join(data_path, "*/")))
    
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = os.path.basename(os.path.normpath(speaker_dir))
        
        # Find all WAV files for this speaker
        wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        for wav_file in wav_files:
            # Extract digit from filename (first character)
            filename = os.path.basename(wav_file)
            digit = int(filename.split('_')[0])
            
            # Load audio and compute log-mel spectrogram
            try:
                y, _ = librosa.load(wav_file, sr=config.sr)
                S = librosa.feature.melspectrogram(
                    y=y, 
                    sr=config.sr,
                    n_mels=config.n_mels,
                    n_fft=config.frame_length,
                    hop_length=config.hop_length
                )
                # Convert to log scale
                log_S = librosa.power_to_db(S, ref=np.max)
                
                # Transpose to have frames as first dimension
                log_S = log_S.T
                
                features.append(log_S)
                labels.append(digit)
                speaker_ids.append(int(speaker_id))
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
    
    return features, np.array(labels), np.array(speaker_ids)

def normalize_features(features, speaker_ids=None, per_speaker=False):
    """
    Normalize features either globally or per speaker.
    
    Args:
        features: List of feature matrices
        speaker_ids: Array of speaker IDs
        per_speaker: Whether to normalize per speaker
        
    Returns:
        List of normalized feature matrices
    """
    if per_speaker:
        # Group features by speaker ID
        unique_speakers = np.unique(speaker_ids)
        normalized_features = []
        
        for speaker in unique_speakers:
            speaker_indices = np.where(speaker_ids == speaker)[0]
            speaker_features = [features[i] for i in speaker_indices]
            
            # Concatenate all frames for this speaker
            all_frames = np.vstack([f for f in speaker_features])
            
            # Fit scaler on all frames for this speaker
            scaler = StandardScaler()
            scaler.fit(all_frames)
            
            # Apply normalization to each utterance
            for idx in speaker_indices:
                normalized_features.append(scaler.transform(features[idx]))
                
        return normalized_features
    else:
        # Global normalization
        # Concatenate all frames from all utterances
        all_frames = np.vstack([f for f in features])
        
        # Fit scaler on all frames
        scaler = StandardScaler()
        scaler.fit(all_frames)
        
        # Apply normalization to each utterance
        normalized_features = [scaler.transform(f) for f in features]
        
        return normalized_features

# PyTorch Dataset classes
class AudioMNISTDataset(Dataset):
    """Base dataset for AudioMNIST"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AverageVectorDataset(AudioMNISTDataset):
    """Dataset for Method 1: Average Vector"""
    def __getitem__(self, idx):
        # Average all frames to get a single vector
        avg_vector = np.mean(self.features[idx], axis=0)
        return avg_vector, self.labels[idx]

class FlattenedUtteranceDataset(AudioMNISTDataset):
    """Dataset for Method 2: Autoencoder Method A (flatten utterance)"""
    def __init__(self, features, labels, max_frames):
        super().__init__(features, labels)
        self.max_frames = max_frames
        self.frame_dim = features[0].shape[1]
        
    def __getitem__(self, idx):
        # Pad or truncate to max_frames
        feature = self.features[idx]
        padded_feature = np.zeros((self.max_frames, self.frame_dim))
        
        n_frames = min(feature.shape[0], self.max_frames)
        padded_feature[:n_frames] = feature[:n_frames]
        
        # Flatten the 2D array into a 1D vector
        flattened = padded_feature.flatten()
        
        return flattened, self.labels[idx]

# Autoencoder Models
class AutoencoderA(nn.Module):
    """Autoencoder for Method 2: Direct compression of flattened utterance"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Calculate intermediate dimensions
        h1_dim = input_dim // 4
        h2_dim = input_dim // 16
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderBC(nn.Module):
    """Autoencoder for Methods 3 and 4: Sequential pairwise compression"""
    def __init__(self, frame_dim):
        super().__init__()
        
        # Encoder: compresses two frames into one frame-sized vector
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim * 2, frame_dim * 4),
            nn.ReLU(),
            nn.Linear(frame_dim * 4, frame_dim * 2),
            nn.ReLU(),
            nn.Linear(frame_dim * 2, frame_dim)
        )
        
        # Decoder: decompresses one frame-sized vector into two frames
        self.decoder = nn.Sequential(
            nn.Linear(frame_dim, frame_dim * 2),
            nn.ReLU(),
            nn.Linear(frame_dim * 2, frame_dim * 4),
            nn.ReLU(), 
            nn.Linear(frame_dim * 4, frame_dim * 2)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

# Method implementations
def method1_baseline(train_features, train_labels, test_features, test_labels):
    """
    Method 1: Baseline - Average Vector
    """
    print("\nRunning Method 1: Baseline (Average Vector)")
    
    # Compute average vector for each utterance
    train_avg_vectors = np.array([np.mean(feat, axis=0) for feat in train_features])
    test_avg_vectors = np.array([np.mean(feat, axis=0) for feat in test_features])
    
    # Train a logistic regression classifier
    classifier = LogisticRegression(max_iter=500, random_state=config.random_state)
    classifier.fit(train_avg_vectors, train_labels)
    
    # Evaluate
    train_preds = classifier.predict(train_avg_vectors)
    test_preds = classifier.predict(test_avg_vectors)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"Method 1 - Training accuracy: {train_acc:.4f}")
    print(f"Method 1 - Test accuracy: {test_acc:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_vectors': train_avg_vectors,
        'test_vectors': test_avg_vectors,
        'train_labels': train_labels,
        'test_labels': test_labels
    }

def method2_autoencoder_a(train_features, train_labels, test_features, test_labels):
    """
    Method 2: Autoencoder Method A - Flatten utterance and compress
    """
    print("\nRunning Method 2: Autoencoder Method A (Flattened Utterance)")
    
    # Prepare datasets with flattened and padded utterances
    max_frames = config.max_frames
    frame_dim = train_features[0].shape[1]
    
    # Create PyTorch datasets
    train_dataset = FlattenedUtteranceDataset(train_features, train_labels, max_frames)
    input_dim = max_frames * frame_dim
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Initialize autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderA(input_dim, config.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    train_losses = []
    
    print("Training Autoencoder A...")
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.float().to(device)
            
            # Forward pass
            encoded, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save the model
    torch.save(model.state_dict(), config.model_paths['autoencoder_a'])
    
    # Extract encoded features for classification
    model.eval()
    
    # Process training data
    train_encoded = []
    for i in tqdm(range(len(train_features)), desc="Encoding training data"):
        feature = train_features[i]
        padded_feature = np.zeros((max_frames, frame_dim))
        n_frames = min(feature.shape[0], max_frames)
        padded_feature[:n_frames] = feature[:n_frames]
        flattened = padded_feature.flatten()
        
        with torch.no_grad():
            encoded, _ = model(torch.tensor(flattened).float().to(device))
        
        train_encoded.append(encoded.cpu().numpy())
    
    # Process test data
    test_encoded = []
    for i in tqdm(range(len(test_features)), desc="Encoding test data"):
        feature = test_features[i]
        padded_feature = np.zeros((max_frames, frame_dim))
        n_frames = min(feature.shape[0], max_frames)
        padded_feature[:n_frames] = feature[:n_frames]
        flattened = padded_feature.flatten()
        
        with torch.no_grad():
            encoded, _ = model(torch.tensor(flattened).float().to(device))
        
        test_encoded.append(encoded.cpu().numpy())
    
    train_encoded = np.array(train_encoded)
    test_encoded = np.array(test_encoded)
    
    # Train a classifier
    classifier = LogisticRegression(max_iter=500, random_state=config.random_state)
    classifier.fit(train_encoded, train_labels)
    
    # Evaluate
    train_preds = classifier.predict(train_encoded)
    test_preds = classifier.predict(test_encoded)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"Method 2 - Training accuracy: {train_acc:.4f}")
    print(f"Method 2 - Test accuracy: {test_acc:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_vectors': train_encoded,
        'test_vectors': test_encoded,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_losses': train_losses
    }

def method3_autoencoder_b(train_features, train_labels, test_features, test_labels):
    """
    Method 3: Autoencoder Method B - Sequential pairwise compression
    """
    print("\nRunning Method 3: Autoencoder Method B (Sequential Compression)")
    
    frame_dim = train_features[0].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset for training the pairwise autoencoder
    # We'll create pairs of consecutive frames from all utterances
    train_pairs = []
    
    for feature in train_features:
        for i in range(len(feature) - 1):
            pair = np.concatenate([feature[i], feature[i+1]])
            train_pairs.append(pair)
    
    train_pairs = np.array(train_pairs)
    
    # Convert to PyTorch dataset
    class PairDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
            
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            return self.pairs[idx]
    
    pair_dataset = PairDataset(train_pairs)
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Initialize autoencoder
    model = AutoencoderBC(frame_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    train_losses = []
    
    print("Training Autoencoder B...")
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for inputs in pair_loader:
            inputs = inputs.float().to(device)
            
            # Forward pass
            encoded, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(pair_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save the model
    torch.save(model.state_dict(), config.model_paths['autoencoder_b'])
    
    # Apply sequential compression to each utterance
    model.eval()
    
    def sequential_compress(feature):
        """Sequentially compress frames using the autoencoder"""
        if len(feature) == 1:
            # If there's only one frame, duplicate it to form a pair
            pair = np.concatenate([feature[0], feature[0]])
            with torch.no_grad():
                encoded = model.encode(torch.tensor(pair).float().to(device))
            return encoded.cpu().numpy()
        
        frames = feature.copy()
        
        while len(frames) > 1:
            # Compress first two frames
            pair = np.concatenate([frames[0], frames[1]])
            with torch.no_grad():
                encoded = model.encode(torch.tensor(pair).float().to(device))
            
            # Replace the pair with the compressed vector
            compressed = encoded.cpu().numpy()
            frames = np.vstack([compressed, frames[2:]]) if len(frames) > 2 else np.array([compressed])
        
        return frames[0]
    
    # Process training data
    train_encoded = []
    for i in tqdm(range(len(train_features)), desc="Encoding training data"):
        compressed = sequential_compress(train_features[i])
        train_encoded.append(compressed)
    
    # Process test data
    test_encoded = []
    for i in tqdm(range(len(test_features)), desc="Encoding test data"):
        compressed = sequential_compress(test_features[i])
        test_encoded.append(compressed)
    
    train_encoded = np.array(train_encoded)
    test_encoded = np.array(test_encoded)
    
    # Train a classifier
    classifier = LogisticRegression(max_iter=500, random_state=config.random_state)
    classifier.fit(train_encoded, train_labels)
    
    # Evaluate
    train_preds = classifier.predict(train_encoded)
    test_preds = classifier.predict(test_encoded)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"Method 3 - Training accuracy: {train_acc:.4f}")
    print(f"Method 3 - Test accuracy: {test_acc:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_vectors': train_encoded,
        'test_vectors': test_encoded,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_losses': train_losses
    }

def method4_autoencoder_c(train_features, train_labels, test_features, test_labels):
    """
    Method 4: Autoencoder Method C - Greedy compression based on reconstruction error
    """
    print("\nRunning Method 4: Autoencoder Method C (Greedy Compression)")
    
    frame_dim = train_features[0].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pairwise autoencoder trained in Method 3
    model = AutoencoderBC(frame_dim).to(device)
    model.load_state_dict(torch.load(config.model_paths['autoencoder_b']))
    model.eval()
    
    # Calculate reconstruction error for a pair of frames
    def reconstruction_error(pair):
        """Calculate reconstruction error for a pair of frames"""
        with torch.no_grad():
            pair_tensor = torch.tensor(pair).float().to(device)
            encoded = model.encode(pair_tensor)
            decoded = model.decode(encoded)
            error = ((decoded - pair_tensor) ** 2).mean().item()
        return error
    
    # Apply greedy compression to each utterance
    def greedy_compress(feature):
        """Greedily compress frames based on reconstruction error"""
        if len(feature) == 1:
            # If there's only one frame, duplicate it to form a pair
            pair = np.concatenate([feature[0], feature[0]])
            with torch.no_grad():
                encoded = model.encode(torch.tensor(pair).float().to(device))
            return encoded.cpu().numpy()
        
        frames = feature.copy()
        
        while len(frames) > 1:
            # Calculate reconstruction error for each adjacent pair
            errors = []
            for i in range(len(frames) - 1):
                pair = np.concatenate([frames[i], frames[i+1]])
                errors.append(reconstruction_error(pair))
            
            # Find pair with lowest error
            min_error_idx = np.argmin(errors)
            
            # Compress the selected pair
            pair = np.concatenate([frames[min_error_idx], frames[min_error_idx+1]])
            with torch.no_grad():
                encoded = model.encode(torch.tensor(pair).float().to(device))
            
            # Replace the pair with the compressed vector
            compressed = encoded.cpu().numpy()
            frames = np.delete(frames, [min_error_idx, min_error_idx+1], axis=0)
            frames = np.insert(frames, min_error_idx, compressed, axis=0)
        
        return frames[0]
    
    # Process training data
    train_encoded = []
    for i in tqdm(range(len(train_features)), desc="Encoding training data"):
        compressed = greedy_compress(train_features[i])
        train_encoded.append(compressed)
    
    # Process test data
    test_encoded = []
    for i in tqdm(range(len(test_features)), desc="Encoding test data"):
        compressed = greedy_compress(test_features[i])
        test_encoded.append(compressed)
    
    train_encoded = np.array(train_encoded)
    test_encoded = np.array(test_encoded)
    
    # Train a classifier
    classifier = LogisticRegression(max_iter=500, random_state=config.random_state)
    classifier.fit(train_encoded, train_labels)
    
    # Evaluate
    train_preds = classifier.predict(train_encoded)
    test_preds = classifier.predict(test_encoded)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"Method 4 - Training accuracy: {train_acc:.4f}")
    print(f"Method 4 - Test accuracy: {test_acc:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_vectors': train_encoded,
        'test_vectors': test_encoded,
        'train_labels': train_labels,
        'test_labels': test_labels
    }

# Visualization functions
def visualize_latent_spaces(results):
    """Visualize latent spaces using t-SNE and PCA"""
    plt.figure(figsize=(20, 15))
    
    methods = ['Method 1: Baseline', 'Method 2: Autoencoder A', 
               'Method 3: Autoencoder B', 'Method 4: Autoencoder C']
    
    # Create a colormap for digits 0-9
    cmap = plt.cm.get_cmap('tab10', 10)
    
    for i, (method_name, result) in enumerate(zip(methods, results)):
        # t-SNE visualization
        plt.subplot(2, 4, i+1)
        
        # Use a smaller subset for t-SNE to speed up visualization
        max_points = 500
        indices = np.random.choice(len(result['train_vectors']), 
                                  size=min(max_points, len(result['train_vectors'])), 
                                  replace=False)
        
        vectors = result['train_vectors'][indices]
        labels = result['train_labels'][indices]
        
        tsne = TSNE(n_components=2, random_state=config.random_state)
        embedded = tsne.fit_transform(vectors)
        
        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            plt.scatter(embedded[digit_indices, 0], embedded[digit_indices, 1], 
                       c=[cmap(digit)], label=str(digit), alpha=0.7, s=50)
        
        if i == 0:  # Only add legend to the first plot
            plt.legend(loc='best', fontsize=10)
        
        plt.title(f"{method_name} (t-SNE)", fontsize=12)
        plt.tight_layout()
        
        # PCA visualization
        plt.subplot(2, 4, i+5)
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        
        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            plt.scatter(reduced[digit_indices, 0], reduced[digit_indices, 1], 
                       c=[cmap(digit)], label=str(digit), alpha=0.7, s=50)
        
        explained_var = pca.explained_variance_ratio_.sum() * 100
        plt.title(f"{method_name} (PCA, {explained_var:.1f}% var)", fontsize=12)
        plt.tight_layout()
    
    plt.savefig('latent_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Latent space visualization saved as 'latent_space_visualization.png'")

def plot_training_curves(results):
    """Plot training loss curves for autoencoder methods"""
    plt.figure(figsize=(12, 6))
    
    for i, method_name in enumerate(['Method 2: Autoencoder A', 'Method 3: Autoencoder B']):
        if 'train_losses' in results[i+1]:  # Skip Method 1 (index 0)
            plt.plot(results[i+1]['train_losses'], label=method_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Curves for Autoencoder Methods')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    print("Training curves saved as 'training_curves.png'")

def compare_methods(results):
    """Compare classification accuracy across methods"""
    methods = ['Method 1: Baseline', 'Method 2: Autoencoder A', 
               'Method 3: Autoencoder B', 'Method 4: Autoencoder C']
    
    train_accs = [result['train_accuracy'] for result in results]
    test_accs = [result['test_accuracy'] for result in results]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Training Accuracy')
    plt.bar(x + width/2, test_accs, width, label='Test Accuracy')
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy Comparison')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, v in enumerate(train_accs):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(test_accs):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.savefig('accuracy_comparison.png', dpi=300)
    plt.close()
    print("Accuracy comparison saved as 'accuracy_comparison.png'")

# Main pipeline function
def main():
    # 1. Load data
    features, labels, speaker_ids = load_audiomnist(config.data_path)
    
    # 2. Normalize features globally
    normalized_features = normalize_features(features, speaker_ids, per_speaker=False)
    
    # 3. Split data
    # Option 1: Random split (non-speaker-independent)
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels
    )
    
    # Option 2: Speaker-independent split (uncomment to use)
    # unique_speakers = np.unique(speaker_ids)
    # train_speakers, test_speakers = train_test_split(
    #     unique_speakers, 
    #     test_size=config.test_size,
    #     random_state=config.random_state
    # )
    # train_indices = np.where(np.isin(speaker
    # train_indices = np.where(np.isin(speaker_ids, train_speakers))[0]
    # test_indices = np.where(np.isin(speaker_ids, test_speakers))[0]
    
    train_features = [normalized_features[i] for i in train_indices]
    test_features = [normalized_features[i] for i in test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    print(f"Data split: {len(train_features)} training samples, {len(test_features)} test samples")
    
    # 4. Run methods
    results = []
    
    # Method 1: Baseline (Average Vector)
    result1 = method1_baseline(train_features, train_labels, test_features, test_labels)
    results.append(result1)
    
    # Method 2: Autoencoder Method A (Flatten utterance)
    result2 = method2_autoencoder_a(train_features, train_labels, test_features, test_labels)
    results.append(result2)
    
    # Method 3: Autoencoder Method B (Sequential pairwise compression)
    result3 = method3_autoencoder_b(train_features, train_labels, test_features, test_labels)
    results.append(result3)
    
    # Method 4: Autoencoder Method C (Greedy compression)
    result4 = method4_autoencoder_c(train_features, train_labels, test_features, test_labels)
    results.append(result4)
    
    # 5. Visualizations
    visualize_latent_spaces(results)
    plot_training_curves(results)
    compare_methods(results)
    
    print("\nAll methods completed. See generated visualizations for comparison.")
    
    return results

if __name__ == "__main__":
    main()