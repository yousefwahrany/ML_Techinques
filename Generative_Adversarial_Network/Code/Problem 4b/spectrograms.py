import os
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import soundfile as sf
from scipy import signal

def augment_audio(y, sr, augmentation_type):
    """
    Apply audio augmentation
    
    Parameters:
    - y: Audio signal
    - sr: Sample rate
    - augmentation_type: String indicating the type of augmentation
    
    Returns:
    - Augmented audio signal
    """
    if augmentation_type == "speed_up":
        # Speed up by 5%
        return librosa.effects.time_stretch(y, rate=1.05)
    
    elif augmentation_type == "slow_down":
        # Slow down by 5%
        return librosa.effects.time_stretch(y, rate=0.95)
    
    elif augmentation_type == "add_noise":
        # Generate speech-like noise (colored noise with emphasis in speech frequencies)
        noise = np.random.normal(0, 0.005, len(y))
        # Apply a filter to make it more speech-like
        b, a = signal.butter(4, [300/sr*2, 3000/sr*2], btype='band')
        speech_noise = signal.lfilter(b, a, noise)
        # Mix with original audio (low noise level to preserve intelligibility)
        return y + speech_noise
    
    else:
        return y

def generate_spectrogram(audio_data, sr=None, audio_path=None):
    """
    Generate spectrogram from audio data or file
    
    Parameters:
    - audio_data: Audio signal (if provided)
    - sr: Sample rate (if audio_data is provided)
    - audio_path: Path to audio file (if audio_data is None)
    
    Returns:
    - Normalized spectrogram image
    """
    try:
        # Load audio file if path is provided
        if audio_data is None and audio_path is not None:
            y, sr = librosa.load(audio_path, sr=None)
        else:
            y = audio_data
        
        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,  # Number of Mel bands
            fmax=8000    # Maximum frequency
        )
        
        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Normalize to 0-255 range
        spectrogram_normalized = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())
        spectrogram_normalized = (spectrogram_normalized * 255).astype(np.uint8)
        
        # Resize to 28x28
        from skimage.transform import resize
        spectrogram_resized = resize(spectrogram_normalized, (28, 28), anti_aliasing=True)
        
        return spectrogram_resized
    
    except Exception as e:
        if audio_path:
            print(f"Error processing {audio_path}: {e}")
        else:
            print(f"Error processing audio data: {e}")
        return None

def split_and_generate_spectrograms(input_base_dir, output_base_dir, augmented_output_dir):
    """
    Split AudioMNIST dataset into training and testing spectrogram sets
    and generate augmented versions
    
    Parameters:
    - input_base_dir: Base directory containing speaker folders with audio files
    - output_base_dir: Base directory to save train and test spectrogram splits
    - augmented_output_dir: Directory to save augmented spectrograms
    """
    # Create output directories
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create augmented output directory
    aug_train_dir = os.path.join(augmented_output_dir, 'train')
    aug_test_dir = os.path.join(augmented_output_dir, 'test')
    os.makedirs(aug_train_dir, exist_ok=True)
    os.makedirs(aug_test_dir, exist_ok=True)

    # Collect all audio file paths and their labels
    audio_files = []
    labels = []
    speakers = []

    # Walk through all speaker folders
    for speaker_folder in tqdm(sorted(os.listdir(input_base_dir)), desc="Collecting Audio Files"):
        speaker_path = os.path.join(input_base_dir, speaker_folder)
        
        # Skip if not a directory
        if not os.path.isdir(speaker_path):
            continue
        
        # Process audio files in this speaker folder
        for audio_file in sorted(os.listdir(speaker_path)):
            # Check if it's a wav file
            if not audio_file.endswith('.wav'):
                continue
            
            # Parse filename
            try:
                digit, speaker, sample = audio_file.split('.')[0].split('_')
            except ValueError:
                print(f"Skipping unexpected filename format: {audio_file}")
                continue
            
            # Full path to the audio file
            audio_path = os.path.join(speaker_path, audio_file)
            
            # Store file details
            audio_files.append(audio_path)
            labels.append(int(digit))
            speakers.append(speaker)

    # Convert to numpy arrays for stratified split
    audio_files = np.array(audio_files)
    labels = np.array(labels)
    speakers = np.array(speakers)

    # Stratified split maintaining digit and speaker distribution
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, 
        labels, 
        test_size=0.2,  # 20% for testing
        stratify=labels,  # Ensure even digit distribution
        random_state=42  # Reproducibility
    )

    # Function to save spectrograms
    def save_spectrograms(files, labels, output_dir):
        for file, label in tqdm(zip(files, labels), total=len(files), desc=f"Generating Spectrograms for {output_dir}"):
            # Generate spectrogram
            spectrogram = generate_spectrogram(None, audio_path=file)
            
            if spectrogram is not None:
                # Create digit subdirectory
                digit_dir = os.path.join(output_dir, str(label))
                os.makedirs(digit_dir, exist_ok=True)
                
                # Generate output filename
                output_filename = os.path.splitext(os.path.basename(file))[0] + '_spectrogram.png'
                output_path = os.path.join(digit_dir, output_filename)
                
                # Save spectrogram
                plt.imsave(output_path, spectrogram, cmap='viridis')

    # Function to save augmented spectrograms
    def save_augmented_spectrograms(files, labels, output_dir):
        augmentation_types = ["speed_up", "slow_down", "add_noise"]
        
        for file, label in tqdm(zip(files, labels), total=len(files), desc=f"Generating Augmented Spectrograms for {output_dir}"):
            # Create digit subdirectory
            digit_dir = os.path.join(output_dir, str(label))
            os.makedirs(digit_dir, exist_ok=True)
            
            # Load original audio
            try:
                y, sr = librosa.load(file, sr=None)
                
                # For each augmentation type
                for aug_type in augmentation_types:
                    # Apply augmentation
                    augmented_y = augment_audio(y, sr, aug_type)
                    
                    # Generate spectrogram
                    spectrogram = generate_spectrogram(augmented_y, sr)
                    
                    if spectrogram is not None:
                        # Generate output filename
                        base_filename = os.path.splitext(os.path.basename(file))[0]
                        output_filename = f"{base_filename}_{aug_type}_spectrogram.png"
                        output_path = os.path.join(digit_dir, output_filename)
                        
                        # Save spectrogram
                        plt.imsave(output_path, spectrogram, cmap='viridis')
                        
            except Exception as e:
                print(f"Error processing augmentation for {file}: {e}")

    # Generate and save regular spectrograms
    save_spectrograms(train_files, train_labels, train_dir)
    save_spectrograms(test_files, test_labels, test_dir)
    
    # Generate and save augmented spectrograms
    save_augmented_spectrograms(train_files, train_labels, aug_train_dir)
    save_augmented_spectrograms(test_files, test_labels, aug_test_dir)

    # Print dataset statistics
    print("\nDataset Split Statistics:")
    print(f"Total Samples: {len(audio_files)}")
    print(f"Training Samples: {len(train_files)} ({len(train_files)/len(audio_files)*100:.2f}%)")
    print(f"Testing Samples: {len(test_files)} ({len(test_files)/len(audio_files)*100:.2f}%)")
    
    # Digit distribution
    print("\nDigit Distribution:")
    for digit in range(10):
        train_digit_count = np.sum(train_labels == digit)
        test_digit_count = np.sum(test_labels == digit)
        print(f"Digit {digit}: Train = {train_digit_count}, Test = {test_digit_count}")
    
    # Augmented dataset statistics
    print("\nAugmented Dataset Statistics:")
    print(f"Total Original Samples: {len(audio_files)}")
    print(f"Total Augmented Samples: {len(audio_files) * 3}")  # 3 augmentations per sample

def main():
    # Paths (adjust as necessary)
    input_base_dir = 'AudioMNIST/data'
    output_base_dir = 'AudioMNIST_spectrograms'
    augmented_output_dir = 'AudioMNIST_augmented_spectrograms'
    
    # Split the dataset and generate spectrograms (both regular and augmented)
    split_and_generate_spectrograms(input_base_dir, output_base_dir, augmented_output_dir)

if __name__ == "__main__":
    main()