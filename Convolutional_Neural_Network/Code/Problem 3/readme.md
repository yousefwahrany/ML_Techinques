# 🎙️ AudioMNIST Classification with CNN

This project focuses on classifying spoken digits from the **AudioMNIST** dataset using a **Convolutional Neural Network (CNN)**. Below are the steps to download, preprocess, and train the model.

---

## 📥 Downloading the AudioMNIST Dataset

To get the dataset, you have two options:

### 🔹 Option 1: Automatic Download  
Run the following script to download the dataset automatically:
```bash
python download_audio_mnist.py
```

### 🔹 Option 2: Manual Download  
Alternatively, download the dataset manually from the official GitHub repository:  
🔗 **[AudioMNIST Repository](https://github.com/soerenab/AudioMNIST)**

---

## 🎛 Preparing the Dataset for Processing and Training

Before training, you need to convert the raw audio files into spectrograms, including augmented versions.

Run the following script:
```bash
python spectrograms.py
```
This will generate spectrograms for both the original and augmented audio samples, preparing them for model training.

---

## 🏋️ Training the CNN Model

To train a **LeNet-5** model on the processed dataset, run:
```bash
python model_training.py
```

At the end of `model_training.py`, you will find two configurable parameters:

- **`augment_images`**: Enables or disables image augmentation.
- **`augment_audio`**: Enables or disables audio augmentation.

Modify their values (`True`/`False`) to experiment with different variations as required in the assignment.

---