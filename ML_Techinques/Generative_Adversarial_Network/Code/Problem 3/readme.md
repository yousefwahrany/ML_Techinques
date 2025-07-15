# GAN and CNN Pipeline

## Overview

This project consists of two main components:
1. A **Generative Adversarial Network (GAN)** that generates synthetic image data.
2. A **Convolutional Neural Network (CNN)** that is trained on the synthetic data produced by the GAN.

To ensure everything works as intended, please follow the steps below.

---

## 📁 File Structure

- `GAN.py` – Trains the GAN and saves the generated image dataset.
- `CNN_GAN.py` – Loads the GAN-generated dataset and trains a CNN classifier on it.

---

## ⚠️ IMPORTANT: Execution Instructions

1. **Run `GAN.py` first.**
   - This script will generate a dataset of synthetic images using the GAN.
   - It includes a variable called `train` that controls whether to train the GAN or just use it for image generation.

   **Set the `train` variable as follows:**
   - `train = True` if you are training the GAN for the first time.
   - `train = False` if the model has already been trained and you only want to generate synthetic data.

   > **Note:** The generated images will be saved in a directory (e.g., `generated_data\`).

2. **Run `CNN_GAN.py` after the GAN has generated the dataset.**
   - This script will load the images from the directory created by the GAN.
   - If the dataset is missing, the CNN will not run correctly.

---
