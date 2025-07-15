import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import os
from torch.utils.data import TensorDataset, DataLoader


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

X_train_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
       # Input is 28x28x1 image
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 7*7*256, bias=False),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(0.2),
            
            # Reshape into 7x7 feature maps
            nn.Unflatten(1, (256, 7, 7)),
            
            # Transposed convolution 1
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Transposed convolution 2
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Transposed convolution 3 (output layer)
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return (self.model(z) + 1)/2
    

def show_fake_images():
    noise = torch.randn(25, 100).to(device)
    fake_imgs = G(noise).view(-1, 28, 28).cpu().detach().numpy()

    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(fake_imgs[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

sizearray = np.array([300, 700, 1000])
train = True

if train:
    for j in range(3):
        size = sizearray[j]
        for i in range(10):
            
            print(f"Train digit {i} with {size} real example")
            G_path = f"models\generator_{i}_{size}.pth"
            D_path = f"models\discriminator_{i}_{size}.pth"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize models
            G = Generator().to(device)
            D = Discriminator().to(device)

            loss_fn = nn.BCELoss()
            optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
            optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

            # Create TensorDataset and DataLoader
            train_dataset = TensorDataset(X_train_tensor[i*1000:(i*1000+size-1)])
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

            epochs = 25
            for epoch in range(epochs):
                for real_imgs in train_loader:
                    real_imgs = real_imgs[0].to(device)
                    batch_size = real_imgs.size(0)

                    real_labels = torch.ones(batch_size, 1).to(device)
                    fake_labels = torch.zeros(batch_size, 1).to(device)

                    # Generate fake images
                    noise = torch.randn(batch_size, 100).to(device)
                    fake_imgs = G(noise)

                    n_critic = 5  # Try 3–5
                    for _ in range(n_critic):
                    # Train Discriminator
                        D_real = D(real_imgs)
                        D_fake = D(fake_imgs.detach())
                        D_loss = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)

                        optimizer_D.zero_grad()
                        D_loss.backward()
                        optimizer_D.step()
                    

                    # Train Generator
                    noise = torch.randn(batch_size, 100).to(device)
                    fake_imgs = G(noise)
                    D_output = D(fake_imgs)
                    G_loss = loss_fn(D_output, real_labels)

                    optimizer_G.zero_grad()
                    G_loss.backward()
                    optimizer_G.step()

                # Save models at the end
                torch.save(G.state_dict(), G_path)
                torch.save(D.state_dict(), D_path)

                #if (epoch + 1) % 10 == 0:
                #   show_fake_images()

                print(f"Epoch {epoch+1}/{epochs} | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")


        #show_fake_images()

generatedarray = np.array([1000, 2000, 3000])

for j in range(3):
    size = sizearray[j]
    for k in range(3):
        generate = generatedarray[k]
        fake_imgs = np.empty((1, 28, 28))
        for i in range(10):

            print(f"generate {generate} image for digit {i} trained with {size} real example")
            G_path = f"models\generator_{i}_{size}.pth"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            G = Generator().to(device)
            G.load_state_dict(torch.load(G_path))

            noise = torch.randn(generate, 100).to(device)
            fake_digit = G(noise).view(-1, 28, 28).cpu().detach().numpy()
            fake_imgs = np.concatenate((fake_imgs, fake_digit))

        save_path = f"generated_data\gen_image_{size}_{generate}.npz"
        np.savez_compressed(save_path, fake_imgs)
        print("batch saved successfully")