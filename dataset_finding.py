import torch
from torchvision import datasets, transforms

# Path to the directory where MNIST is downloaded
data_dir = './data/mnist/MNIST/raw'  # Update this path based on where you stored the dataset

# Define the transformation (if any)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale, so single-channel (0.5, 0.5)
])

# Load the MNIST dataset from the local directory
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

# DataLoader to load the dataset in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Checking if the dataset is loaded correctly
for images, labels in train_loader:
    print(f"Batch of images: {images.shape}")
    print(f"Batch of labels: {labels.shape}")
    break

