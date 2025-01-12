import torchvision

# Create a dummy dataset to trigger the download check
from torchvision import datasets

# This will either trigger a download or check the existing location
mnist = datasets.MNIST(root='./data')  # or use a relative path like './data'
print(f"Dataset stored in: {mnist.root}")
