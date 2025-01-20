import torch
import scipy.linalg

from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np

def sqrtm_custom(cov_matrix):
    # Convert the torch tensor to a numpy array
    cov_matrix_np = cov_matrix.cpu().numpy()
    sqrtm_matrix = scipy.linalg.sqrtm(cov_matrix_np)
    # Convert it back to a tensor and ensure it's on the same device as the original tensor
    return torch.tensor(sqrtm_matrix).to(cov_matrix.device)
# Load the pre-trained VGG16 model
def load_vgg16(device='cuda'):
    model = models.vgg16(pretrained=True).to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the necessary transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB (3 channels)
    transforms.Resize(224),  # Resize to 224x224 (required input size for VGG16)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per VGG16 requirements
])

# Function to load an image and apply transformations
def load_image(image_path):
    image = Image.open(image_path).convert("L")  # Ensure image is in grayscale (1 channel)
    
    # Apply transformation pipeline: convert to RGB, resize, and normalize
    image = transform(image)

    return image

# Function to extract features from a given model (VGG16)
def extract_features(model, image_tensor, device='cuda'):
    # Ensure the tensor has the correct batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device (GPU/CPU)
    
    with torch.no_grad():  # Disable gradient computation
        features = model.features(image_tensor)  # Forward pass through VGG16 (using only features part)
    
    # Flatten the features (channels x height x width)
    features = features.view(features.size(0), -1)  # Flatten the feature maps to shape [batch_size, num_features]
    return features

# Function to compute the FID score
# Function to compute the FID score
def calculate_fid_inception(real_images, generated_images, device='cuda'):
    # Load VGG16 model
    model = load_vgg16(device)

    # Get the features for real and generated images
    real_features = []
    generated_features = []

    for real_image, generated_image in zip(real_images, generated_images):
        real_features.append(extract_features(model, real_image, device))
        generated_features.append(extract_features(model, generated_image, device))

    # Convert lists to tensors
    real_features = torch.stack(real_features)
    generated_features = torch.stack(generated_features)

    # Flatten the features into 2D tensors [batch_size, num_features]
    real_features = real_features.view(real_features.size(0), -1)
    generated_features = generated_features.view(generated_features.size(0), -1)

    # Compute the mean and covariance for real and generated features
    mu_real = torch.mean(real_features, dim=0)
    mu_generated = torch.mean(generated_features, dim=0)

    # Covariance matrices for real and generated features
    sigma_real = torch.cov(real_features.T)  # Transpose the tensor to make it [num_features, batch_size]
    sigma_generated = torch.cov(generated_features.T)

    # Regularize covariance matrices by adding a small value to the diagonal (for numerical stability)
    epsilon = 1e-6
    sigma_real += torch.eye(sigma_real.size(0), device=sigma_real.device) * epsilon
    sigma_generated += torch.eye(sigma_generated.size(0), device=sigma_generated.device) * epsilon

    # Handle NaNs in covariance matrices
    if torch.isnan(sigma_real).any():
        print("NaN detected in sigma_real. Replacing with identity matrix.")
        sigma_real = torch.eye(sigma_real.size(0), device=sigma_real.device) * epsilon
    if torch.isnan(sigma_generated).any():
        print("NaN detected in sigma_generated. Replacing with identity matrix.")
        sigma_generated = torch.eye(sigma_generated.size(0), device=sigma_generated.device) * epsilon

    # Calculate the FID score
    diff = mu_real - mu_generated
    try:
        cov_product = torch.mm(sigma_real, sigma_generated)
        cov_mean = sqrtm_custom(cov_product)
        #cov_mean = torch.linalg.sqrtm(cov_product)
        if torch.isnan(cov_mean).any():
            print("NaN detected in cov_mean. Replacing with identity matrix.")
            cov_mean = torch.eye(sigma_real.size(0), device=sigma_real.device) * epsilon
    except RuntimeError as e:
        print(f"Error in sqrtm computation: {e}. Replacing cov_mean with identity matrix.")
        cov_mean = torch.eye(sigma_real.size(0), device=sigma_real.device) * epsilon

    trace_term = torch.trace(sigma_real + sigma_generated - 2 * cov_mean)

    # Handle NaN in the trace term
    if torch.isnan(trace_term):
        print("NaN detected in trace_term. Setting trace_term to 0.")
        trace_term = torch.tensor(0.0, device=device)

    fid_score = torch.norm(diff) ** 2 + trace_term

    # Handle NaN in the final FID score
    if torch.isnan(fid_score):
        print("NaN detected in final FID score. Returning a large default value.")
        fid_score = torch.tensor(float('inf'))

    return fid_score.item()

def calculate_fid_inception_(real_images, generated_images, device='cuda'):
    # Load VGG16 model
    model = load_vgg16(device)
    
    # Get the features for real and generated images
    real_features = []
    generated_features = []

    for real_image, generated_image in zip(real_images, generated_images):
        real_features.append(extract_features(model, real_image, device))
        generated_features.append(extract_features(model, generated_image, device))

    # Convert lists to tensors
    real_features = torch.stack(real_features)
    generated_features = torch.stack(generated_features)
    
    # Flatten the features into 2D tensors [batch_size, num_features]
    real_features = real_features.view(real_features.size(0), -1)
    generated_features = generated_features.view(generated_features.size(0), -1)
    
    # Compute the mean and covariance for real and generated features
    mu_real = torch.mean(real_features, dim=0)
    mu_generated = torch.mean(generated_features, dim=0)
    
    # Covariance matrices for real and generated features
    sigma_real = torch.cov(real_features.T)  # Transpose the tensor to make it [num_features, batch_size]
    sigma_generated = torch.cov(generated_features.T)
    
    # Regularize covariance matrices by adding a small value to the diagonal (for numerical stability)
    epsilon = 1e-6  # You can experiment with this value (increase if NaN persists)
    sigma_real += torch.eye(sigma_real.size(0), device=sigma_real.device) * epsilon
    sigma_generated += torch.eye(sigma_generated.size(0), device=sigma_generated.device) * epsilon
    
    # Check for NaNs in covariance matrices
    if torch.isnan(sigma_real).any() or torch.isnan(sigma_generated).any():
        print("NaN detected in covariance matrices!")
        return np.nan

    # Calculate the FID score
    diff = mu_real - mu_generated
    trace_term = torch.trace(sigma_real + sigma_generated - 2 * torch.sqrt(torch.mm(sigma_real, sigma_generated)))
    
    # If there is a NaN in the trace_term, return NaN
    if torch.isnan(trace_term):
        print("NaN detected in trace_term!")
        return np.nan
    
    fid_score = torch.norm(diff) ** 2 + trace_term
    
    return fid_score.item()

# Function to load images from a list of image paths
def load_image_paths(image_paths):
    images = []
    for path in image_paths:
        images.append(load_image(path))  # Load and preprocess the image
    return images

