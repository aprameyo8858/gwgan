import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # Fall back to CPU if CUDA is not available
    print("CUDA is not available. Using CPU.")
