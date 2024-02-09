import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available. Defaulting to CPU.")
