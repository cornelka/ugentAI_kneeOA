import torch
import torch.nn as nn


def settest_torch_gpu():
    # Check available device: CUDA, MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = "cuda"
        print("PyTorch CUDA available: True")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("PyTorch MPS (Apple Silicon) available: True")
    else:
        device = "cpu"
        print("No GPU found — running on CPU")

    # Small smoke-test: run a tiny model on the selected device
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    x = torch.randn(5, 32, device=device)
    y = model(x)
    print(f"Prediction shape: {tuple(y.shape)}  — running on device: {device}")

    return device
