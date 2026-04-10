import os
import keras
import torch


def settest_torch_gpu():
    os.environ["KERAS_BACKEND"] = "torch"  # only needed if not set globally

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

    # Check Keras sees the GPU
    print(f"Keras backend: {keras.backend.backend()}")

    # Small Keras model on GPU
    import numpy as np

    model = keras.Sequential(
        [
            keras.Input(shape=(32,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    x = np.random.rand(100, 32).astype("float32")
    y = model.predict(x[:5])
    print(
        f"Prediction shape: {y.shape}  — Keras is running on {keras.backend.backend()} , device: {device}"
    )

    return device
