import os
import keras
import torch


def settest_torch_gpu():
    os.environ["KERAS_BACKEND"] = "torch"  # only needed if not set globally

    # Check PyTorch CUDA
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
        f"Prediction shape: {y.shape}  — Keras is running on {keras.backend.backend()}"
    )
