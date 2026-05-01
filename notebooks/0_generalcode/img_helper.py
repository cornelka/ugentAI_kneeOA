import shutil
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_images(images, labels=None, title="", max_cols=5) -> None:
    """
    Plot images in a wrapped grid.
    `images` can be a list of file paths (str/Path), torch tensors, or numpy arrays.
    """
    n = len(images)
    cols = min(n, max_cols)
    rows = -(-n // cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, layout="constrained"
    )
    axes_flat = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_flat):
        if i < n:
            img = images[i]
            if isinstance(img, (str, Path)):
                img = mpimg.imread(img)
            elif hasattr(img, "numpy"):
                img = img.squeeze().numpy()
            else:
                img = np.asarray(img).squeeze()
            ax.imshow(img, cmap="gray")
            if labels:
                ax.set_title(labels[i], fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.show()


def show_images(
    indices,
    dataset,
    all_labels: "np.ndarray | None" = None,
    all_preds: "np.ndarray | None" = None,
    all_probs: "np.ndarray | None" = None,
    class_names: "list | None" = None,
    titles: "list | None" = None,
    title="",
    max_cols=5,
    overlays: "list | None" = None,
) -> None:
    """
    Plot dataset images by index.

    Prediction mode (all_labels / all_preds / all_probs / class_names provided):
        shows "True: X / Pred: Y (p)" per image; green = correct, red = wrong.

    Simple mode (titles provided, no prediction arrays):
        shows a plain string label per image — useful for data exploration.

    Overlay mode (overlays provided):
        each image gets a paired Grad-CAM row directly below it.
    """
    n = len(indices)
    cols = min(n, max_cols)
    n_groups = -(-n // cols)
    rows = n_groups * 2 if overlays is not None else n_groups

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, layout="constrained"
    )

    for i, idx in enumerate(indices):
        group, col = divmod(i, cols)
        top = group * 2 if overlays is not None else group

        ax = axes[top, col]
        img, _ = dataset[idx]
        ax.imshow(img.squeeze(), cmap="gray")
        if all_preds is not None:
            correct = all_preds[idx] == all_labels[idx]
            ax.set_title(
                f"True: {class_names[int(all_labels[idx])]}\n"
                f"Pred: {class_names[int(all_preds[idx])]} ({all_probs[idx]:.2f})",
                color="green" if correct else "red",
                fontsize=9,
            )
        elif titles:
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

        if overlays is not None:
            ax_bot = axes[top + 1, col]
            ax_bot.imshow(overlays[i])
            ax_bot.set_title("Grad-CAM", fontsize=8)
            ax_bot.axis("off")

    # hide unused axes in the last row group
    for j in range(n, n_groups * cols):
        group, col = divmod(j, cols)
        top = group * 2 if overlays is not None else group
        axes[top, col].axis("off")
        if overlays is not None:
            axes[top + 1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.show()
