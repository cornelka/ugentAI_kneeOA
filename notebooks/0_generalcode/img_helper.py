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
) -> None:
    """
    Plot dataset images by index.

    Prediction mode (all_labels / all_preds / all_probs / class_names provided):
        shows "True: X / Pred: Y (p)" per image; green = correct, red = wrong.

    Simple mode (titles provided, no prediction arrays):
        shows a plain string label per image — useful for data exploration.
    """
    n = len(indices)
    cols = min(n, max_cols)
    rows = -(-n // cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, layout="constrained"
    )
    axes_flat = axes.flat

    for i, (ax, idx) in enumerate(zip(axes_flat, indices)):
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

    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.show()


def overlay_heatmap(img_tensor, cam, alpha=0.5) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto a grayscale image tensor.
    img_tensor : (1, H, W) float tensor in [0, 1]
    cam        : (h, w) numpy array in [0, 1]
    Returns    : (H, W, 3) numpy array ready for imshow
    """
    img_np = img_tensor.squeeze().numpy()
    H, W = img_np.shape

    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().numpy()

    heatmap = plt.colormaps["jet"](cam_up)[:, :, :3]
    img_rgb = np.stack([img_np] * 3, axis=-1)
    return np.clip(alpha * heatmap + (1 - alpha) * img_rgb, 0, 1)


def show_gradcam(
    indices,
    display_dataset,
    test_dataset,
    grad_cam,
    all_labels,
    all_preds,
    all_probs,
    class_names,
    device,
    title="",
    n=5,
) -> None:
    """
    Plot Grad-CAM results in a 2-row grid: original image (top) + heatmap overlay (bottom).
    At most n columns. Title colour is green/red for correct/wrong predictions.
    """
    count = min(n, len(indices))
    cols = min(count, 5)
    n_groups = -(-count // cols)  # number of image/overlay row pairs
    rows = 2 * n_groups

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 6 * n_groups),
        squeeze=False,
        layout="constrained",
    )

    for i, idx in enumerate(indices[:count]):
        group = i // cols
        col = i % cols
        top = group * 2
        bot = group * 2 + 1

        img_display, _ = display_dataset[idx]
        img_tensor, _ = test_dataset[idx]
        cam = grad_cam.generate(img_tensor.unsqueeze(0).to(device))
        overlay = overlay_heatmap(img_display, cam)

        true_name = class_names[int(all_labels[idx])]
        pred_name = class_names[int(all_preds[idx])]
        color = "green" if all_preds[idx] == all_labels[idx] else "red"

        axes[top, col].imshow(img_display.squeeze(), cmap="gray")
        axes[top, col].set_title(
            f"True: {true_name}\nPred: {pred_name} ({all_probs[idx]:.2f})",
            color=color,
            fontsize=8,
        )
        axes[top, col].axis("off")
        axes[bot, col].imshow(overlay)
        axes[bot, col].set_title("Grad-CAM", fontsize=8)
        axes[bot, col].axis("off")

    # hide unused axes in the last group
    for col in range(count % cols if count % cols else cols, cols):
        axes[-2, col].axis("off")
        axes[-1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.show()
