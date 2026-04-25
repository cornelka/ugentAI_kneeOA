import shutil
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFile, ImageEnhance
from PIL.PngImagePlugin import PngInfo

g_seed = 42


def make_image_dataframe(src_path: str) -> pd.DataFrame:
    rows = []

    for split in ["train", "val", "test"]:
        for label in ["0", "1"]:
            folder = Path(src_path) / split / label

            for file in folder.glob("*.png"):

                im = Image.open(file)
                name = file.stem.split("_")[0]  # e.g. "123456L" from "123456L_xxx_9

                if name[-1] not in ["L", "R"]:
                    continue

                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "filename": file.name,
                        "id": name[:-1],
                        "side": name[-1],
                        "folder": f"{split}/{label}",
                        "img_format": im.format,
                        "img_size": im.size,
                        "img_mode": im.mode,
                        "img_info": im.info,
                    }
                )

    images_df = pd.DataFrame(rows)

    # additional step to add the information from foreign objects; might be usefull when evaluating classification results.
    foreign_objects = {
        ("train/0", "9082640L.png"),
        ("train/0", "9160026L.png"),
        ("train/0", "9170536L.png"),
        ("train/1", "9008322L.png"),
        ("train/1", "9065272L.png"),
        ("train/1", "9070442L.png"),
        ("train/1", "9425996L.png"),
        ("train/1", "9510943L.png"),
        ("train/1", "9529676R.png"),
        ("val/0", "9031141L.png"),
        ("val/1", "9375300L.png"),
        ("val/1", "9387265L.png"),
        ("test/1", "9087632L.png"),
        ("test/1", "9559547R.png"),
        ("test/1", "9688649L.png"),
    }

    images_df["foreign_objects"] = (
        images_df[["folder", "filename"]]
        .apply(tuple, axis=1)
        .isin(foreign_objects)
        .astype(int)
    )

    return images_df


def resample_xray(im, size=(128, 128)):
    info = {}
    info["resample_size"] = size
    out = im.resize(
        size,
        resample=Image.Resampling.LANCZOS,
        reducing_gap=3.0,
    )
    out.info.update(im.info)
    out.info.update(info)
    return out


def transpose_xray(im, rnd=True):
    # Use this function to transpose the image ()
    # Base size = 128*128; but you can provide different size when needed.
    out = im
    info = {}
    info["transposed"] = "False"
    if rnd:
        if np.random.choice([True, False]):
            out = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            info["transposed"] = "True"
    else:
        out = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        info["transposed"] = "True"

    out.info.update(im.info)
    out.info.update(info)
    return out


def rotate_xray(im, range=(-5, 5)):
    # use small rotation degrees +-5; more might produce unrealistic samples
    info = {}
    rand_angle = np.random.uniform(*range)
    info["rotation_range"] = range
    info["rotation_angle"] = rand_angle
    out = im.rotate(
        angle=rand_angle,
        resample=Image.Resampling.BICUBIC,  # BICUBIC for higher quality (medical images)
        expand=False,  # keep size
        fillcolor=0,
    )
    out.info.update(im.info)
    out.info.update(info)
    return out


def brightness_xray(im, range=(0.9, 1.1)):
    # use small brightness adjustments (max 10%)
    info = {}
    rand_brightness = np.random.uniform(*range)
    info["brigthness_range"] = range
    info["brigthness"] = rand_brightness

    out = ImageEnhance.Brightness(im).enhance(rand_brightness)
    out.info.update(im.info)
    out.info.update(info)
    return out


def contrast_xray(im, range=(0.85, 1.15)):
    # use small contrast adjustments (max 15%)
    info = {}
    rand_contrast = np.random.uniform(*range)
    info["contrast_range"] = range
    info["contrast"] = rand_contrast

    out = ImageEnhance.Contrast(im).enhance(rand_contrast)
    out.info.update(im.info)
    out.info.update(info)
    return out


def save_xray(im, output_path: str, embed_metadata=True):
    if embed_metadata:
        meta = PngInfo()
        for k, v in im.info.items():
            meta.add_text(k, str(v))

        im.save(output_path, pnginfo=meta)
    else:
        im.save(output_path)


def generate_imageset(
    images_frame,
    src_path: str,
    dst_path: str,
    file_suffix: str,
    conversion_function,
    add_originals=False,
    initialize_target=True,
    only_train_split=True,
) -> None:

    if only_train_split:
        images_frame = images_frame[images_frame["split"].isin(["train"])]

    # create new filenames for the next fileset
    df_paths = pd.DataFrame(
        {
            "src_path": src_path
            + "/"
            + images_frame["split"]
            + "/"
            + images_frame["label"]
            + "/"
            + images_frame["filename"],
            "dst_path": dst_path
            + "/"
            + images_frame["split"]
            + "/"
            + images_frame["label"]
            + "/"
            + images_frame["filename"].str.replace(
                r"\.png$", "_" + file_suffix + ".png", regex=True
            ),
        }
    )

    # re-create the target folders (drop existing files)
    if initialize_target:
        if only_train_split:
            dst = Path(dst_path) / "train"
        else:
            dst = Path(dst_path)

        dst = Path(dst_path)
        if dst.exists():
            shutil.rmtree(dst)

    # we recreate the folders (if needed)
    for path in df_paths["dst_path"]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    if add_originals:
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    # generate the changed images
    df_paths.apply(lambda r: conversion_function(r["src_path"], r["dst_path"]), axis=1)


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
    all_labels,
    all_preds,
    all_probs,
    class_names,
    title="",
    max_cols=5,
) -> None:
    """
    Plot images from a dataset by index with True/Pred labels.
    Title is green when prediction is correct, red when wrong.
    Wraps at max_cols columns.
    """
    n = len(indices)
    cols = min(n, max_cols)
    rows = -(-n // cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, layout="constrained"
    )
    axes_flat = axes.flat

    for ax, idx in zip(axes_flat, indices):
        img, _ = dataset[idx]
        ax.imshow(img.squeeze(), cmap="gray")
        correct = all_preds[idx] == all_labels[idx]
        ax.set_title(
            f"True: {class_names[int(all_labels[idx])]}\n"
            f"Pred: {class_names[int(all_preds[idx])]} ({all_probs[idx]:.2f})",
            color="green" if correct else "red",
            fontsize=9,
        )
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
