import shutil
import pandas as pd
import numpy as np
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

    return pd.DataFrame(rows)


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
