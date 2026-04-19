from torchvision import transforms

g_seed = 42

# normalisation values determined in preprocessing, added here for re-use
g_train_main = 0.6063
g_train_std = 0.1950


def MyToTensor(img, channels: int = 1):
    tens = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize(
                size=(128, 128),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    return tens(img)


def MyPreprocess(
    img, channels: int = 1, tr_mean: float = g_train_main, tr_std: float = g_train_std
):
    # Val/test: resize + normalize only — no augmentation
    preproc = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize(
                size=(128, 128),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[tr_mean], std=[tr_std]),
        ]
    )
    return preproc(img)


def MyTransform(
    img, channels: int = 1, tr_mean: float = g_train_main, tr_std: float = g_train_std
):
    # Training: online augmentation applied every epoch before normalization
    transf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels),
            transforms.RandomHorizontalFlip(),  # left/right knee are mirror images
            transforms.RandomAffine(
                degrees=5, translate=(0.03, 0.03)  # small rotation (+-5 degrees)
            ),  # small x/y shifts (+- 3%)
            transforms.transforms.Resize(
                size=(128, 128),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.15  # exposure variation
            ),  # contrast variation
            transforms.ToTensor(),
            transforms.Normalize(mean=[tr_mean], std=[tr_std]),
        ]
    )
    return transf(img)
