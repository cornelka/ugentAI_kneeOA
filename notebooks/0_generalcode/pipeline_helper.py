from torchvision import transforms

# normalisation values determined in preprocessing, added here for re-use
g_train_mean = 0.6063
g_train_std = 0.1950
g_train_no_outliers_mean = 0.6106
g_train_no_outliers_std = 0.1915


def MyToTensor(channels: int = 1, size=(128, 128)):
    """
    Use MyToTensor for online resize and to_tensor (no normalisation)
    - channels gets a default value = 1; but we could also use 3 to fill the RGB with the same gray values
    - size: default scaling to 128/128, but you can use your own size if needed

    """

    steps = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(
            size=size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ]

    return transforms.Compose(steps)


def MyPreprocess(
    channels: int = 1,
    size=(128, 128),
    tr_mean: float = g_train_mean,
    tr_std: float = g_train_std,
    centercrop: int = 0,
):
    """
    Use MyPreprocess for online resize, to_tensor and normalization applied every validation or inference
    - channels gets a default value = 1; but we could also use 3 to fill the RGB with the same gray values
    - size: default scaling to 128/128, but you can use your own size if needed
    - tr_mean normalisation value gets the mean value found in the preprocessing step as a default
    - tr_std normalisation value gets the std value found in the preprocessing step as a default
    - centercrop: provide a non-zero value if you need this transform (for transferlearning)
    """
    steps = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(
            size=size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]

    if centercrop != 0:
        steps.append(transforms.CenterCrop(centercrop))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[tr_mean] * channels, std=[tr_std] * channels),
    ]

    return transforms.Compose(steps)


def MyTransform(
    channels: int = 1,
    size=(128, 128),
    tr_mean: float = g_train_mean,
    tr_std: float = g_train_std,
    centercrop: int = 0,
):
    """
    Use MyTransform for online augmentation, resizing, to_tensor and normalization applied every epoch during training
    - channels gets a default value = 1; but we could also use 3 to fill the RGB with the same gray values
    - size: default scaling to 128/128, but you can use your own size if needed
    - tr_mean normalisation value gets the mean value found in the preprocessing step as a default
    - tr_std normalisation value gets the std value found in the preprocessing step as a default
    - centercrop: provide a non-zero value if you need this transform (for transferlearning)
    - seed: optional int; if set, seeds PyTorch's RNG so the augmentation sequence is reproducible across runs
    """
    steps = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
        transforms.Resize(
            size=size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.15),
    ]

    if centercrop != 0:
        steps.append(transforms.CenterCrop(centercrop))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[tr_mean] * channels, std=[tr_std] * channels),
    ]

    return transforms.Compose(steps)
