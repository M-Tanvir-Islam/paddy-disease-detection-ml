"""
Albumentations pipelines.

Baseline (augment=False): resize + normalize only.

Experiment augmentation (augment=True): train-split-only transforms for exp05,
matching the MobileNetV3-Large augmentation experiment. Validation and test
always use the baseline transform.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(input_size: int = 224, augment: bool = False):
    if not augment:
        return _basic(input_size)

    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
        A.HueSaturationValue(10, 25, 10, p=0.5),
        # Albumentations 2.x uses std_range in normalized units. This is the
        # equivalent intent of the older var_limit=(10, 50) recipe.
        A.GaussNoise(std_range=(0.012, 0.028), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(1, 28),
            hole_width_range=(1, 28),
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = 224):
    return _basic(input_size)


def _basic(input_size: int):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
