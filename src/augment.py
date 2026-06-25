"""
Albumentations pipelines.

Baseline (augment=False) — resize + normalize only. Used in exp01 to
establish a clean baseline before testing whether augmentation helps.

Full pipeline (augment=True) — designed for close-up paddy leaf images
(per product UX gate that enforces close-up framing at inference).
Used from exp02 onward.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(input_size: int = 224, augment: bool = False):
    if not augment:
        return _basic(input_size)
    return A.Compose([
        A.Resize(int(input_size * 1.15), int(input_size * 1.15)),
        A.RandomCrop(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=10, p=0.5
        ),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.CoarseDropout(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(input_size: int = 224):
    return _basic(input_size)


def get_val_transforms(input_size: int = 224):
    """Backward-compatible alias for deterministic evaluation preprocessing."""
    return get_eval_transforms(input_size)

def _basic(input_size: int):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
