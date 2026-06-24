# exp05 - MobileNetV3-Large + Augmentation

## Hypothesis

Adding train-time augmentation to the exp02 MobileNetV3-Large baseline should reduce overfitting and improve validation macro-F1 without changing model size or inference latency.

## Changed Variable

Only `data.augmentation` changes from `false` to `true` compared with exp02.

Weighted loss remains disabled. The model, dataset split, LR schedule, input size, seed, batch size, and epoch count stay the same.

## Augmentation Recipe

Train split only:

```python
A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
    A.HueSaturationValue(10, 25, 10, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.CoarseDropout(max_holes=6, max_height=28, max_width=28, p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])
```

The implementation uses Albumentations 2.x-compatible arguments for the noise and dropout transforms while preserving the intended strength.

## Result

Pending.

## Next Step

If validation macro-F1 improves or stays close with better generalization, continue to exp06 by adding weighted CrossEntropy on top of this augmentation setup.
