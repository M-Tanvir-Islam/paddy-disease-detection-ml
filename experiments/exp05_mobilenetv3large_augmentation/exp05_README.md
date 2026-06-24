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

| Metric | exp02 baseline | exp05 augmentation | Delta |
| --- | ---: | ---: | ---: |
| Val macro-F1 | 0.9518 | 0.9573 | +0.0055 |
| Val accuracy | 0.9558 | 0.9609 | +0.0051 |
| CPU mean latency | 15.54 ms | 16.37 ms | +0.83 ms |

Best checkpoint was selected at epoch 21 with validation macro-F1 0.9573. The run stayed well within the production budget: 4.215M params, 17.07 MB checkpoint, and 16.37 ms mean CPU latency in the latest saved metrics.

Per-class weak spot remains `downy_mildew` with F1 0.9162 and recall 0.8817. Augmentation helped the overall score, but there is still room to improve minority/fragile classes.

## Conclusion

Augmentation alone produced a modest useful gain over exp02 without hurting the production constraints. The next experiment should test whether weighted CrossEntropy adds improvement on top of this augmented setup.

## Next Step

Create `exp06_mobilenetv3large_weighted` from this branch and change only `data.weighted_loss` from `false` to `true`.