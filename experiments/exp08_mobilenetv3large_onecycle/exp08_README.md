# exp08 - MobileNetV3-Large + OneCycleLR

## Purpose

Test whether OneCycleLR improves the current best conservative MobileNetV3-Large setup.

Base configuration is exp06:

- MobileNetV3-Large
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- input size 224
- batch size 64
- 5 head warmup epochs, 40 total epochs
- seed 42

## Changed Variable

Only the full fine-tuning scheduler changes.

exp06 uses CosineAnnealingLR after the 5-epoch head warmup. exp08 keeps the same head warmup, then unfreezes all layers and uses OneCycleLR stepped once per training batch.

## Scheduler Setup

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs - warmup_epochs,
    pct_start=0.1,
)
```

The implementation includes `div_factor=25.0` and `final_div_factor=10000.0`, which are PyTorch defaults written explicitly in `config.yaml`.

## Baselines To Beat

| Experiment | Scheduler / LR | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| exp06 | default LR + CosineAnnealingLR | 0.9670 | 0.9673 | 19.89 | conservative base, strong `downy_mildew` recall |
| exp07 high | high LR + CosineAnnealingLR | 0.9676 | 0.9693 | 16.35 | numerical best, but only +0.0006 vs exp06 |

## Success Criteria

Treat OneCycleLR as useful only if it gives a meaningful gain, for example:

- validation macro-F1 at or above 0.9700, or
- clear per-class improvement without hurting `downy_mildew` recall.

Small gains around +0.001 should not replace exp06 as the conservative base.

## Result

Pending.

## Next Step

If OneCycleLR is not meaningfully better, proceed to Dhan-Shomadhan deployment-style validation or run the EfficientNet-B0 track.