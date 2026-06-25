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

## Result

| Metric | exp06 cosine | exp07 high cosine | exp08 OneCycleLR |
| --- | ---: | ---: | ---: |
| Val macro-F1 | 0.9670 | 0.9676 | 0.9746 |
| Val accuracy | 0.9673 | 0.9693 | 0.9750 |
| CPU mean latency | 19.89 ms | 16.35 ms | 15.83 ms |
| Best epoch | 22 | 33 | 32 |
| `downy_mildew` F1 | 0.9424 | 0.9348 | 0.9524 |
| `downy_mildew` recall | 0.9677 | 0.9247 | 0.9677 |

OneCycleLR produced a meaningful gain over both prior MobileNet results:

- +0.0076 macro-F1 vs exp06
- +0.0070 macro-F1 vs exp07 high-LR cosine
- `downy_mildew` recall stayed at 0.9677 while F1 improved to 0.9524
- CPU latency remained very low at 15.83 ms

The curve dipped during the early high-LR phase after unfreezing, then recovered strongly as the cycle annealed. The best checkpoint was selected at epoch 32.

## Conclusion

OneCycleLR is the current best MobileNetV3-Large configuration. It passes the pre-declared success criteria and should become the MobileNet base for any later MobileNet work.

## Next Step

Do not run the test set yet. Recommended next move is either:

1. Run the EfficientNet-B0 track with the same recipe so the final architecture choice is fair: augmentation, weighted loss, then OneCycleLR if useful.
2. Evaluate exp08 on Dhan-Shomadhan as deployment-style validation without training on that data.

My preference: run the EfficientNet-B0 track next, then compare the best MobileNet and EfficientNet candidates on the same Dhan-Shomadhan holdout.