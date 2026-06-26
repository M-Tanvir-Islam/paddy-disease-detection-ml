# exp11 tmp log

Branch-local notes for EfficientNet-B0 augmentation experiment.

## Goal

Measure whether train-only augmentation improves EfficientNet-B0 over the exp03 baseline.

## Planned Run

| Variant | Aug | WL | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp11 | Yes | No | pending | - | - | - | changed only augmentation vs exp03 |

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp11_efficientnet_b0_augmentation/train.py
```

## Notes

- Test set remains locked.
- Current EfficientNet baseline to beat: exp03 macro-F1 0.9609, CPU 25.79 ms.
- Keep weighted loss disabled in this experiment.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.