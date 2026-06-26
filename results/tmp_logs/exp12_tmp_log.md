# exp12 tmp log

Branch-local notes for EfficientNet-B0 weighted-loss experiment.

## Goal

Measure whether weighted CrossEntropy helps EfficientNet-B0 after augmentation.

## Planned Run

| Variant | Aug | WL | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp12 | Yes | Yes | pending | - | - | - | changed only weighted_loss vs exp11 |

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp12_efficientnet_b0_weighted/train.py
```

## Notes

- Test set remains locked.
- Current EfficientNet baseline: exp03 macro-F1 0.9609.
- Current prior branch: exp11 macro-F1 0.9469.
- Keep augmentation enabled in this experiment.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.