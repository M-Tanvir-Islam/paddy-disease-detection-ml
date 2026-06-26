# exp13 tmp log

Branch-local notes for EfficientNet-B0 OneCycleLR experiment.

## Goal

Measure whether OneCycleLR can rescue/improve the EfficientNet-B0 augmentation + weighted-loss recipe.

## Planned Run

| Variant | Aug | WL | LR config | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp13 | Yes | Yes | OneCycle max_lr 1e-3 | pending | - | - | - | changed only scheduler vs exp12 |

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp13_efficientnet_b0_onecycle/train.py
```

## Notes

- Test set remains locked.
- EfficientNet baseline to beat: exp03 macro-F1 0.9609.
- Prior branch: exp12 macro-F1 0.9329.
- Keep augmentation and weighted loss enabled.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.