# exp13 tmp log

Branch-local notes for EfficientNet-B0 OneCycleLR experiment.

## Goal

Measure whether OneCycleLR can rescue/improve the EfficientNet-B0 augmentation + weighted-loss recipe.

## Planned Run

| Variant | Aug | WL | LR config | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp13 | Yes | Yes | OneCycle max_lr 1e-3 | complete | 0.9701 | 0.9718 | 26.01 | +0.0372 vs exp12; +0.0092 vs exp03 |

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

## Completed Summary

| Experiment | Aug | WL | LR config | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | Best epoch | downy_mildew F1 | hispa F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp03 baseline | No | No | head 3e-4 / full 5e-5 | 0.9609 | 0.9641 | 25.79 | 32.80 | 30 | 0.9189 | 0.9538 |
| exp12 | Yes | Yes | head 3e-4 / full 5e-5 | 0.9329 | 0.9379 | 24.07 | 31.30 | 38 | 0.8796 | 0.9409 |
| exp13 | Yes | Yes | OneCycle max_lr 1e-3 | 0.9701 | 0.9718 | 26.01 | 34.85 | 36 | 0.9263 | 0.9684 |

Conclusion: OneCycleLR is useful for EfficientNet-B0 and is the best EfficientNet result so far. Test set was not evaluated.
