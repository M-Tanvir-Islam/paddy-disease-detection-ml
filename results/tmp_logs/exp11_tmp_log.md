# exp11 tmp log

Branch-local notes for EfficientNet-B0 augmentation experiment.

## Goal

Measure whether train-only augmentation improves EfficientNet-B0 over the exp03 baseline.

## Planned Run

| Variant | Aug | WL | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp11 | Yes | No | complete | 0.9469 | 0.9520 | 24.35 | -0.0140 macro-F1 vs exp03; augmentation alone rejected |

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

## Completed Summary

| Experiment | Aug | WL | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | Best epoch | downy_mildew F1 | hispa F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp03 baseline | No | No | 0.9609 | 0.9641 | 25.79 | 32.80 | 30 | 0.9189 | 0.9538 |
| exp11 | Yes | No | 0.9469 | 0.9520 | 24.35 | 30.74 | 37 | 0.8663 | 0.9522 |

Conclusion: augmentation alone hurt EfficientNet-B0 on the current split. Test set was not evaluated.
