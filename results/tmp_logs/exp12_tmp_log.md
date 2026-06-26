# exp12 tmp log

Branch-local notes for EfficientNet-B0 weighted-loss experiment.

## Goal

Measure whether weighted CrossEntropy helps EfficientNet-B0 after augmentation.

## Planned Run

| Variant | Aug | WL | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp12 | Yes | Yes | complete | 0.9329 | 0.9379 | 24.07 | -0.0140 vs exp11; -0.0280 vs exp03 |

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

## Completed Summary

| Experiment | Aug | WL | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | Best epoch | downy_mildew F1 | hispa F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp03 baseline | No | No | 0.9609 | 0.9641 | 25.79 | 32.80 | 30 | 0.9189 | 0.9538 |
| exp11 | Yes | No | 0.9469 | 0.9520 | 24.35 | 30.74 | 37 | 0.8663 | 0.9522 |
| exp12 | Yes | Yes | 0.9329 | 0.9379 | 24.07 | 31.30 | 38 | 0.8796 | 0.9409 |

Conclusion: weighted loss on top of augmentation hurt EfficientNet-B0 further. Test set was not evaluated.
