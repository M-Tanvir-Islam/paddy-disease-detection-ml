# exp15 tmp log

Branch-local notes for the EfficientNet-B0 baseline + OneCycleLR control.

## Goal

Measure whether OneCycleLR improves exp03 without augmentation or weighted
CrossEntropy.

## Planned Run

| Variant | Aug | Weighted loss | LR config | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp15 | No | No | OneCycle max LR `5e-4` | complete | 0.9666 | 0.9667 | 25.74 | +0.0057 vs exp03; below exp14 and exp09 |

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp15_efficientnet_b0_onecycle_control/train.py
```

## Notes

- Parent experiment: exp03.
- Only the full fine-tuning scheduler changes.
- W&B is disabled unless login and online mode are explicitly confirmed.
- Record partial, failed, or debug runs here, not in the canonical experiment
  log.

## Completed Summary

- Best validation macro-F1: `0.9666` at epoch 20.
- Improvement over exp03: `+0.0057` macro-F1.
- Delta versus exp14: `-0.0047` macro-F1.
- Delta versus MobileNet exp09: `-0.0113` macro-F1.
- Validation accuracy: `0.9667`.
- CPU latency mean / p95: `25.74 / 33.50 ms`.
- Checkpoint size: `16.38 MB`.
- Training time: `3032.5 seconds`; peak VRAM: `2977.0 MB`.
- Test set was not evaluated.

Conclusion: OneCycleLR improves the clean EfficientNet baseline, but exp14
remains the best EfficientNet recipe and MobileNet exp09 remains the overall
validation-selected winner. No exp15 LR sweep is justified by the predefined
success criterion.
