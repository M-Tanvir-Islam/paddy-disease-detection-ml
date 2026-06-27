# exp14 - EfficientNet-B0 OneCycleLR max_lr Sweep

## Purpose

Tune `onecycle_max_lr` for the best EfficientNet-B0 recipe found in exp13.

Base configuration is exp13:

- EfficientNet-B0
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- input size 224
- OneCycleLR after 5-epoch head warmup
- batch size 64
- 40 epochs
- seed 42

## Hypothesis

EfficientNet-B0 improved strongly when switching from fixed LR fine-tuning to OneCycleLR in exp13. This experiment tests whether the peak OneCycle learning rate was too high, too low, or close to optimal.

## Changed Variable

Only `training.onecycle_max_lr` changed. Model, data split, augmentation, weighted loss, batch size, epochs, seed, and OneCycle shape stayed fixed.

## Baseline To Beat

| Experiment | max_lr | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| exp13 | 1e-3 | 0.9701 | 0.9718 | 26.01 | previous best EfficientNet |

## Sweep Results

| Variant | Config | onecycle_max_lr | Best epoch | Val macro-F1 | Val Acc | CPU ms | Delta vs exp13 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| maxlr_5e4 | `configs/maxlr_5e4.yaml` | 5e-4 | 38 | 0.9713 | 0.9744 | 24.01 | +0.0012 | best exp14 variant |
| maxlr_1e3 | `configs/maxlr_1e3.yaml` | 1e-3 | 39 | 0.9633 | 0.9686 | 24.70 | -0.0068 | repeat underperformed exp13 |
| maxlr_15e4 | `configs/maxlr_15e4.yaml` | 1.5e-3 | 35 | 0.9697 | 0.9731 | 22.59 | -0.0004 | close, but weaker downy_mildew F1 |

## Result

The best EfficientNet-B0 setting from this sweep is `maxlr_5e4`, with validation macro-F1 `0.9713`, validation accuracy `0.9744`, and CPU latency `24.01 ms`.

This is a small improvement over exp13 (`+0.0012` macro-F1), so `maxlr_5e4` becomes the preferred EfficientNet-B0 configuration. However, it still does not beat the current MobileNetV3-Large winner from exp09 (`0.9779` macro-F1, `15.02 ms` CPU), so MobileNetV3-Large remains the stronger cross-track candidate.

## Interpretation

A lower OneCycle peak LR helped EfficientNet-B0 slightly. The original `1e-3` peak was not consistently reproducible here, and `1.5e-3` was close but had weaker `downy_mildew` F1. The best result came from a gentler cycle that likely reduced destructive updates after unfreezing while still allowing enough adaptation.

## Next Step

Use `maxlr_5e4` as the EfficientNet-B0 representative in comparison reports. Do not continue EfficientNet tuning unless there is a specific reason, because MobileNetV3-Large exp09 remains both more accurate and faster. The next meaningful step is cross-track comparison and later field-data/domain validation, while keeping the test set locked until final Phase 3.