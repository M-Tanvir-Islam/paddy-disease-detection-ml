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

## Changed Variable

Only `training.onecycle_max_lr` changes. Keep model, data split, augmentation, weighted loss, batch size, epochs, seed, and OneCycle shape fixed.

## Baseline To Beat

| Experiment | max_lr | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| exp13 | 1e-3 | 0.9701 | 0.9718 | 26.01 | best EfficientNet so far |

## Sweep Variants

| Variant | Config | onecycle_max_lr | Purpose |
| --- | --- | ---: | --- |
| maxlr_5e4 | `configs/maxlr_5e4.yaml` | 5e-4 | lower amplitude cycle |
| maxlr_1e3 | `configs/maxlr_1e3.yaml` | 1e-3 | exp13 repeat / baseline |
| maxlr_15e4 | `configs/maxlr_15e4.yaml` | 1.5e-3 | moderate-high cycle; watch stability |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `downy_mildew`, `hispa`, and `bacterial_leaf_blight`
- Training stability after unfreezing
- Best epoch
- CPU latency and checkpoint size should stay close to exp13

## Success Criteria

A variant replaces exp13 only if it meaningfully improves macro-F1 or improves weak-class F1 without hurting macro-F1. Compare against MobileNet exp09 after selecting the best EfficientNet variant.

## Result

Pending.

## Next Step

After the sweep, decide whether EfficientNet-B0 can compete with MobileNet exp09. If not, stop EfficientNet tuning and move to final cross-track comparison/domain validation.