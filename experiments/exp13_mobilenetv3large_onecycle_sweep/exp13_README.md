# exp13 - MobileNetV3-Large OneCycleLR max_lr Sweep

## Purpose

Push the current best MobileNetV3-Large setup by tuning only `onecycle_max_lr`.

Base configuration is exp08:

- MobileNetV3-Large
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- OneCycleLR after 5-epoch head warmup
- input size 224
- batch size 64
- 40 epochs
- seed 42

## Changed Variable

Only `training.onecycle_max_lr` changes. Keep all other training, data, augmentation, and scheduler settings fixed.

## Baseline To Beat

| Experiment | max_lr | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| exp08 | 1e-3 | 0.9746 | 0.9750 | 15.83 | current best MobileNet |

## Sweep Variants

| Variant | Config | onecycle_max_lr | Purpose |
| --- | --- | ---: | --- |
| maxlr_3e4 | `configs/maxlr_3e4.yaml` | 3e-4 | lower amplitude cycle |
| maxlr_5e4 | `configs/maxlr_5e4.yaml` | 5e-4 | moderate-low cycle |
| maxlr_1e3 | `configs/maxlr_1e3.yaml` | 1e-3 | exp08 baseline repeat |
| maxlr_15e4 | `configs/maxlr_15e4.yaml` | 1.5e-3 | moderate-high cycle |
| maxlr_2e3 | `configs/maxlr_2e3.yaml` | 2e-3 | high cycle, watch instability |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml

python experiments/exp13_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_3e4.yaml
python experiments/exp13_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp13_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp13_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
python experiments/exp13_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_2e3.yaml
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `downy_mildew`
- Training stability after unfreezing
- Best epoch
- CPU latency and checkpoint size should remain effectively unchanged

## Success Criteria

A variant should replace exp08 only if it reaches at least 0.9770 validation macro-F1 or matches exp08 while improving minority-class behavior.

## Result

Pending.

## Next Step

If a clear winner appears, optionally run a narrower second sweep around that max_lr. Otherwise keep exp08 as the best MobileNet configuration and move to EfficientNet-B0 or Dhan-Shomadhan holdout validation.