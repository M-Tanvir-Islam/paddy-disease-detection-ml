# exp09 - MobileNetV3-Large OneCycleLR max_lr Sweep

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

python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_3e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_2e3.yaml
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

Completed. The `maxlr_15e4` variant is the winner.

| Variant | onecycle_max_lr | Val macro-F1 | Val Acc | CPU ms | Best epoch | downy_mildew F1 | hispa F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxlr_3e4 | 3e-4 | 0.9686 | 0.9699 | 15.97 | 35 | 0.9149 | 0.9520 |
| maxlr_5e4 | 5e-4 | 0.9683 | 0.9705 | 15.67 | 30 | 0.9405 | 0.9600 |
| maxlr_1e3 | 1e-3 | 0.9749 | 0.9744 | 16.84 | 33 | 0.9508 | 0.9627 |
| maxlr_15e4 | 1.5e-3 | 0.9779 | 0.9763 | 15.02 | 31 | 0.9622 | 0.9617 |
| maxlr_2e3 | 2e-3 | 0.9733 | 0.9725 | 16.01 | 32 | 0.9565 | 0.9542 |

Compared with exp08, the best variant improved validation macro-F1 from 0.9746
to 0.9779, a +0.0033 gain. It also improved downy_mildew F1 from 0.9524 to
0.9622. Checkpoint size stayed 17.07 MB and CPU latency was 15.02 ms in the
validation measurement.

The 2e-3 run did not improve over 1.5e-3, so 1.5e-3 is the best observed
OneCycleLR `max_lr` for the MobileNetV3-Large track.

## Next Step

Use `maxlr_15e4` as the current best MobileNetV3-Large configuration. The next
practical step is to move to the EfficientNet-B0 track using the same staged
recipe, then compare the best MobileNet and EfficientNet candidates before
touching the locked test set.
