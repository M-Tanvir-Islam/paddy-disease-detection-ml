# exp07 - MobileNetV3-Large LR Sweep

## Purpose

This branch tests whether learning-rate tuning improves the current best MobileNetV3-Large setup before adding new data or label smoothing.

Base configuration is exp06:

- MobileNetV3-Large
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- input size 224
- batch size 64
- 5 head warmup epochs, 40 total epochs
- seed 42

## Question

Does a different head/full fine-tuning LR improve validation macro-F1 or training stability on top of augmentation + weighted loss?

## Sweep Variants

| Variant | Config | Head LR | Full LR | Purpose |
| --- | --- | ---: | ---: | --- |
| high | `configs/high.yaml` | 1e-3 | 1e-4 | Test faster learning and possible instability after unfreezing |
| default | `configs/default.yaml` | 3e-4 | 5e-5 | Reproduce current exp06 LR recipe for comparison |
| low | `configs/low.yaml` | 1e-4 | 1e-5 | Test slower, more conservative fine-tuning |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml

python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/high.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/default.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/low.yaml
```

Re-evaluate a finished variant on validation:

```powershell
python scripts/evaluate_checkpoint.py experiments/exp07_mobilenetv3large_lr_sweep
```

Note: `evaluate_checkpoint.py` expects `config.yaml`. For sweep variants, either evaluate from the training-produced metrics or temporarily copy the chosen variant config to `config.yaml` before re-evaluation.

## What To Compare

- Validation macro-F1 first
- Validation accuracy second
- Per-class F1/recall, especially `downy_mildew`
- Training loss shape
- Validation macro-F1 curve smoothness
- Whether high LR spikes or degrades after unfreezing
- Whether low LR underfits or plateaus early
- Best epoch
- CPU latency and checkpoint size should stay effectively unchanged

A gain of at least +0.003 macro-F1 over exp06 is meaningful enough to consider adopting the new LR recipe.

## Baseline To Beat

| Metric | exp06 weighted loss |
| --- | ---: |
| Val macro-F1 | 0.9670 |
| Val accuracy | 0.9673 |
| CPU mean latency | 19.89 ms |
| Best epoch | 22 |
| `downy_mildew` F1 | 0.9424 |
| `downy_mildew` recall | 0.9677 |

## Result

Pending.

## Next Step

After the sweep, choose the best LR recipe. Then decide whether to continue with Dhan-Shomadhan external validation / data work or run the EfficientNet-B0 track.