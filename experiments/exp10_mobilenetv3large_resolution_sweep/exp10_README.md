# exp10 - MobileNetV3-Large Input Resolution Sweep

## Purpose

Test whether increasing input resolution improves the current best MobileNetV3-Large configuration for disease patterns that may be small or texture-sensitive.

Base configuration is exp09 best variant:

- MobileNetV3-Large
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- OneCycleLR after 5-epoch head warmup
- OneCycle max_lr 1.5e-3
- batch size 64
- 40 epochs
- seed 42

## Changed Variable

Only `data.input_size` changes. Keep model, dataset split, augmentation recipe, weighted loss, optimizer, scheduler, seed, and epochs fixed.

## Baseline To Beat

| Experiment | Input | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| exp09 maxlr_15e4 | 224 | 0.9779 | 0.9763 | 15.02 | current best MobileNet |

## Sweep Variants

| Variant | Config | Input | Purpose |
| --- | --- | ---: | --- |
| input_256 | `configs/input_256.yaml` | 256 | moderate resolution increase |
| input_288 | `configs/input_288.yaml` | 288 | larger increase; watch latency and overfitting |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml

python experiments/exp10_mobilenetv3large_resolution_sweep/train.py --config configs/input_256.yaml
python experiments/exp10_mobilenetv3large_resolution_sweep/train.py --config configs/input_288.yaml
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `downy_mildew`, `hispa`, and `normal`
- CPU latency and checkpoint size
- Training stability and best epoch

## Success Criteria

A higher resolution replaces exp09 only if it improves validation macro-F1 by at least +0.002 to +0.003 without hurting weak classes or making CPU latency impractical.

## Result

Completed with early stop after `input_256`. The 256 run was clearly worse than exp09 and slower, so `input_288` was not run.

| Variant | Input | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | MACs G | Best epoch | downy_mildew F1 | hispa F1 | normal F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp09 baseline | 224 | 0.9779 | 0.9763 | 15.02 | 17.55 | 0.234 | 31 | 0.9622 | 0.9617 | 0.9681 | keep |
| input_256 | 256 | 0.9718 | 0.9718 | 17.90 | 23.72 | 0.304 | 31 | 0.9622 | 0.9602 | 0.9677 | reject |
| input_288 | 288 | - | - | - | - | - | - | - | - | - | skipped |

Compared with exp09, 256x256 changed validation macro-F1 by -0.0061 and CPU latency by +2.88 ms. It did not improve the weak classes enough to justify the extra compute. Because 256 already moved in the wrong direction, running 288 was not worth the training time for this stage.

The result suggests the current 224x224 preprocessing is already sufficient for the Kaggle split, and the remaining errors are more likely model/data-domain issues than image-resolution bottlenecks.

## Next Step

Keep exp09 at 224 as the best MobileNetV3-Large configuration. Move to the EfficientNet-B0 track using the same staged recipe, or run Dhan-Shomadhan deployment-style validation before spending more time on MobileNet tuning.
