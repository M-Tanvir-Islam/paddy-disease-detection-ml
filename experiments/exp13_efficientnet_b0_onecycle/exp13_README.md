# exp13 - EfficientNet-B0 OneCycleLR

## Purpose

Test whether OneCycleLR can improve the underperforming EfficientNet-B0 augmentation + weighted-loss recipe.

Base configuration is exp12:

- EfficientNet-B0
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- input size 224
- head warmup: 5 epochs at lr 3e-4
- full fine-tune: conservative fixed/cosine scheduler
- batch size 64
- seed 42

## Changed Variable

Only the full fine-tuning scheduler changes after warmup.

Exp12 used the shared conservative training loop. Exp13 uses batch-level OneCycleLR after unfreezing:

```text
onecycle_max_lr: 1e-3
pct_start: 0.1
div_factor: 25
final_div_factor: 10000
```

Keep augmentation, weighted loss, model, split, input size, batch size, epochs, and seed fixed.

## Baselines To Compare

| Experiment | Aug | WL | LR config | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp03 | No | No | head 3e-4 / full 5e-5 | 0.9609 | 0.9641 | 25.79 | EfficientNet-B0 baseline |
| exp11 | Yes | No | head 3e-4 / full 5e-5 | 0.9469 | 0.9520 | 24.35 | augmentation alone hurt |
| exp12 | Yes | Yes | head 3e-4 / full 5e-5 | 0.9329 | 0.9379 | 24.07 | weighted loss hurt further |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml
python experiments/exp13_efficientnet_b0_onecycle/train.py
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `bacterial_leaf_blight`, `downy_mildew`, and `hispa`
- Whether OneCycleLR recovers performance toward exp03
- Training stability after unfreezing
- CPU latency and checkpoint size should stay close to exp12

## Success Criteria

OneCycleLR is useful only if it clearly improves over exp12. To continue this recipe seriously, it should recover close to or above exp03's 0.9609 macro-F1. If it remains well below exp03, stop this EfficientNet recipe or switch to a lighter augmentation design.

## Result

Pending.

## Next Step

If OneCycleLR is competitive, create `exp14_efficientnet_b0_onecycle_sweep` and tune `onecycle_max_lr`. If it remains below exp03, stop the current EfficientNet augmentation + weighted-loss recipe.