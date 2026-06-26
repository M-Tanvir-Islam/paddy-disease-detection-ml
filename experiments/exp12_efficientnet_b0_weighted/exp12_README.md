# exp12 - EfficientNet-B0 Weighted Loss

## Purpose

Test whether weighted CrossEntropy improves EfficientNet-B0 on top of train-only augmentation.

Base configuration is exp11:

- EfficientNet-B0
- Kaggle Paddy 2022 split
- augmentation enabled
- no weighted CrossEntropy
- input size 224
- head warmup: 5 epochs at lr 3e-4
- full fine-tune: 35 epochs at lr 5e-5
- batch size 64
- seed 42

## Changed Variable

Only `data.weighted_loss` changes:

```text
false -> true
```

Keep augmentation enabled. This isolates the weighted-loss effect on top of augmentation.

## Baselines To Compare

| Experiment | Aug | WL | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| exp03 | No | No | 0.9609 | 0.9641 | 25.79 | EfficientNet-B0 baseline |
| exp11 | Yes | No | 0.9469 | 0.9520 | 24.35 | augmentation alone hurt |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml
python experiments/exp12_efficientnet_b0_weighted/train.py
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `bacterial_leaf_blight`, `downy_mildew`, and `hispa`
- Whether weighted loss recovers the exp11 downy_mildew drop
- CPU latency and checkpoint size should stay close to exp11

## Success Criteria

Weighted loss is useful if it improves over exp11 and ideally recovers or beats exp03 macro-F1. If exp12 remains below exp03, the current augmentation + weighted-loss recipe is not enough for EfficientNet-B0.

## Result

Pending.

## Next Step

If exp12 is competitive, create `exp13_efficientnet_b0_onecycle` and test OneCycleLR. If exp12 still clearly underperforms exp03, consider a lighter EfficientNet-specific augmentation recipe or stop the EfficientNet track early.