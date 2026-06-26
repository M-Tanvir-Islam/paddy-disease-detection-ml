# exp11 - EfficientNet-B0 Augmentation

## Purpose

Test whether the train-only augmentation recipe that helped MobileNetV3-Large also improves EfficientNet-B0.

Base configuration is exp03:

- EfficientNet-B0
- Kaggle Paddy 2022 split
- no augmentation
- no weighted CrossEntropy
- input size 224
- head warmup: 5 epochs at lr 3e-4
- full fine-tune: 35 epochs at lr 5e-5
- batch size 64
- seed 42

## Changed Variable

Only `data.augmentation` changes:

```text
false -> true
```

Keep weighted loss disabled. This isolates the augmentation effect.

## Baseline To Beat

| Experiment | Aug | WL | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| exp03 | No | No | 0.9609 | 0.9641 | 25.79 | EfficientNet-B0 baseline |

## Commands

Run from repo root:

```powershell
conda activate krishidoc_ml
python experiments/exp11_efficientnet_b0_augmentation/train.py
```

## What To Compare

- Primary: validation macro-F1
- Secondary: validation accuracy
- Per-class F1/recall, especially `bacterial_leaf_blight`, `downy_mildew`, and `hispa`
- Training stability and best epoch
- CPU latency and checkpoint size should stay close to exp03

## Success Criteria

Augmentation is worth keeping if validation macro-F1 improves meaningfully over exp03, or if macro-F1 is similar but weak-class F1/recall improves without a major cost regression.

## Result

Pending.

## Next Step

If augmentation helps, create `exp12_efficientnet_b0_weighted` from this branch and add weighted CrossEntropy. If it clearly hurts, reconsider whether EfficientNet-B0 needs a lighter augmentation recipe before continuing.