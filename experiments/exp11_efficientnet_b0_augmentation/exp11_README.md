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

Completed. The MobileNet augmentation recipe did not help EfficientNet-B0 in this run.

| Experiment | Aug | WL | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | Best epoch | downy_mildew F1 | hispa F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp03 baseline | No | No | 0.9609 | 0.9641 | 25.79 | 32.80 | 30 | 0.9189 | 0.9538 | baseline |
| exp11 | Yes | No | 0.9469 | 0.9520 | 24.35 | 30.74 | 37 | 0.8663 | 0.9522 | worse macro-F1 |

Compared with exp03, exp11 changed validation macro-F1 by -0.0140. The biggest concern is downy_mildew F1, which dropped from 0.9189 to 0.8663. CPU latency stayed similar and slightly lower in this measurement, but the accuracy trade-off is not acceptable by itself.

This suggests EfficientNet-B0 may be more sensitive to the current augmentation recipe than MobileNetV3-Large, or that weighted loss/scheduler changes are needed before the augmentation recipe becomes useful.

## Next Step

Create `exp12_efficientnet_b0_weighted` from this branch and add weighted CrossEntropy to answer the planned question: does weighted loss recover or improve performance on top of augmentation? If exp12 also underperforms exp03, consider either a lighter EfficientNet-specific augmentation recipe or stopping the EfficientNet track early.
