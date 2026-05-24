# exp01 - MobileNetV3-Small baseline

## What this experiment tests

The cheapest viable production model on the cleanest single dataset, with
no augmentation and no class weighting. This is the floor — every subsequent
experiment must beat it.

## Why MobileNetV3-Small first

- ONNX size budget: < 50 MB. MobileNetV3-Small ≈ 6 MB (well under).
- CPU inference budget: < 500 ms. MobileNetV3-Small ≈ 8 ms on a desktop CPU.
- A larger model only earns its weight if the small model fails the accuracy
  target. Start small, scale up only if needed.

## Setup

- Dataset: Kaggle Paddy 2022 only (10,407 images, 10 classes)
- Split: 70 / 15 / 15 stratified, seed=42
- Augmentation: none
- Loss: plain CrossEntropyLoss
- Optimizer: AdamW, weight_decay=1e-2
- Training: Phase A (5 epochs, head only, lr=3e-4) → Phase B (35 epochs, full, lr=5e-5, cosine)
- Mixed precision: on (CUDA)

## How to run

```powershell
conda activate krishidoc_ml
python data/prepare_kaggle.py                # one time
python data/verify_dataset.py                # sanity-check
python experiments/exp01_mobilenetv3small_baseline/train.py
```

## Results

| Metric                   | Value                 |
| ------------------------ | --------------------- |
| **Val macro-F1**         | **0.9251**            |
| Val accuracy             | 0.9347                |
| Top-2 accuracy           | 0.9737                |
| Top-3 accuracy           | 0.9865                |
| Macro AUC-OvR            | 0.9956                |
| Best epoch               | 35 / 40               |
| Total params             | 1.528 M               |
| MACs                     | 0.061 G (~122 MFLOPs) |
| Checkpoint size          | 6.25 MB               |
| CPU latency (mean / p95) | 8.28 / 10.15 ms       |
| GPU latency (mean / p95) | 8.91 / 11.42 ms       |
| Total training time      | 48.1 min (RTX 3060)   |
| Mean epoch time          | 72.1 s                |
| Peak VRAM                | 613 MB                |

## Confusion matrix

See `results/confusion_matrix_val.png`.

## What actually happened

MobileNetV3-Small reached **0.9251 val macro-F1** — already above the 0.85
production target on the cleanest single dataset, with no augmentation and
no class weighting. Training loss collapsed to ~0.002 by epoch 15 while
validation plateaued around 0.92, indicating **severe overfitting**. Two
minority classes underperform: `bacterial_leaf_blight` (recall 79%) and
`downy_mildew` (recall 82%) — classic class-imbalance signature, since
both have the smallest training-set counts.

## Key insight

This small model has more than enough _capacity_ for the dataset — the
bottleneck is **regularization** (overfitting) and **class imbalance**,
not architecture. The Phase 2 optimization track (augmentation, then
weighted CrossEntropy) should each give measurable lifts on top of this
baseline. Production budget is barely touched: CPU latency 8 ms vs the
500 ms target, checkpoint 6 MB vs the 50 MB target — leaves headroom to
scale up if needed.

## Next experiment

Continue Phase 1 bake-off with the other three model candidates
(MobileNetV3-Large, EfficientNet-B0, MobileViT-XXS), then pick a winner
for Phase 2. See [`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md).

## If this model wins the bake-off — optimization plan

Apply these in order on top of exp01, each as a separate experiment.
Stop early if diminishing returns appear.

1. **exp05 — + augmentation** (`data.augmentation: true`). Hypothesis:
   cuts the train/val gap; modest macro-F1 lift. Risk: too aggressive
   augmentation could hurt this small model's capacity.
2. **exp06 — exp05 + weighted CrossEntropy** (`data.weighted_loss: true`).
   Hypothesis: lifts `bacterial_leaf_blight` and `downy_mildew` recall
   without hurting majority classes.
3. **exp07 — + Dhan-Shomadhan Field dataset** mapped into the 10-class
   taxonomy. Hypothesis: improves robustness on close-up + natural-background
   inputs (the actual deployment distribution).
4. **exp08 — + label smoothing (0.1)** in the CE loss. Hypothesis: small
   accuracy bump + better confidence calibration for the production
   "consult expert" routing.

Final winner gets ONNX export and test-set evaluation per
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md) Phase 3.
