# exp02 - MobileNetV3-Large baseline

## What this experiment tests

Second candidate in the Phase 1 bake-off (see
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md)). Identical
pipeline to exp01 — only the model differs. Tests whether the extra
capacity of the Large variant (~2.7× more params) gives a meaningful
accuracy lift over MobileNetV3-Small at acceptable cost.

## Why MobileNetV3-Large

- ~4.2 M params (vs 1.5 M in exp01) — same family, isolates "depth + width"
- Projected ONNX size ~17 MB (well under the 50 MB budget)
- Projected CPU latency ~300–500 ms (could be near the 500 ms ceiling)
- exp01 hit 0.925 val macro-F1 with severe overfitting — Large may have
  the same problem (capacity wasn't the bottleneck) OR may pull ahead
  because of richer features

## Setup

- Dataset: Kaggle Paddy 2022 only — identical 70/15/15 stratified split, seed=42
- Augmentation: none (baseline)
- Loss: plain CrossEntropyLoss (baseline)
- Optimizer: AdamW
- Training: Phase A (5 epochs, head only, lr=3e-4) → Phase B (35 epochs, full, lr=5e-5, cosine)
- Mixed precision: on (CUDA)

## How to run

```powershell
conda activate krishidoc_ml
python experiments/exp02_mobilenetv3large_baseline/train.py
```

Expected runtime: ~35–50 min on RTX 3060 (longer per epoch than exp01 due to higher param count).

## Results

_Fill in after training completes._

| Metric                   | Value |
| ------------------------ | ----- |
| Val macro-F1             | _TBD_ |
| Val accuracy             | _TBD_ |
| Top-2 / Top-3 accuracy   | _TBD_ |
| Macro AUC-OvR            | _TBD_ |
| Best epoch               | _TBD_ |
| Total params             | _TBD_ |
| MACs                     | _TBD_ |
| Checkpoint size          | _TBD_ |
| CPU latency (mean / p95) | _TBD_ |
| GPU latency (mean / p95) | _TBD_ |
| Total training time      | _TBD_ |
| Peak VRAM                | _TBD_ |

## Confusion matrix

See `results/confusion_matrix_val.png`.

## What actually happened

_2–3 sentences after the run. Compare against exp01's 0.9251 baseline._

## Key insight

_One takeaway. Did extra capacity help? Where did it help vs hurt?_

## If this model wins the bake-off — optimization plan

Same plan as exp01 (apply on top of exp02 weights):

1. **+ augmentation** (`data.augmentation: true`) — reduces overfitting
2. **+ weighted CrossEntropy** (`data.weighted_loss: true`) — lifts minority-class recall
3. **+ Dhan-Shomadhan Field dataset** — robustness to deployment distribution
4. **+ label smoothing (0.1)** (optional) — calibration + small accuracy bump

Final winner gets ONNX export and test-set evaluation per
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md) Phase 3.
