# exp03 - EfficientNet-B0 baseline

## What this experiment tests

Third candidate in the Phase 1 bake-off (see
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md)). Identical
pipeline to exp01 / exp02 — only the model differs. EfficientNet-B0 is
the most-cited backbone in the paddy-disease literature.

## Why EfficientNet-B0

- ~4.0 M params — similar count to MobileNetV3-Large, but heavier compute
  (~0.4 GMACs vs 0.23 GMACs for Large) due to compound scaling
- Projected ONNX size ~20 MB (under the 50 MB budget)
- The `others_model/` notebook used B4 at 75 MB ONNX (out of budget);
  B0 tests whether the EfficientNet family is competitive at small sizes
- Reference architecture for "compound scaling" — important for the paper's
  related-work section

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
python experiments/exp03_efficientnetb0_baseline/train.py
```

Expected runtime: **~50–65 min on RTX 3060** — slowest of the bake-off
candidates because of higher MACs. CPU latency expected near 30–40 ms
(still well under the 500 ms budget).

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

_2–3 sentences after the run. Compare against exp01 (0.9251) and exp02 (0.9518)._

## Key insight

_One takeaway. Is the EfficientNet family worth the extra compute over
MobileNetV3-Large?_

## If this model wins the bake-off — optimization plan

Same plan as exp01:

1. **+ augmentation** (`data.augmentation: true`)
2. **+ weighted CrossEntropy** (`data.weighted_loss: true`)
3. **+ Dhan-Shomadhan Field dataset**
4. **+ label smoothing (0.1)** (optional)

Note: B0 is the slowest CPU-inference candidate. If it wins on accuracy
but the optimized version exceeds the 500 ms CPU budget under augmentation,
fall back to the next-best model that meets all production constraints.

Final winner gets ONNX export and test-set evaluation per
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md) Phase 3.
