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

Completed. EfficientNet-B0 produced the highest Phase 1 validation macro-F1,
but with higher CPU latency than MobileNetV3-Large.

| Metric                   | Value |
| ------------------------ | ----- |
| Val macro-F1             | 0.9609 |
| Val accuracy             | 0.9641 |
| Top-2 / Top-3 accuracy   | 0.9865 / 0.9936 |
| Macro AUC-OvR            | 0.9979 |
| Best epoch               | 30 |
| Total params             | 4.020 M |
| MACs                     | 0.414 G |
| FLOPs                    | 0.828 G |
| Checkpoint size          | 16.38 MB |
| CPU latency (mean / p95) | 25.79 / 32.80 ms |
| GPU latency (mean / p95) | 12.64 / 13.17 ms |
| Total training time      | 3069.6 s |
| Peak VRAM                | 2977.0 MB |

## Confusion matrix

See `results/confusion_matrix_val.png`.

## What actually happened

EfficientNet-B0 reached 0.9609 validation macro-F1, beating MobileNetV3-Small by +0.0358 and MobileNetV3-Large baseline by +0.0091. It was slower on CPU than MobileNetV3-Large baseline, but still far below the 500 ms production budget.

## Key insight

EfficientNet-B0 is worth optimizing because it started with the highest Phase 1 macro-F1. The trade-off is latency: 25.79 ms CPU versus 15.54 ms for MobileNetV3-Large baseline, so future EfficientNet gains must justify the extra compute.

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
