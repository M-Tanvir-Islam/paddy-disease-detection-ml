# exp04 - MobileViT-XXS baseline

## What this experiment tests

Final candidate in the Phase 1 bake-off (see
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md)). First and
only transformer-based candidate. Replaces the originally-planned
ResNet18 (no headroom under the 50 MB ONNX budget, and a 2015-era
architecture).

## Why MobileViT-XXS

- **Smallest** in the bake-off — ~1.3 M params (lighter than even MobileNetV3-Small at 1.5 M)
- **Transformer-CNN hybrid** (Apple, 2022) — different inductive bias than the
  pure CNNs in exp01-03. Worth knowing for the paper's architecture-coverage table.
- **Loaded via `timm`**, not torchvision — first non-torchvision backbone
- **Transformers are data-hungry** — vanilla baseline likely underperforms
  the CNN candidates because we use no augmentation. The interesting question
  is _how much_ augmentation helps it in Phase 2 (transformers usually gain
  more from augmentation than CNNs).

## Setup

- Dataset: Kaggle Paddy 2022 only — identical 70/15/15 stratified split, seed=42
- Augmentation: none (baseline)
- Loss: plain CrossEntropyLoss (baseline)
- Optimizer: AdamW
- Input size: 224×224 (MobileViT was pretrained at 256, downsampled here for fair comparison)
- Training: Phase A (5 epochs, head only, lr=3e-4) → Phase B (35 epochs, full, lr=5e-5, cosine)
- Mixed precision: on (CUDA)

## How to run

```powershell
conda activate krishidoc_ml
python experiments/exp04_mobilevit_xxs_baseline/train.py
```

Expected runtime: **~35–45 min on RTX 3060** — transformers tend to have
higher MAC counts per param than CNNs, but XXS is the smallest model in
the bake-off so it should land somewhere between exp02 and exp03 in wall time.

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

_2–3 sentences after the run. Compare against exp01 (0.9251), exp02 (0.9518),
exp03 (0.9609). Note whether the transformer-based candidate is competitive
with the CNN-based ones at this scale._

## Key insight

_One takeaway. Does the transformer family belong in production at this
data scale, or is it an "interesting also-ran" for the paper?_

## If this model wins the bake-off — optimization plan

Same plan as exp01:

1. **+ augmentation** (`data.augmentation: true`) — transformers tend to
   gain MORE from augmentation than CNNs; this experiment may show a
   larger lift than it does for the other candidates
2. **+ weighted CrossEntropy** (`data.weighted_loss: true`)
3. **+ Dhan-Shomadhan Field dataset**
4. **+ label smoothing (0.1)** (optional)

Final winner gets ONNX export and test-set evaluation per
[`docs/EXPERIMENT_PLAN.md`](../../docs/EXPERIMENT_PLAN.md) Phase 3.
