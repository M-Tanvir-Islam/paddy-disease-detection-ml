# exp01 - MobileNetV3-Small baseline

## What this experiment tests

The cheapest viable production model on the cleanest single dataset, with
no augmentation and no class weighting. This is the floor - every subsequent
experiment must beat it.

## Why MobileNetV3-Small first

- ONNX size budget: < 50 MB. MobileNetV3-Small ~= 8 MB.
- CPU inference budget: < 500 ms. MobileNetV3-Small ~= 150-300 ms.
- A larger model (e.g. EfficientNet-B0 / B4) only earns its weight if the
  small model fails the accuracy target. Start small, scale up only if needed.

## Setup

- Dataset: Kaggle Paddy 2022 only (10,407 images, 10 classes)
- Split: 70 / 15 / 15 stratified, seed=42
- Augmentation: none
- Loss: plain CrossEntropyLoss
- Optimizer: AdamW
- Training: Phase A (5 epochs, head only, lr=3e-4) -> Phase B (35 epochs, full, lr=5e-5, cosine)
- Mixed precision: on (if CUDA available)

## How to run

```bash
python data/prepare_kaggle.py            # one time
python data/verify_dataset.py            # sanity-check
python experiments/exp01_mobilenetv3small_baseline/train.py
```

## Results

_Fill in after training completes._

| Metric        | Value |
| ------------- | ----- |
| Val macro-F1  | _TBD_ |
| Val accuracy  | _TBD_ |
| Best epoch    | _TBD_ |
| Training time | _TBD_ |

## Confusion matrix

_See `results/confusion_matrix_val.png`._

## What actually happened

_2-3 sentences after the run._

## Key insight

_One takeaway._

## Next experiment

exp02 - same model + augmentation pipeline. Tests whether close-up-simulating
crops + photometric augmentation closes the training-vs-deployment gap.
