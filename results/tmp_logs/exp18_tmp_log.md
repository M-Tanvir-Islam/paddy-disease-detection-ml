# exp18 temporary log

## Setup

- Parent experiment: exp14 winning `maxlr_5e4` configuration
- Changed variable: `data.split_csv` only
- New split: `data/splits/kaggle_grouped_v2.csv` (data commits cherry-picked
  from the exp16 branch)
- Augmentation: EfficientNet-track `src/augment.py`, unchanged from exp14
- Shared code change: Matplotlib `Agg` backend in `src/evaluate.py`
- Initialization: fresh ImageNet weights; exp14 checkpoint is not reused
- Test evaluation: locked and not run

## Runs

### Completed run

- Validation macro-F1: 0.9765 (best epoch 34 of 40)
- Validation accuracy: 0.9775
- Training time: 55.8 minutes; peak VRAM 2,977 MB
- Logged CPU latency 24.57 / 32.04 ms; back-to-back re-measurement with
  exp16 gives 23.2 ms vs 15.5 ms
- Test set: not evaluated

### Cross-track comparison

- `compare_with_exp16.py`: exp16 0.9800 / 29 errors / ECE 0.0128 vs exp18
  0.9765 / 35 errors / ECE 0.0147
- Decision: exp16 MobileNetV3-Large is the cross-track winner for Phase 3
