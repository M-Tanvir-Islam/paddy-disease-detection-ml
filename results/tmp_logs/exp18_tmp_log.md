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
