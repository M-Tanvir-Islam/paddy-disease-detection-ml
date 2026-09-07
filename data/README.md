# data/

This directory contains dataset preparation scripts and split files.
Raw image data lives in `datasets/` at the repo root (gitignored).

## Scripts

- `prepare_kaggle.py` — builds the stratified train/val/test split for the
  Kaggle Paddy 2022 dataset. Writes `splits/kaggle.csv`. No file copying.
- `verify_dataset.py` — sanity-checks the prepared CSV: filepath existence,
  class-name match, per-split counts, imbalance ratio.
- `../scripts/audit_split_leakage.py` — audit exact and perceptual duplicate
  leakage across train, validation, and test without running model evaluation.

## Workflow

```bash
python data/prepare_kaggle.py     # one-time
python data/verify_dataset.py     # re-run any time
python scripts/audit_split_leakage.py  # before final model selection/testing
```

The leakage audit writes reports under
`results/data_integrity/kaggle_split_audit/`. It does not modify the split CSV
or calculate test-set model metrics. Review perceptual candidates before
changing any split assignment.

## Split CSV format

```csv
filepath,class,split
E:/.../train_images/blast/100123.jpg,blast,train
...
```

The split CSV is committed (small), but raw images are not.
