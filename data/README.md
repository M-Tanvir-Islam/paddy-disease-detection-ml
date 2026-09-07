# data/

This directory contains dataset preparation scripts and split files.
Raw image data lives in `datasets/` at the repo root (gitignored).

## Scripts

- `prepare_kaggle.py` — builds the stratified train/val/test split for the
  Kaggle Paddy 2022 dataset. Writes `splits/kaggle.csv`. No file copying.
- `prepare_grouped_kaggle.py` — builds the versioned, group-aware split used
  after the duplicate audit. Exact/verified near-duplicate components stay in
  one partition; cross-label components are written to a quarantine CSV.
- `prepare_dhan_eval.py` — maps the three Dhan-Shomadhan classes compatible
  with the locked taxonomy and prepares a cross-dataset overlap-audit manifest.
- `filter_dhan_capture_gates.py` — scores compatible Dhan images with the
  versioned blur/green-coverage proxy and writes eligible/rejected manifests.
- `verify_dataset.py` — sanity-checks the prepared CSV: filepath existence,
  class-name match, per-split counts, imbalance ratio.
- `../scripts/audit_split_leakage.py` — audit exact and perceptual duplicate
  leakage across train, validation, and test without running model evaluation.

## Workflow

```bash
python data/prepare_kaggle.py     # one-time
python data/verify_dataset.py     # re-run any time
python scripts/audit_split_leakage.py  # before final model selection/testing
python data/prepare_grouped_kaggle.py  # after reviewing the audit candidates
```

The leakage audit writes reports under
`results/data_integrity/kaggle_split_audit/`. It does not modify the split CSV
or calculate test-set model metrics. Review perceptual candidates before
changing any split assignment.

`kaggle_grouped_v2.csv` is a new evaluation protocol. Models trained on the
historical `kaggle.csv` must not be resumed or evaluated against it because
images moved between partitions may already have influenced those weights.

## Split CSV format

```csv
filepath,class,split
E:/.../train_images/blast/100123.jpg,blast,train
...
```

The split CSV is committed (small), but raw images are not.
