# exp16 - MobileNetV3-Large Group-Aware Retrain

## Hypothesis

The exp09 training recipe should remain strong when evaluated with a
leakage-resistant split in which accepted exact and near-duplicate image
components cannot cross train, validation, or test partitions.

## Changed Variable

Only the dataset split changes relative to the winning exp09 `maxlr_15e4`
configuration:

- exp09: `data/splits/kaggle.csv`
- exp16: `data/splits/kaggle_grouped_v2.csv`

The model, ImageNet initialization, augmentation, weighted CrossEntropy,
OneCycleLR settings, batch size, epoch budget, and seed remain fixed.

This is a **fresh retrain from ImageNet weights**. The exp09 checkpoint must not
be loaded because images moved by the new split may already have influenced
those weights.

## Data Protocol

- 10,349 usable images
- 7,241 train / 1,554 validation / 1,554 locked test
- 58 images from 11 cross-label duplicate groups quarantined for manual review
- zero accepted duplicate groups crossing partitions
- seed 42, approximately 70/15/15 and stratified by class

The historical `kaggle.csv` remains unchanged. The exp16 validation score is a
new, stricter baseline and should not be interpreted as a direct incremental
gain or loss against exp09's 0.9779 score.

## Run

From the repository root, either activate the environment or call Conda
directly:

```powershell
& C:\ProgramData\miniconda3\Scripts\conda.exe run -n krishidoc_ml `
  python experiments\exp16_mobilenetv3large_grouped_split\train.py
```

W&B remains disabled. This command trains and evaluates validation only; it
does not evaluate the locked test partition.

## Result

Pending training. Record validation macro-F1, per-class metrics, best epoch,
latency, and checkpoint size here after the run.

## Next Step

After exp16 finishes, inspect validation errors and stability. Do not run the
test set until the MobileNet candidate and evaluation protocol are accepted as
the Phase 3 final candidate.
