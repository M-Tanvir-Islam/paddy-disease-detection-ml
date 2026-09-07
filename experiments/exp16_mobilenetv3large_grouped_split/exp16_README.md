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

Completed successfully.

| Metric | Result |
| --- | ---: |
| Validation macro-F1 | **0.9800** |
| Validation accuracy | 0.9813 |
| Best epoch | 38 / 40 |
| Validation errors | 29 / 1,554 |
| CPU latency, mean / p95 | 21.15 / 36.59 ms |
| GPU latency, mean / p95 | 11.19 / 12.26 ms |
| Checkpoint size | 17.07 MB |
| Peak VRAM | 1,586.2 MB |
| Training time | 48.9 minutes |

The strongest class was `bacterial_leaf_streak` (F1 1.0000). The weakest was
`downy_mildew` (F1 0.9545, recall 0.9231; 7 of 91 images missed). Its errors
were mainly predictions of `blast` and `tungro`. The run was volatile shortly
after unfreezing at the peak learning rate, but recovered and formed a stable
0.9789-0.9800 plateau over the final six epochs.

This result shows that the exp09 recipe remains strong under the corrected
group-aware protocol. It does not prove a +0.0021 improvement over exp09,
because exp09 and exp16 use different validation partitions.

## Next Step

Review the 29 validation errors, especially the seven `downy_mildew` misses,
and manually resolve or permanently exclude the 11 quarantined cross-label
groups. If that review finds no systematic data problem, accept exp16 as the
MobileNet Phase 3 candidate and perform the single locked-test evaluation.
