# exp19 - MobileNetV3-Large Seed Replicates

## Hypothesis

exp16 is a single training run. Repeating its recipe with different random
seeds measures how much validation macro-F1 varies from run to run, so the
selected model can be reported with error bars and the +0.003
meaningful-change threshold can be checked against real seed noise.

## Changed Variable

Only `training.seed` changes relative to exp16:

| Run               | Seed | Config                                             |
| ----------------- | ---: | -------------------------------------------------- |
| exp16 (reference) |   42 | `exp16_mobilenetv3large_grouped_split/config.yaml` |
| exp19 seed 1      |    1 | `configs/seed_1.yaml`                              |
| exp19 seed 2      |    2 | `configs/seed_2.yaml`                              |

The seed controls classifier-head initialisation, training-data order, and
augmentation sampling. Model, ImageNet weights, split, augmentation, weighted
CrossEntropy, OneCycleLR settings, batch size, and epoch budget are unchanged.

Seeds 1 and 2 were fixed before any run. The deployable model remains the
seed-42 exp16 checkpoint regardless of these results: choosing the
best-scoring seed would be a further selection on the validation set.

## Data Protocol

Identical to exp16: 7,241 train / 1,554 validation / 1,554 locked test images
from `kaggle_grouped_v2`. The test set is not evaluated.

## How To Run

From the repository root:

```bash
python experiments/exp19_mobilenetv3large_seed_replicates/train.py --config configs/seed_1.yaml
python experiments/exp19_mobilenetv3large_seed_replicates/train.py --config configs/seed_2.yaml
python experiments/exp19_mobilenetv3large_seed_replicates/summarize_seeds.py
```

## Result

| Run | Seed | Val macro-F1 | Val accuracy | Best epoch | CPU ms (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp16 | 42 | 0.9800 | 0.9813 | 38 / 40 | 21.15* |
| exp19 | 1 | 0.9787 | 0.9807 | 37 / 40 | 15.59 |
| exp19 | 2 | 0.9794 | 0.9813 | 29 / 40 | 14.94 |
| **Mean ± SD** | | **0.9794 ± 0.0007** | **0.9811 ± 0.0004** | | |

\* exp16's logged latency was measured under machine load; a later
back-to-back re-measurement gave 15.5 ms, in line with the replicates.

SD is the sample standard deviation over three seeds
(`results/seed_summary.json`). With n = 3 it is a rough estimate.

Per-class F1, mean ± SD over the three seeds:

| Class | F1 |
| --- | ---: |
| bacterial_leaf_blight | 0.9817 ± 0.0040 |
| bacterial_leaf_streak | 1.0000 ± 0.0000 |
| bacterial_panicle_blight | 0.9697 ± 0.0000 |
| blast | 0.9821 ± 0.0048 |
| brown_spot | 0.9777 ± 0.0019 |
| dead_heart | 0.9969 ± 0.0013 |
| downy_mildew | **0.9454 ± 0.0087** |
| hispa | 0.9837 ± 0.0012 |
| normal | 0.9794 ± 0.0032 |
| tungro | 0.9769 ± 0.0035 |

**Stability.** The recipe is highly reproducible: the three runs span only
0.0013 macro-F1. The seed-42 exp16 run happens to be the highest of the
three, so the recipe's expected validation score is better reported as
0.9794 ± 0.0007 than as the single-run 0.9800.

**Threshold check.** The +0.003 meaningful-change threshold is about four
standard deviations of seed noise, so it is appropriately conservative for
this recipe and split.

**Earlier decisions re-checked against seed noise.**

- exp18 EfficientNet-B0 (0.9765) is below every MobileNet seed (lowest
  0.9787), so the cross-track decision does not depend on the seed.
- exp17 label smoothing + EMA (0.9812) sits 0.0018 above the MobileNet mean,
  more than seed noise alone would suggest but still below the +0.003
  threshold, and its calibration was much worse. The rejection stands.

**Weakest class.** `downy_mildew` is the weakest class in every run and also
the least stable (SD 0.0087, about ten times the macro-F1 SD). It is the
clearest target for future data collection.

## Next Step

The selected model remains the seed-42 exp16 checkpoint, as fixed before these
runs; choosing the best-scoring seed would be a further selection on
validation. Proceed to Phase 3 with exp16 and report validation performance
as 0.9794 ± 0.0007 across seeds alongside the single locked-test result.
