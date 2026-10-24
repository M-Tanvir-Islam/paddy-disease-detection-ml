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

Pending.

## Next Step

Pending.
