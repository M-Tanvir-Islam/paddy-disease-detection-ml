# exp17 - MobileNetV3-Large with Label Smoothing and EMA Weights

## Hypothesis

Validation macro-F1 is saturated at 0.98 on the grouped split, so the
remaining headroom is in how well-calibrated and stable the final weights
are rather than in raw accuracy. Two standard, low-risk regularisation tricks
are expected to hold validation macro-F1 at or above exp16 while producing a
smoother, better-calibrated model for export:

- **label smoothing** (0.1) discourages over-confident logits, which matters
  because the product applies a confidence threshold server-side;
- an **exponential moving average** of the weights averages out the
  epoch-to-epoch volatility seen in exp16 after the OneCycle peak.

## Changed Variables

Relative to exp16, only `config.yaml` and the training entrypoint change:

| Setting                    | exp16     | exp17                                             |
| -------------------------- | --------- | ------------------------------------------------- |
| `training.label_smoothing` | 0.0       | 0.1                                               |
| `training.ema_decay`       | none      | 0.999 (decay ramped up over the first ~100 steps) |
| Checkpoint weights         | raw model | EMA model                                         |

Model, ImageNet initialization, split (`kaggle_grouped_v2`), augmentation,
weighted CrossEntropy, OneCycleLR schedule, batch size, epoch budget, and seed
are unchanged. The shared `src/loss.py` gained an optional `label_smoothing`
argument with default 0.0, so no earlier experiment is affected.

The two changes are bundled deliberately: each alone is expected to move
validation macro-F1 by less than the +0.003 meaningful-gain threshold, and the
aim is a robustness recipe rather than a leaderboard gain.

## Data Protocol

Identical to exp16: 7,241 train / 1,554 validation / 1,554 locked test images
from `kaggle_grouped_v2`, seed 42. The test set is not evaluated.

## Result

Pending.

## Next Step

Pending.
