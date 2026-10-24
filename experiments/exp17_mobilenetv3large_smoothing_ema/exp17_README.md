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

| Metric (validation, 1,554 images) | exp16 | exp17 |
| --- | ---: | ---: |
| Macro-F1 | 0.9800 | **0.9812** |
| Accuracy | 0.9813 | 0.9826 |
| Errors | 29 | 27 |
| Best epoch | 38 / 40 | 35 / 40 |
| Expected calibration error, 15 bins | **0.0128** | 0.1492 |
| Negative log-likelihood | **0.084** | 0.223 |
| Mean confidence, correct predictions | 0.995 | 0.837 |
| Mean confidence, incorrect predictions | 0.849 | 0.635 |
| Top-2 accuracy | **0.9974** | 0.9936 |
| CPU latency, mean / p95 (re-measured together) | 27.2 / 36.5 ms | 28.9 / 44.0 ms |
| Checkpoint size | 17.07 MB | 17.07 MB |

Calibration and latency figures come from `calibration_check.py`, which
evaluates both checkpoints back to back in one process
(`results/calibration_check.json`). The CPU latency logged at the end of
training (37.9 ms) was inflated by other load on the machine; the
architecture is identical to exp16, so latency is unchanged.

**Accuracy.** The +0.0012 macro-F1 gain (two fewer errors) is below the
+0.003 meaningful-gain threshold, so the two runs tie on validation score.
Per-class F1 moved by at most ±0.01; `downy_mildew` remains the weakest class
(0.9492).

**Stability.** The EMA did what it was meant to do. After the backbone
unfroze at the OneCycle peak, the raw model fell to 0.617 macro-F1 (epoch 8),
while the EMA model never dropped below 0.908 and improved almost
monotonically from epoch 9. exp16 fell to 0.643 in the same phase.

**Calibration.** The calibration hypothesis is refuted. exp16 was already
well calibrated (ECE 0.013), so label smoothing made exp17 systematically
under-confident: it is correct 98% of the time but reports a mean top-class
confidence of 0.84. NLL and top-2 accuracy also worsened, consistent with
the known effect of label smoothing compressing the logit geometry. A
product-side confidence threshold tuned on exp16 would reject many correct
exp17 predictions unless the model were recalibrated (e.g. temperature
scaling).

## Next Step

Rejected as the Track A candidate. exp16 remains the MobileNetV3-Large
candidate: it ties on accuracy and is far better calibrated without any
post-hoc correction. EMA alone could be revisited if training stability
becomes a concern, but it is not expected to change validation macro-F1
meaningfully. The next step is a fair cross-track comparison: rerun the best
EfficientNet-B0 recipe (exp14) on `kaggle_grouped_v2`.
