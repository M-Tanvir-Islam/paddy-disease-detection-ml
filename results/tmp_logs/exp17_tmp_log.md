# exp17 temporary log

## Setup

- Parent experiment: exp16 (exp09 recipe on `kaggle_grouped_v2`)
- Changed variables: `training.label_smoothing` 0.0 -> 0.1 and a new
  `training.ema_decay` 0.999 weight average; both are standard regularisation
  tricks and are tested together because each alone is expected to sit
  below the +0.003 meaningful-gain threshold
- Shared code change: `src/loss.py::get_weighted_criterion` gained an
  optional `label_smoothing` argument (default 0.0, so earlier experiments
  are unaffected)
- Checkpoint selection and the saved `.pt` use the EMA weights; the raw
  model's per-epoch validation score is logged alongside for comparison
- Initialization: fresh ImageNet weights
- Test evaluation: locked and not run

## Runs

### Completed run

- EMA validation macro-F1: 0.9811902162 (best epoch 35 of 40)
- EMA validation accuracy: 0.9826254826
- Best raw-model epoch score: 0.9824 (epoch 34); not used for selection
- Training time: 94.3 minutes; roughly double exp16 because both the raw and
  EMA models are validated every epoch, and later epochs ran under other
  machine load (epoch time rose from ~130 s to ~210 s after epoch 34)
- Peak VRAM: 1,601.8 MB
- Logged CPU latency 37.92 / 58.54 ms was load-affected; the back-to-back
  re-measurement gives 28.9 ms for exp17 and 27.2 ms for exp16
- Test set: not evaluated

### Calibration check

- `calibration_check.py` on validation: ECE 0.1492 vs 0.0128 for exp16;
  NLL 0.223 vs 0.084; mean confidence on correct predictions 0.837 vs 0.995
- Label smoothing made the model under-confident; exp16 was already well
  calibrated
- Decision: reject exp17; exp16 stays the MobileNetV3-Large candidate
