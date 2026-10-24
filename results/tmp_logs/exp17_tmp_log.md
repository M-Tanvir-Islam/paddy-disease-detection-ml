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
