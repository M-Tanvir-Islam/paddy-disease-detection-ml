# exp15 - EfficientNet-B0 Baseline + OneCycleLR Control

## Hypothesis

OneCycleLR may improve the clean EfficientNet-B0 baseline more than it improved
the later augmentation and weighted-loss recipe. This control closes the
interaction gap left by exp11-exp14.

## Changed Variable

Relative to exp03, only the full fine-tuning scheduler changes after the same
five-epoch classifier-head warmup.

| Setting | exp03 | exp15 |
| --- | --- | --- |
| Model | EfficientNet-B0 | EfficientNet-B0 |
| Dataset and split | Kaggle Paddy 2022 | Kaggle Paddy 2022 |
| Augmentation | No | No |
| Weighted loss | No | No |
| Input size | 224 | 224 |
| Epochs / batch size / seed | 40 / 64 / 42 | 40 / 64 / 42 |
| Full fine-tuning | fixed LR `5e-5` | OneCycleLR, max LR `5e-4` |

The `5e-4` peak is the best observed EfficientNet OneCycleLR value from exp14.
No other model, data, loss, preprocessing, or training-budget setting changes.

## Baselines

| Experiment | Aug | Weighted loss | Scheduler | Val macro-F1 | CPU mean ms |
| --- | --- | --- | --- | ---: | ---: |
| exp03 | No | No | fixed full LR | 0.9609 | 25.79 |
| exp14 best | Yes | Yes | OneCycleLR `5e-4` | 0.9713 | 24.01 |
| MobileNet exp09 | Yes | Yes | OneCycleLR `1.5e-3` | 0.9779 | 15.02 |

## Success Criteria

- Primary: validation macro-F1.
- The experiment must exceed exp03 clearly to show that OneCycleLR helps the
  clean baseline.
- A result near or above exp09 (`0.9779`) justifies a narrow follow-up sweep.
- Otherwise, close the EfficientNet track and proceed with MobileNet robustness
  validation.

## Command

Run from the repository root:

```powershell
conda activate krishidoc_ml
python experiments/exp15_efficientnet_b0_onecycle_control/train.py
```

W&B remains disabled by default. The test set is not loaded by this entrypoint.

## Result

Pending.

## Next Step

After a meaningful completed run, record the final validation metrics here and
in `results/experiment_log.md`, then compare exp15 with exp14 and MobileNet
exp09. Do not evaluate the locked test set yet.
