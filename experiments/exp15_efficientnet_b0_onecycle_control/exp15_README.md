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

Completed. OneCycleLR improved the clean EfficientNet-B0 baseline, but did not
beat the optimized EfficientNet recipe or the MobileNet winner.

| Experiment | Aug | Weighted loss | Scheduler | Val macro-F1 | Val Acc | CPU mean ms | Best epoch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| exp03 | No | No | fixed full LR | 0.9609 | 0.9641 | 25.79 | 30 |
| exp15 | No | No | OneCycleLR `5e-4` | 0.9666 | 0.9667 | 25.74 | 20 |
| exp14 best | Yes | Yes | OneCycleLR `5e-4` | 0.9713 | 0.9744 | 24.01 | 38 |
| MobileNet exp09 | Yes | Yes | OneCycleLR `1.5e-3` | 0.9779 | 0.9763 | 15.02 | 31 |

Exp15 improved macro-F1 by `+0.0057` over exp03, confirming that OneCycleLR
helps the clean baseline. It remained `-0.0047` below exp14 and `-0.0113`
below MobileNet exp09. Accuracy improved by `+0.0026` over exp03, while CPU
latency and checkpoint size remained effectively unchanged.

Exp15 reached its best validation score at epoch 20. Later epochs did not
improve the checkpoint even as training loss approached zero, so extending the
same run is not supported by the observed curve.

Selected exp15 metrics:

- Validation macro-F1: `0.9666`
- Validation accuracy: `0.9667`
- Top-2 accuracy: `0.9898`
- Macro AUC OVR: `0.9981`
- CPU latency mean / p95: `25.74 / 33.50 ms`
- Checkpoint size: `16.38 MB`
- Best epoch: `20`
- Training time: `3032.5 seconds`
- Peak VRAM: `2977.0 MB`

## Next Step

Keep exp14 `maxlr_5e4` as the best EfficientNet-B0 representative and keep
MobileNet exp09 `maxlr_15e4` as the cross-track winner. The predefined success
criterion for a narrow exp15 learning-rate sweep was not met, so close this
EfficientNet control track rather than tuning it further.

Proceed next to the robustness plan for MobileNet exp09: seed stability,
split-leakage audit, and deployment-like field validation. Do not evaluate the
locked test set yet.
