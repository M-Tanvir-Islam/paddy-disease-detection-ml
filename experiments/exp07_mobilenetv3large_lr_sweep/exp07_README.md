# exp07 - MobileNetV3-Large LR Sweep

## Purpose

This branch tests whether learning-rate tuning improves the current best MobileNetV3-Large setup before adding new data or label smoothing.

Base configuration is exp06:

- MobileNetV3-Large
- Kaggle Paddy 2022 split
- augmentation enabled
- weighted CrossEntropy enabled
- input size 224
- batch size 64
- 5 head warmup epochs, 40 total epochs
- seed 42

## Question

Does a different head/full fine-tuning LR improve validation macro-F1 or training stability on top of augmentation + weighted loss?

## Sweep Variants

| Variant | Config | Head LR | Full LR | Purpose |
| --- | --- | ---: | ---: | --- |
| high | `configs/high.yaml` | 1e-3 | 1e-4 | Test faster learning and possible instability after unfreezing |
| default | `configs/default.yaml` | 3e-4 | 5e-5 | Reproduce current exp06 LR recipe for comparison |
| low | `configs/low.yaml` | 1e-4 | 1e-5 | Test slower, more conservative fine-tuning |

## Baseline To Beat

| Metric | exp06 weighted loss |
| --- | ---: |
| Val macro-F1 | 0.9670 |
| Val accuracy | 0.9673 |
| CPU mean latency | 19.89 ms |
| Best epoch | 22 |
| `downy_mildew` F1 | 0.9424 |
| `downy_mildew` recall | 0.9677 |

## Results

| Variant | Head LR | Full LR | Val macro-F1 | Val Acc | CPU ms | Best epoch | Delta vs exp06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 1e-3 | 1e-4 | 0.9676 | 0.9693 | 16.35 | 33 | +0.0006 |
| default | 3e-4 | 5e-5 | 0.9612 | 0.9635 | 17.19 | 23 | -0.0058 |
| low | 1e-4 | 1e-5 | 0.9006 | 0.9039 | 16.22 | 40 | -0.0664 |

The high-LR variant produced the best sweep score, but the gain over exp06 is only +0.0006 macro-F1. That is below the pre-declared +0.003 threshold for a meaningful LR improvement.

## Curve Notes

- High LR reached the best score late at epoch 33. It improved over exp06 only marginally and showed some mid-run validation fluctuation.
- Default LR underperformed the original exp06 rerun, despite using the same LR recipe. Treat this as normal run-to-run variance or artifact of the sweep run, not a reason to discard exp06.
- Low LR clearly underfit or learned too slowly: best epoch was 40 and macro-F1 only reached 0.9006.

## Per-Class Notes

The high-LR variant had lower `downy_mildew` recall than exp06:

| Metric | exp06 | exp07 high |
| --- | ---: | ---: |
| `downy_mildew` F1 | 0.9424 | 0.9348 |
| `downy_mildew` recall | 0.9677 | 0.9247 |

So although high LR had the best overall sweep macro-F1, exp06 remains more attractive if preserving the minority-class recall gain from weighted loss is important.

## Conclusion

Do not treat LR tuning as a meaningful improvement over exp06. The high-LR variant is the best sweep run, but the gain is too small and it weakens `downy_mildew` recall compared with exp06.

Recommended base for the next MobileNetV3-Large step: keep exp06's default LR recipe (`head_lr=3e-4`, `full_lr=5e-5`) unless a future rerun confirms the high-LR gain consistently.

## Next Step

Proceed to deployment-style validation or Dhan-Shomadhan data work from the exp06/exp07 knowledge state. If training a new branch, use the exp06 default LR recipe as the conservative base.