# exp06 - MobileNetV3-Large + Augmentation + Weighted Loss

## Hypothesis

Adding class-weighted CrossEntropy on top of the exp05 augmentation setup may improve minority or fragile disease classes, especially classes with lower recall such as `downy_mildew`, without materially hurting overall validation macro-F1.

## Changed Variable

Only `data.weighted_loss` changes from `false` to `true` compared with exp05.

Augmentation remains enabled. The model, dataset split, LR schedule, input size, seed, batch size, epoch count, and augmentation recipe stay the same.

## Baseline To Beat

| Metric | exp05 augmentation |
| --- | ---: |
| Val macro-F1 | 0.9573 |
| Val accuracy | 0.9609 |
| CPU mean latency | 16.37 ms |
| Weakest class | `downy_mildew` F1 0.9162, recall 0.8817 |

## Result

| Metric | exp05 augmentation | exp06 weighted loss | Delta |
| --- | ---: | ---: | ---: |
| Val macro-F1 | 0.9573 | 0.9670 | +0.0097 |
| Val accuracy | 0.9609 | 0.9673 | +0.0064 |
| CPU mean latency | 16.37 ms | 19.89 ms | +3.52 ms |
| `downy_mildew` F1 | 0.9162 | 0.9424 | +0.0262 |
| `downy_mildew` recall | 0.8817 | 0.9677 | +0.0860 |

Best checkpoint was selected at epoch 22 with validation macro-F1 0.9670. The run stayed within the production budget: 4.215M params, 17.07 MB checkpoint, and 19.89 ms mean CPU latency.

Weighted loss improved the overall validation macro-F1 and produced a strong recall gain for `downy_mildew`, the main weak class from exp05. The trade-off is a small measured CPU latency increase, but the model architecture is unchanged and latency remains far below the 500 ms budget.

## Conclusion

Weighted CrossEntropy adds a meaningful gain on top of augmentation for the MobileNetV3-Large track. This is the current best MobileNetV3-Large result and should be the base for the next Track A experiment.

## Next Step

Create `exp07_mobilenetv3large_dhan` from this branch and change only the dataset/input data setup needed to add Dhan-Shomadhan field data.