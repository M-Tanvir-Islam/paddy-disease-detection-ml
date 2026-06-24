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

Pending.

## Next Step

If weighted loss improves macro-F1 or materially improves minority-class recall without damaging overall performance, continue to exp07 by adding Dhan-Shomadhan field data on top of this setup.