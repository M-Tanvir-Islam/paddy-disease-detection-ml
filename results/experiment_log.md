# Experiment Log

Running table of completed and planned experiments. One row per experiment.
Primary metric is **val macro-F1**.

Use this file for clean, canonical experiment rows that are useful for reports
and research writing. Use `results/tmp_logs/expNN_tmp_log.md` for branch-local
scratch notes, partial runs, failed runs, command notes, and debugging details.

Do not record test scores here. The final winning model evaluates on the test
set once in Phase 3.

## Phase 1 - Bake-off (vanilla baselines, identical pipeline)

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | MobileNetV3-Small | No | No | head 3e-4 / full 5e-5 | 224 | 0.9251 | 0.9347 | 8.28 | baseline |
| exp02 | MobileNetV3-Large | No | No | head 3e-4 / full 5e-5 | 224 | 0.9518 | 0.9558 | 15.54 | +0.0267 vs exp01 |
| exp03 | EfficientNet-B0 | No | No | head 3e-4 / full 5e-5 | 224 | 0.9609 | 0.9641 | 25.79 | +0.0091 vs exp02 |
| exp04 | MobileViT-XXS | No | No | head 3e-4 / full 5e-5 | 224 | 0.9227 | 0.9321 | 21.57 | -0.0382 vs exp03 |

Phase 1 selected two Phase 2 candidates: exp02 as the best Pareto trade-off and
exp03 as the highest validation macro-F1 model.

## Phase 2 - Optimize The Winners

### Track A - MobileNetV3-Large

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp05 | MobileNetV3-Large | Yes | No | head 3e-4 / full 5e-5 | 224 | 0.9573 | 0.9609 | 16.37 | +0.0055 vs exp02 |
| exp06 | MobileNetV3-Large | Yes | Yes | head 3e-4 / full 5e-5 | 224 | 0.9670 | 0.9673 | 19.89 | +0.0097 vs exp05 |
| exp07 | MobileNetV3-Large LR sweep | Yes | Yes | best: head 1e-3 / full 1e-4 | 224 | 0.9676 | 0.9693 | 16.35 | +0.0006 vs exp06 |
| exp08 | MobileNetV3-Large OneCycleLR | Yes | Yes | OneCycle max_lr 1e-3 | 224 | 0.9746 | 0.9750 | 15.83 | +0.0076 vs exp06 |
| exp09 | MobileNetV3-Large OneCycle max_lr sweep | Yes | Yes | best OneCycle max_lr 1.5e-3 | 224 | 0.9779 | 0.9763 | 15.02 | +0.0033 vs exp08 |
| exp10 | MobileNetV3-Large input resolution sweep | Yes | Yes | OneCycle max_lr 1.5e-3 | 256 | 0.9718 | 0.9718 | 17.90 | -0.0061 vs exp09; rejected |

### Track B - EfficientNet-B0

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp11 | EfficientNet-B0 | Yes | No | head 3e-4 / full 5e-5 | 224 | 0.9469 | 0.9520 | 24.35 | -0.0140 vs exp03; rejected alone |
| exp12 | EfficientNet-B0 | Yes | Yes | head 3e-4 / full 5e-5 | 224 | 0.9329 | 0.9379 | 24.07 | -0.0140 vs exp11; rejected |
| exp13 | EfficientNet-B0 OneCycleLR | Yes | Yes | OneCycle max_lr 1e-3 | 224 | 0.9701 | 0.9718 | 26.01 | +0.0372 vs exp12 |
| exp14 | EfficientNet-B0 OneCycle max_lr sweep | Yes | Yes | max_lr sweep | 224 | - | - | - | planned vs exp13 |

## Phase 3 - Final

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | TBD | TBD | TBD | TBD | TBD | - | - | - | selected cross-track winner |