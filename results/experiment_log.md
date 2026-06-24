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
| exp05 | MobileNetV3-Large | Yes | No | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp02 |
| exp06 | MobileNetV3-Large | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp05 |
| exp07 | MobileNetV3-Large + Dhan-Shomadhan | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp06 |
| exp08 | MobileNetV3-Large + label smoothing | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp07 |

### Track B - EfficientNet-B0

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp09 | EfficientNet-B0 | Yes | No | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp03 |
| exp10 | EfficientNet-B0 | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp09 |
| exp11 | EfficientNet-B0 + Dhan-Shomadhan | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp10 |
| exp12 | EfficientNet-B0 + label smoothing | Yes | Yes | head 3e-4 / full 5e-5 | 224 | - | - | - | pending vs exp11 |

## Phase 3 - Final

| ID | Model | Aug | WL | LR config | Input | Val F1 | Val Acc | CPU ms | Delta vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | TBD | TBD | TBD | TBD | TBD | - | - | - | selected cross-track winner |