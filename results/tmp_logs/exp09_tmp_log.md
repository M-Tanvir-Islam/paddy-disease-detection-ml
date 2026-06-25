# exp09 tmp log

Branch-local notes for MobileNetV3-Large OneCycleLR max_lr sweep.

## Goal

Find whether tuning OneCycleLR `max_lr` improves over exp08.

## Planned Runs

| Variant | max_lr | Status | Val macro-F1 | Notes |
| --- | ---: | --- | ---: | --- |
| maxlr_3e4 | 3e-4 | pending | - | - |
| maxlr_5e4 | 5e-4 | pending | - | - |
| maxlr_1e3 | 1e-3 | pending | - | exp08 repeat |
| maxlr_15e4 | 1.5e-3 | pending | - | - |
| maxlr_2e3 | 2e-3 | pending | - | watch instability |

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_3e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_2e3.yaml
```

## Notes

- Test set remains locked.
- Current best to beat: exp08 macro-F1 0.9746.
- Target for clear win: >= 0.9770 macro-F1 or same score with better minority-class behavior.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.
