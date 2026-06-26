# exp14 tmp log

Branch-local notes for EfficientNet-B0 OneCycleLR max_lr sweep.

## Goal

Find whether tuning OneCycleLR `max_lr` improves over exp13.

## Planned Runs

| Variant | max_lr | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| maxlr_5e4 | 5e-4 | pending | - | - | - | lower amplitude |
| maxlr_1e3 | 1e-3 | pending | - | - | - | exp13 repeat |
| maxlr_15e4 | 1.5e-3 | pending | - | - | - | watch stability |

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
```

## Notes

- Test set remains locked.
- Current EfficientNet best to beat: exp13 macro-F1 0.9701, CPU 26.01 ms.
- Current MobileNet best to beat later: exp09 macro-F1 0.9779, CPU 15.02 ms.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.