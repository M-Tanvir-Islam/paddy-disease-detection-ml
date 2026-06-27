# exp14 tmp log

Branch-local notes for EfficientNet-B0 OneCycleLR max_lr sweep.

## Goal

Find whether tuning OneCycleLR `max_lr` improves over exp13.

## Completed Runs

| Variant | max_lr | Status | Best epoch | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| maxlr_5e4 | 5e-4 | complete | 38 | 0.9713 | 0.9744 | 24.01 | winner; +0.0012 vs exp13 |
| maxlr_1e3 | 1e-3 | complete | 39 | 0.9633 | 0.9686 | 24.70 | repeat underperformed exp13 |
| maxlr_15e4 | 1.5e-3 | complete | 35 | 0.9697 | 0.9731 | 22.59 | close but weaker downy_mildew F1 |

## Commands Used

```powershell
conda activate krishidoc_ml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
```

## Notes

- Test set was not evaluated.
- Current EfficientNet best after this branch: exp14 `maxlr_5e4`, macro-F1 0.9713, CPU 24.01 ms.
- Current MobileNet best to beat later: exp09 macro-F1 0.9779, CPU 15.02 ms.
- Use exp14 `maxlr_5e4` as EfficientNet representative in comparison reports.