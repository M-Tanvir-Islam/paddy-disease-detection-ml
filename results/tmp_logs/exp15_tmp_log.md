# exp15 tmp log

Branch-local notes for the EfficientNet-B0 baseline + OneCycleLR control.

## Goal

Measure whether OneCycleLR improves exp03 without augmentation or weighted
CrossEntropy.

## Planned Run

| Variant | Aug | Weighted loss | LR config | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| exp15 | No | No | OneCycle max LR `5e-4` | pending | - | - | - | test set remains locked |

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp15_efficientnet_b0_onecycle_control/train.py
```

## Notes

- Parent experiment: exp03.
- Only the full fine-tuning scheduler changes.
- W&B is disabled unless login and online mode are explicitly confirmed.
- Record partial, failed, or debug runs here, not in the canonical experiment
  log.
