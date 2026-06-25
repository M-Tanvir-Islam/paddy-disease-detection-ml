# exp07 tmp log

Branch-local notes for MobileNetV3-Large LR sweep.

## Goal

Compare high/default/low learning-rate recipes on top of exp06.

## Planned Runs

| Variant | Head LR | Full LR | Status | Val macro-F1 | Notes |
| --- | ---: | ---: | --- | ---: | --- |
| high | 1e-3 | 1e-4 | pending | - | - |
| default | 3e-4 | 5e-5 | optional/pending | - | repeat of exp06 LR recipe |
| low | 1e-4 | 1e-5 | pending | - | - |

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/high.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/default.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/low.yaml
```

## Notes

- Test set remains locked.
- Compare primarily against exp06: macro-F1 0.9670.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.