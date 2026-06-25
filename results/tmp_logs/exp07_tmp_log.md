# exp07 tmp log

Branch-local notes for MobileNetV3-Large LR sweep.

## Goal

Compare high/default/low learning-rate recipes on top of exp06.

## Completed Runs

| Variant | Head LR | Full LR | Status | Val macro-F1 | Val Acc | CPU ms | Best epoch | Notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| high | 1e-3 | 1e-4 | complete | 0.9676 | 0.9693 | 16.35 | 33 | best sweep score, but only +0.0006 vs exp06 |
| default | 3e-4 | 5e-5 | complete | 0.9612 | 0.9635 | 17.19 | 23 | under exp06 rerun result |
| low | 1e-4 | 1e-5 | complete | 0.9006 | 0.9039 | 16.22 | 40 | underfit / too slow |

## Commands Run

```powershell
conda activate krishidoc_ml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/high.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/default.yaml
python experiments/exp07_mobilenetv3large_lr_sweep/train.py --config configs/low.yaml
```

## Notes

- Test set was not evaluated.
- High LR is the numerical winner, but below the +0.003 meaningful-gain threshold.
- Exp06 remains the conservative base because it had stronger `downy_mildew` recall.
- Low LR should not be used for this model/setup.