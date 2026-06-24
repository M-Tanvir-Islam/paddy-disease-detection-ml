# exp05 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + no weighted loss.

## Goal

Measure how much augmentation alone helps compared with exp02.

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp05_mobilenetv3large_augmentation/train.py
python scripts/evaluate_checkpoint.py experiments/exp05_mobilenetv3large_augmentation
```

## Notes

- Test set remains locked during this experiment.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.
