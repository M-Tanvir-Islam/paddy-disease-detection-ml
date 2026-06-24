# exp06 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + weighted loss.

## Goal

Measure whether weighted CrossEntropy adds useful gains on top of the exp05 augmentation setup.

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp06_mobilenetv3large_weighted/train.py
python scripts/evaluate_checkpoint.py experiments/exp06_mobilenetv3large_weighted
```

## Notes

- Test set remains locked during this experiment.
- Compare primarily against exp05, not exp02.
- Watch minority/fragile class recall, especially `downy_mildew`.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.