# exp08 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + weighted loss + OneCycleLR.

## Goal

Test whether OneCycleLR improves over the exp06 default-LR cosine setup.

## Command

```powershell
conda activate krishidoc_ml
python experiments/exp08_mobilenetv3large_onecycle/train.py
```

## Notes

- Test set remains locked.
- Compare primarily against exp06: macro-F1 0.9670.
- Secondary comparison: exp07 high LR cosine at 0.9676.
- Watch `downy_mildew` F1/recall.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.