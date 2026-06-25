# exp08 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + weighted loss + OneCycleLR.

## Goal

Test whether OneCycleLR improves over the exp06 default-LR cosine setup.

## Command Run

```powershell
conda activate krishidoc_ml
python experiments/exp08_mobilenetv3large_onecycle/train.py
```

## Final Validation Result

- Val macro-F1: 0.9746
- Val accuracy: 0.9750
- Best epoch: 32
- CPU mean latency: 15.83 ms
- Checkpoint size: 17.07 MB
- Total training time: 50.4 min
- Peak VRAM: 1586.2 MB
- Test set: not evaluated

## Comparison

- Delta vs exp06 macro-F1: +0.0076
- Delta vs exp07 high macro-F1: +0.0070
- `downy_mildew`: F1 0.9524, recall 0.9677

## Notes

- OneCycleLR is now the best MobileNetV3-Large result.
- Early OneCycle high-LR phase caused temporary validation dips, but later annealing recovered strongly.
- Test set remains locked.
- Next best action: run EfficientNet-B0 track or evaluate exp08 on Dhan-Shomadhan holdout.