# exp05 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + no weighted loss.

## Goal

Measure how much augmentation alone helps compared with exp02.

## Commands Run

```powershell
conda activate krishidoc_ml
python experiments/exp05_mobilenetv3large_augmentation/train.py
python scripts/evaluate_checkpoint.py experiments/exp05_mobilenetv3large_augmentation
```

## Final Validation Result

- Val macro-F1: 0.9573
- Val accuracy: 0.9609
- Best epoch: 21
- CPU mean latency: 16.37 ms in latest saved metrics
- Checkpoint size: 17.07 MB
- Test set: not evaluated

## Notes

- Delta vs exp02 macro-F1: +0.0055.
- `downy_mildew` remains the weakest class: F1 0.9162, recall 0.8817.
- Result supports continuing to exp06 with weighted loss on top of augmentation.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.