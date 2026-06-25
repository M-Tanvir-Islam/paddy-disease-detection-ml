# exp06 tmp log

Branch-local notes for MobileNetV3-Large + augmentation + weighted loss.

## Goal

Measure whether weighted CrossEntropy adds useful gains on top of the exp05 augmentation setup.

## Commands Run

```powershell
conda activate krishidoc_ml
python experiments/exp06_mobilenetv3large_weighted/train.py
python scripts/evaluate_checkpoint.py experiments/exp06_mobilenetv3large_weighted
```

## Final Validation Result

- Val macro-F1: 0.9670
- Val accuracy: 0.9673
- Best epoch: 22
- CPU mean latency: 19.89 ms
- Checkpoint size: 17.07 MB
- Total training time: 53.0 min
- Peak VRAM: 1586.2 MB
- Test set: not evaluated

## Notes

- Delta vs exp05 macro-F1: +0.0097.
- Delta vs exp05 accuracy: +0.0064.
- `downy_mildew` improved from F1 0.9162 / recall 0.8817 to F1 0.9424 / recall 0.9677.
- Weighted loss helped enough to continue Track A from exp06.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.