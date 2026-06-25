# exp10 tmp log

Branch-local notes for MobileNetV3-Large input resolution sweep.

## Goal

Find whether increasing input size from 224 improves the exp09 best MobileNetV3-Large setup.

## Planned Runs

| Variant | Input | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| input_256 | 256 | pending | - | - | - | moderate increase |
| input_288 | 288 | pending | - | - | - | larger increase; watch latency |

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp10_mobilenetv3large_resolution_sweep/train.py --config configs/input_256.yaml
python experiments/exp10_mobilenetv3large_resolution_sweep/train.py --config configs/input_288.yaml
```

## Notes

- Test set remains locked.
- Current best to beat: exp09 input 224, macro-F1 0.9779, CPU 15.02 ms.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.