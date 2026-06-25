# exp10 tmp log

Branch-local notes for MobileNetV3-Large input resolution sweep.

## Goal

Find whether increasing input size from 224 improves the exp09 best MobileNetV3-Large setup.

## Planned Runs

| Variant | Input | Status | Val macro-F1 | Val Acc | CPU ms | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| input_256 | 256 | complete | 0.9718 | 0.9718 | 17.90 | worse than exp09; reject |
| input_288 | 288 | skipped | - | - | - | early-stopped after 256 underperformed |

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

## Completed Summary

| Variant | Input | Val macro-F1 | Val Acc | CPU ms | CPU p95 ms | MACs G | Best epoch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| input_256 | 256 | 0.9718 | 0.9718 | 17.90 | 23.72 | 0.304 | 31 | -0.0061 macro-F1 vs exp09; slower |
| input_288 | 288 | - | - | - | - | - | - | skipped after input_256 missed success criteria |

Conclusion: reject higher input resolution for the current MobileNetV3-Large track. Keep exp09 at 224. Test set was not evaluated.
