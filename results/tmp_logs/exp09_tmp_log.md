# exp09 tmp log

Branch-local notes for MobileNetV3-Large OneCycleLR max_lr sweep.

## Goal

Find whether tuning OneCycleLR `max_lr` improves over exp08.

## Planned Runs

| Variant | max_lr | Status | Val macro-F1 | Notes |
| --- | ---: | --- | ---: | --- |
| maxlr_3e4 | 3e-4 | complete | 0.9686 | under exp08 |
| maxlr_5e4 | 5e-4 | complete | 0.9683 | under exp08 |
| maxlr_1e3 | 1e-3 | complete | 0.9749 | exp08 repeat level |
| maxlr_15e4 | 1.5e-3 | complete | 0.9779 | best run |
| maxlr_2e3 | 2e-3 | complete | 0.9733 | too high; below 1.5e-3 |

## Commands

```powershell
conda activate krishidoc_ml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_3e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_5e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_2e3.yaml
```

## Notes

- Test set remains locked.
- Current best to beat: exp08 macro-F1 0.9746.
- Target for clear win: >= 0.9770 macro-F1 or same score with better minority-class behavior.
- Record failed/partial/debug runs here, not in `results/experiment_log.md`.

## Completed Summary

| Variant | max_lr | Val macro-F1 | Val Acc | CPU ms | Best epoch | downy_mildew F1 | hispa F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxlr_3e4 | 3e-4 | 0.9686 | 0.9699 | 15.97 | 35 | 0.9149 | 0.9520 |
| maxlr_5e4 | 5e-4 | 0.9683 | 0.9705 | 15.67 | 30 | 0.9405 | 0.9600 |
| maxlr_1e3 | 1e-3 | 0.9749 | 0.9744 | 16.84 | 33 | 0.9508 | 0.9627 |
| maxlr_15e4 | 1.5e-3 | 0.9779 | 0.9763 | 15.02 | 31 | 0.9622 | 0.9617 |
| maxlr_2e3 | 2e-3 | 0.9733 | 0.9725 | 16.01 | 32 | 0.9565 | 0.9542 |

Conclusion: `maxlr_15e4` is the best MobileNetV3-Large run so far. It beats
exp08 by +0.0033 macro-F1 and meets the pre-set replacement threshold. Test set
was not evaluated.
