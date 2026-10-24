# exp18 - EfficientNet-B0 Group-Aware Retrain

## Hypothesis

The best EfficientNet-B0 recipe (exp14 `maxlr_5e4`) should remain strong on
the leakage-resistant split, and comparing it with MobileNetV3-Large exp16
under the same split gives a fair basis for choosing the final model.

## Changed Variable

Only the dataset split changes relative to exp14 `maxlr_5e4`:

- exp14: `data/splits/kaggle.csv`
- exp18: `data/splits/kaggle_grouped_v2.csv`

The model, ImageNet initialization, augmentation, weighted CrossEntropy,
OneCycleLR settings (max_lr 5e-4), batch size, epoch budget, and seed remain
fixed. This is a fresh retrain from ImageNet weights; the exp14 checkpoint is
not reused.

The branch is cut from exp14, so it keeps the EfficientNet track's own
training augmentation in `src/augment.py` (resize to 1.15x then random crop,
motion blur, and default noise/dropout settings). This differs from the
MobileNet track's pipeline (plain resize, explicit noise and dropout ranges).
The cross-track comparison is therefore between each track's best recipe,
not between architectures under identical augmentation.

The only shared-code change is forcing Matplotlib's headless `Agg` backend in
`src/evaluate.py`, the same fix applied on the exp16 branch. It affects plot
rendering only.

## Data Protocol

- 10,349 usable images
- 7,241 train / 1,554 validation / 1,554 locked test
- 58 images from 11 cross-label duplicate groups quarantined
- seed 42; the test set is not evaluated

## Result

| Metric | Value |
| --- | ---: |
| Validation macro-F1 | **0.9765** |
| Validation accuracy | 0.9775 |
| Best epoch | 34 / 40 |
| Validation errors | 35 / 1,554 |
| CPU latency, mean / p95 (logged at end of training) | 24.57 / 32.04 ms |
| GPU latency, mean / p95 | 13.18 / 13.43 ms |
| Parameters / MACs | 4.02 M / 0.414 G |
| Checkpoint size | 16.39 MB |
| Peak VRAM | 2,977 MB |
| Training time | 55.8 minutes |

The strongest classes were `bacterial_leaf_streak` and `dead_heart` (F1
1.0000). The weakest was `downy_mildew` (F1 0.9282), the same weak class as
in exp16 but lower. Validation macro-F1 plateaued between 0.970 and 0.977
over the last ten epochs.

### Cross-track comparison under the grouped split

`compare_with_exp16.py` evaluates both checkpoints back to back on the same
validation partition (`results/compare_with_exp16.json`):

| Metric (validation, 1,554 images) | exp16 MobileNetV3-Large | exp18 EfficientNet-B0 |
| --- | ---: | ---: |
| Macro-F1 | **0.9800** | 0.9765 |
| Errors | **29** | 35 |
| Expected calibration error, 15 bins | **0.0128** | 0.0147 |
| Negative log-likelihood | **0.084** | 0.101 |
| CPU latency, mean / p95 | **15.5 / 17.8 ms** | 23.2 / 26.1 ms |
| Checkpoint size | 17.07 MB | **16.39 MB** |

Per-class F1 differences were small apart from `downy_mildew` (0.9545 vs
0.9282) and `hispa` (0.9851 vs 0.9720) in MobileNet's favour, and
`brown_spot` (0.9789 vs 0.9861) and `bacterial_panicle_blight` (0.9697 vs
0.9800) in EfficientNet's favour.

The −0.0035 macro-F1 gap exceeds the ±0.003 meaningful-change threshold, and
MobileNetV3-Large is also better calibrated and about 1.5x faster on CPU.
Both comparisons use each track's own best recipe; the EfficientNet track's
augmentation differs from MobileNet's (see Changed Variable), so this ranks
recipes rather than isolating architecture alone.

## Next Step

EfficientNet-B0 does not displace MobileNetV3-Large. exp16 is the cross-track
winner and the candidate for Phase 3: one-time evaluation on the locked
`kaggle_grouped_v2` test partition, ONNX export and verification, and the
model card.
