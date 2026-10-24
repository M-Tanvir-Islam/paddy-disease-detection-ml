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

Pending.

## Next Step

Pending.
