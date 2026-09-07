# Dhan–Kaggle overlap audit review

The manifest compares all 10,349 usable `kaggle_grouped_v2` images against
the 606 Dhan-Shomadhan images whose labels map to the locked model taxonomy.

- Exact file matches: 0
- Exact decoded-pixel matches: 0
- Accepted geometrically supported near-duplicate matches: 0
- Broad perceptual-hash candidates: 35

All 35 broad candidates are low-confidence `possible_near_duplicate` matches.
None met the audit's high-confidence threshold for geometric verification, so
none are accepted as overlap. The Dhan evaluation set is independent under the
same conservative acceptance policy used for the grouped Kaggle split.

This audit contains no model predictions and does not access the locked Kaggle
test metrics.
