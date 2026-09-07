# Kaggle Split Leakage Audit - Review Notes

## Scope

This review covers the integrity of `data/splits/kaggle.csv`. It does not use a
trained model, calculate predictions, or evaluate validation/test performance.
The split CSV and source images remain unchanged.

Visual review covered:

- The 100 highest-priority cross-split pairs.
- All 29 geometrically supported high-confidence cross-label pairs.
- The largest connected duplicate groups and their split/class composition.

## Confirmed Findings

- 34 cross-split pairs are byte-identical and decoded-pixel-identical.
- Those exact matches form 33 groups containing 67 unique images.
- 15 validation images and 14 test images have an exact duplicate in training.
- Exact duplicate labels are consistent; no exact-match label conflict was
  found.
- No image read failures occurred.

## High-Confidence Near-Duplicate Findings

Perceptual hashing produced a deliberately broad set of 5,282 candidates. The
audit then restricted geometric verification to the 889 strongest pHash/dHash
candidates. All 889 have at least 18 ORB homography inliers and an inlier ratio
of at least 0.5455. Median support is 283 inliers at a 0.9689 inlier ratio.

Visual inspection confirms that the highest-ranked pairs are predominantly
repeat photographs or minimally changed versions of the same plant and scene,
not merely unrelated leaves with similar color.

Using exact matches plus geometrically supported high-confidence pairs gives a
conservative grouping result:

| Measure | Result |
| --- | ---: |
| Connected duplicate groups | 461 |
| Unique images in groups | 1,264 (12.1% of the dataset) |
| Validation images grouped with training | 272 / 1,561 (17.4%) |
| Test images grouped with training | 288 / 1,562 (18.4%) |
| Cross-label groups | 11 |
| Cross-label supported pairs | 29 |

Cross-label repeat-scene patterns include:

- `blast` and `downy_mildew`
- `brown_spot` and `bacterial_leaf_streak`
- `normal` and `hispa`
- `bacterial_leaf_streak` and `downy_mildew`
- additional isolated combinations listed in `candidate_pairs.csv`

These cross-label groups require dataset review. They may represent inconsistent
annotation, multiple symptoms in one scene, or a capture-sequence labeling
problem. They must not be automatically relabeled by this audit.

## Metadata Finding

The original split is image-level stratified random sampling. Of 152
`class + variety + age` groups, 148 span multiple splits. Variety and age are
only weak proxies for acquisition groups, so this is not proof of leakage by
itself. It does confirm that the current split was not designed to hold out
varieties, ages, farms, plants, or capture sessions.

## Interpretation

Risk level: **high for absolute performance estimation**.

The current validation results remain useful for historical comparisons because
all experiments used the same protocol. However, the absolute `0.9779`
MobileNet validation macro-F1 is likely optimistic for unseen plants and field
conditions. The current locked test split is also compromised as a final
unbiased estimate because exact and near-duplicate groups cross into training.

The test set was not evaluated during this audit, so no model-selection leakage
was introduced through predictions or metrics.

## Recommended Remediation

1. Preserve `data/splits/kaggle.csv` as the version-1 historical split.
2. Review and resolve the 11 cross-label groups before changing labels or
   generating a replacement split.
3. Create a versioned group-aware split CSV, keeping every accepted duplicate
   group entirely within one split.
4. Prefer true plant/capture-session grouping if source metadata can provide it;
   otherwise use the conservative connected groups from `duplicate_groups.csv`.
5. Recheck class balance and verify zero exact/group leakage in the new split.
6. Retrain the selected MobileNet exp09 recipe and EfficientNet exp14 reference
   on the corrected split before comparing them.
7. Reserve the corrected test partition until the new winner is frozen.

Do not delete source images, silently replace the historical split, or compare
new-split metrics directly against old-split metrics as if the protocols were
identical.
