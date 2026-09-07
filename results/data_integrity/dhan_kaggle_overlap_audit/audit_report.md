# Kaggle Split Leakage Audit

This is a data-integrity audit. No model predictions or test metrics were computed.
The existing split CSV was not modified.

## Dataset

- Images audited: `10955`
- Split counts: `{'train': 10349, 'val': 606}`
- Classes: `10`
- Image read errors: `0`

## Cross-Split Findings

- Confirmed byte-identical pairs: `0`
- Confirmed decoded-pixel-identical pairs: `0`
- Exact duplicate label conflicts: `0`
- Cross-split exact duplicate groups: `0`
- Unique images in those groups: `0`
- Validation images with a training duplicate: `0`
- Test images with a training duplicate: `0`
- Total perceptual candidates (including exact pairs): `35`
- Geometrically supported high-confidence near-duplicate pairs: `0`
- Supported high-confidence cross-label pairs: `0`

Candidate confidence counts:

- `possible_near_duplicate`: `35`

Candidate split-pair counts:

- `train-val`: `35`

## Conservative Duplicate Groups

- Groups: `0`
- Unique images in groups: `0`
- Cross-label groups: `0`
- Validation images grouped with training images: `0`
- Test images grouped with training images: `0`
- Grouping basis: Confirmed exact pairs plus high-confidence perceptual pairs with at least 12 ORB homography inliers and inlier ratio >= 0.5.

## Metadata Distribution Risk

Metadata audit unavailable: metadata CSV not found

## Review Guidance

- `confirmed_exact_file` and `confirmed_exact_pixels` are definitive duplicates.
- Perceptual candidates require visual review; similar disease symptoms are not automatically duplicates.
- Prioritize cross-label candidates and train-to-test candidates.
- Do not change the split until candidate pairs have been reviewed and grouped.

## Outputs

- `candidate_pairs.csv`: ranked cross-split exact and perceptual candidates.
- `hash_manifest.csv`: reproducible fingerprints for every audited image.
- `metadata_groups.csv`: class/variety/age distribution across splits.
- `duplicate_groups.csv`: conservative exact/near-duplicate group membership.
- `audit_summary.json`: machine-readable summary and parameters.
- `contact_sheets/`: visual review sheets for the highest-priority pairs.
