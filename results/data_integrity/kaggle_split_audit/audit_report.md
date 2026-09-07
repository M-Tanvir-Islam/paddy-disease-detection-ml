# Kaggle Split Leakage Audit

This is a data-integrity audit. No model predictions or test metrics were computed.
The existing split CSV was not modified.

## Dataset

- Images audited: `10407`
- Split counts: `{'train': 7284, 'val': 1561, 'test': 1562}`
- Classes: `10`
- Image read errors: `0`

## Cross-Split Findings

- Confirmed byte-identical pairs: `34`
- Confirmed decoded-pixel-identical pairs: `34`
- Exact duplicate label conflicts: `0`
- Cross-split exact duplicate groups: `33`
- Unique images in those groups: `67`
- Validation images with a training duplicate: `15`
- Test images with a training duplicate: `14`
- Total perceptual candidates (including exact pairs): `5282`
- Geometrically supported high-confidence near-duplicate pairs: `889`
- Supported high-confidence cross-label pairs: `29`

Candidate confidence counts:

- `confirmed_exact_file`: `34`
- `likely_near_duplicate`: `533`
- `possible_near_duplicate`: `4359`
- `very_likely_near_duplicate`: `356`

Candidate split-pair counts:

- `train-test`: `2412`
- `train-val`: `2327`
- `val-test`: `543`

## Conservative Duplicate Groups

- Groups: `461`
- Unique images in groups: `1264`
- Cross-label groups: `11`
- Validation images grouped with training images: `272`
- Test images grouped with training images: `288`
- Grouping basis: Confirmed exact pairs plus high-confidence perceptual pairs with at least 12 ORB homography inliers and inlier ratio >= 0.5.

## Metadata Distribution Risk

- Metadata rows matched: `10407`
- Metadata rows missing: `0`
- Class/variety/age groups: `152`
- Groups spanning more than one split: `148`
- Interpretation: Variety and age are proxies, not capture-session identifiers. A group spanning splits is a distribution-risk flag, not proof of leakage.

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
