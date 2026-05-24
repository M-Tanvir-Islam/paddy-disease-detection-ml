"""
Locked class taxonomy for paddy disease classification.

Currently: Kaggle Paddy 2022 — 10 classes.

When Dhan-Shomadhan is added in a later experiment, APPEND new classes
(`leaf_scald`, `sheath_blight`) to keep existing class indices stable —
the API contract in the product repo depends on these indices.

This file is the source of truth for class indices across all experiments.
Copy to `krishidoc/services/api/core/classes.py` during integration.

Bangla display names are intentionally not stored here. The product repo
maintains its own localized label map keyed by the English class name —
keeps the ML repo free of translation concerns.
"""

CLASS_NAMES_EN: list[str] = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

NUM_CLASSES: int = len(CLASS_NAMES_EN)
INPUT_SIZE: int = 224
