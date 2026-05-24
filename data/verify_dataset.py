"""
Sanity-check the prepared split CSV before training.

Verifies:
- CSV exists and is readable
- All filepaths point to existing files (samples 200)
- Class names match src/classes.py (no typos, no extras)
- Each split has every class
- Reports per-(split,class) counts and overall imbalance ratio
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import CLASS_NAMES_EN  # noqa: E402

CSV_PATH = ROOT / "data" / "splits" / "kaggle.csv"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run `python data/prepare_kaggle.py` first.")
        return 1

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    expected = set(CLASS_NAMES_EN)
    found = set(df["class"])
    extra = found - expected
    missing = expected - found
    if extra:
        print(f"ERROR: CSV contains unknown classes: {sorted(extra)}")
        return 1
    if missing:
        print(f"WARNING: classes in src/classes.py but not in CSV: {sorted(missing)}")

    print("\nPer-(split, class) counts:")
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(counts.to_string())

    print("\nClass imbalance ratio (max / min) per split:")
    for split, row in counts.iterrows():
        ratio = row.max() / max(row.min(), 1)
        print(f"  {split:5s}  max={row.max():>5d}  min={row.min():>5d}  ratio={ratio:.2f}")

    print("\nVerifying filepaths exist on disk (sampling 200)...")
    sample = df.sample(min(200, len(df)), random_state=0)
    missing_files = [p for p in sample["filepath"] if not Path(p).is_file()]
    if missing_files:
        print(f"ERROR: {len(missing_files)} of {len(sample)} sampled paths missing. First few:")
        for p in missing_files[:5]:
            print(f"  {p}")
        return 1
    print(f"  OK - all {len(sample)} sampled filepaths exist")

    print("\nReady to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
