"""
Build a stratified train/val/test split for the Kaggle Paddy 2022 dataset.

Reads:  datasets/paddy-disease-classification/train_images/<class>/*.jpg
Writes: data/splits/kaggle.csv   columns: filepath, class, split

Splits: 70% train / 15% val / 15% test, stratified by class, seed=42.
No file copying — the CSV references images at their original paths.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import CLASS_NAMES_EN  # noqa: E402

DATASET_ROOT = ROOT / "datasets" / "paddy-disease-classification" / "train_images"
OUT_CSV = ROOT / "data" / "splits" / "kaggle.csv"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test = 1 - train - val


def collect_samples() -> pd.DataFrame:
    rows = []
    missing = []
    for cls in CLASS_NAMES_EN:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.is_dir():
            missing.append(cls)
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({
                    "filepath": img_path.as_posix(),
                    "class": cls,
                })
    if missing:
        print(f"WARNING: missing class folders: {missing}")
    return pd.DataFrame(rows)


def main() -> None:
    df = collect_samples()
    print(f"\nTotal images discovered: {len(df)}")
    print("\nPer-class counts:")
    print(df["class"].value_counts().sort_index().to_string())

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=df["class"],
        random_state=SEED,
    )
    val_size_of_temp = VAL_RATIO / (1.0 - TRAIN_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_of_temp),
        stratify=temp_df["class"],
        random_state=SEED,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(out)} rows -> {OUT_CSV}")
    print("\nSplit x class counts:")
    print(out.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
