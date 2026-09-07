"""Prepare a three-class Dhan-Shomadhan deployment-evaluation manifest.

Only labels that map unambiguously to the locked Kaggle taxonomy are included.
Leaf Scald and Sheath Blight remain out of scope and are recorded separately.
The generated combined audit manifest labels Kaggle as ``train`` and Dhan as
``val`` solely so the duplicate-audit tool searches across the two sources.
It is not a training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DHAN_ROOT = ROOT / "datasets" / "Dhan-Shomadhan"
KAGGLE_SPLIT = ROOT / "data" / "splits" / "kaggle_grouped_v2.csv"
OUTPUT = ROOT / "data" / "splits" / "dhan_supported_eval.csv"
EXCLUDED_OUTPUT = ROOT / "data" / "splits" / "dhan_out_of_scope.csv"
AUDIT_MANIFEST = (
    ROOT / "results" / "data_integrity" / "dhan_kaggle_overlap_audit" / "input_manifest.csv"
)
SUMMARY_OUTPUT = (
    ROOT / "results" / "data_integrity" / "dhan_kaggle_overlap_audit" / "manifest_summary.json"
)

LABEL_MAP = {
    "Brown Spot": "brown_spot",
    "Browon Spot": "brown_spot",
    "Rice Blast": "blast",
    "Rice Tungro": "tungro",
    "Rice Turgro": "tungro",
}
OUT_OF_SCOPE = {
    "Leaf Scaled": "leaf_scald",
    "Sheath Blight": "sheath_blight",
    "Shath Blight": "sheath_blight",
}


def collect_dhan() -> tuple[pd.DataFrame, pd.DataFrame]:
    supported: list[dict] = []
    excluded: list[dict] = []
    for background_dir in sorted(path for path in DHAN_ROOT.iterdir() if path.is_dir()):
        background = background_dir.name.lower().replace(" ", "_")
        for label_dir in sorted(path for path in background_dir.iterdir() if path.is_dir()):
            if label_dir.name in LABEL_MAP:
                target = supported
                mapped_class = LABEL_MAP[label_dir.name]
                reason = ""
            elif label_dir.name in OUT_OF_SCOPE:
                target = excluded
                mapped_class = OUT_OF_SCOPE[label_dir.name]
                reason = "class_not_in_locked_kaggle_taxonomy"
            else:
                raise ValueError(f"Unmapped Dhan directory: {label_dir}")

            for image_path in sorted(label_dir.glob("*.jpg")):
                target.append(
                    {
                        "filepath": image_path.resolve().as_posix(),
                        "class": mapped_class,
                        "split": "dhan_eval" if target is supported else "excluded",
                        "background": background,
                        "source_label": label_dir.name,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(supported), pd.DataFrame(excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-split", type=Path, default=KAGGLE_SPLIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--excluded-output", type=Path, default=EXCLUDED_OUTPUT)
    parser.add_argument("--audit-manifest", type=Path, default=AUDIT_MANIFEST)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()

    supported, excluded = collect_dhan()
    if supported.empty:
        raise ValueError(f"No supported Dhan images found under {DHAN_ROOT}")
    if supported["filepath"].duplicated().any() or excluded["filepath"].duplicated().any():
        raise ValueError("Duplicate Dhan paths discovered")
    if not all(Path(path).is_file() for path in supported["filepath"]):
        raise FileNotFoundError("A supported Dhan image path does not exist")

    kaggle = pd.read_csv(args.kaggle_split)[["filepath", "class"]].copy()
    kaggle["split"] = "train"
    dhan_for_audit = supported[["filepath", "class"]].copy()
    dhan_for_audit["split"] = "val"
    audit_manifest = pd.concat([kaggle, dhan_for_audit], ignore_index=True)

    for path in (args.output, args.excluded_output, args.audit_manifest, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    supported.to_csv(args.output, index=False)
    excluded.to_csv(args.excluded_output, index=False)
    audit_manifest.to_csv(args.audit_manifest, index=False)

    summary = {
        "supported_images": int(len(supported)),
        "out_of_scope_images": int(len(excluded)),
        "supported_class_counts": supported["class"].value_counts().sort_index().to_dict(),
        "supported_background_counts": supported["background"].value_counts().sort_index().to_dict(),
        "out_of_scope_class_counts": excluded["class"].value_counts().sort_index().to_dict(),
        "audit_manifest_rows": int(len(audit_manifest)),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
