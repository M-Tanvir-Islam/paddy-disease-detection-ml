"""Build a leakage-resistant, group-aware Kaggle train/val/test split.

The accepted duplicate components produced by ``scripts/audit_split_leakage.py``
are indivisible allocation units. Components containing more than one label are
quarantined for manual review instead of being relabelled automatically.

This script never reads model predictions or evaluates the test partition.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "splits" / "kaggle.csv"
DEFAULT_GROUPS = (
    ROOT / "results" / "data_integrity" / "kaggle_split_audit" / "duplicate_groups.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "kaggle_grouped_v2.csv"
DEFAULT_QUARANTINE = ROOT / "data" / "splits" / "kaggle_grouped_v2_quarantine.csv"
DEFAULT_REPORT = (
    ROOT / "results" / "data_integrity" / "kaggle_grouped_v2_split_report.json"
)
SEED = 42
SPLIT_VERSION = "kaggle_grouped_v2"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def path_key(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve().as_posix().casefold()


def choose_groups(groups: list[dict], target: int, rng: random.Random) -> set[str]:
    """Choose whole groups with total size as close as possible to target."""
    shuffled = list(groups)
    rng.shuffle(shuffled)
    max_size = max((g["size"] for g in shuffled), default=0)
    limit = target + max_size
    possible: dict[int, tuple[str, ...]] = {0: ()}

    for group in shuffled:
        additions: dict[int, tuple[str, ...]] = {}
        for total, selected in list(possible.items()):
            new_total = total + group["size"]
            if new_total <= limit and new_total not in possible and new_total not in additions:
                additions[new_total] = selected + (group["id"],)
        possible.update(additions)

    best_total = min(
        possible,
        key=lambda total: (abs(total - target), total > target, total),
    )
    return set(possible[best_total])


def build_split(source: pd.DataFrame, duplicate_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_source = {"filepath", "class", "split"}
    required_groups = {"group_id", "cross_label_group", "path"}
    if not required_source.issubset(source.columns):
        raise ValueError(f"Source CSV must contain {sorted(required_source)}")
    if not required_groups.issubset(duplicate_rows.columns):
        raise ValueError(f"Duplicate CSV must contain {sorted(required_groups)}")
    if source["filepath"].map(path_key).duplicated().any():
        raise ValueError("Source CSV contains duplicate file paths")

    group_by_path: dict[str, str] = {}
    cross_label_groups: set[str] = set()
    for row in duplicate_rows.itertuples(index=False):
        key = path_key(row.path)
        if key in group_by_path and group_by_path[key] != row.group_id:
            raise ValueError(f"Image assigned to multiple duplicate groups: {row.path}")
        group_by_path[key] = row.group_id
        if str(row.cross_label_group).strip().lower() == "true":
            cross_label_groups.add(row.group_id)

    work = source.copy()
    work["original_split"] = work["split"]
    work["_path_key"] = work["filepath"].map(path_key)
    work["group_id"] = [
        group_by_path.get(key, f"single::{index:05d}")
        for index, key in enumerate(work["_path_key"])
    ]

    quarantine = work[work["group_id"].isin(cross_label_groups)].copy()
    quarantine["reason"] = "accepted duplicate component contains conflicting labels"
    quarantine = quarantine[
        ["filepath", "class", "original_split", "group_id", "reason"]
    ].sort_values(["group_id", "class", "filepath"])

    clean = work[~work["group_id"].isin(cross_label_groups)].copy()
    group_labels = clean.groupby("group_id")["class"].nunique()
    bad_groups = group_labels[group_labels != 1]
    if not bad_groups.empty:
        raise ValueError(f"Non-quarantined groups with conflicting labels: {bad_groups.index.tolist()}")

    assignment: dict[str, str] = {}
    for class_index, (class_name, class_rows) in enumerate(clean.groupby("class", sort=True)):
        groups = [
            {"id": group_id, "size": len(rows)}
            for group_id, rows in class_rows.groupby("group_id", sort=True)
        ]
        class_total = len(class_rows)
        rng = random.Random(SEED + class_index)

        test_target = round(class_total * RATIOS["test"])
        test_ids = choose_groups(groups, test_target, rng)
        remaining = [group for group in groups if group["id"] not in test_ids]

        val_target = round(class_total * RATIOS["val"])
        val_ids = choose_groups(remaining, val_target, rng)

        for group in groups:
            group_id = group["id"]
            assignment[group_id] = (
                "test" if group_id in test_ids else "val" if group_id in val_ids else "train"
            )

    clean["split"] = clean["group_id"].map(assignment)
    clean["split_version"] = SPLIT_VERSION
    output = clean[
        ["filepath", "class", "split", "group_id", "split_version"]
    ].sort_values(["split", "class", "group_id", "filepath"])

    if len(output) + len(quarantine) != len(source):
        raise AssertionError("Clean and quarantine rows do not reconstruct the source dataset")
    if output["filepath"].map(path_key).duplicated().any():
        raise AssertionError("Output contains duplicate paths")
    if (output.groupby("group_id")["split"].nunique() > 1).any():
        raise AssertionError("A duplicate group crosses partitions")
    if set(output["group_id"]) & cross_label_groups:
        raise AssertionError("A cross-label group escaped quarantine")

    return output.reset_index(drop=True), quarantine.reset_index(drop=True)


def make_report(source: pd.DataFrame, output: pd.DataFrame, quarantine: pd.DataFrame) -> dict:
    split_counts = output.groupby("split").size().reindex(RATIOS, fill_value=0)
    class_counts = (
        output.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=RATIOS)
    )
    return {
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "source_rows": int(len(source)),
        "usable_rows": int(len(output)),
        "quarantined_rows": int(len(quarantine)),
        "quarantined_groups": int(quarantine["group_id"].nunique()),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "split_ratios": {
            key: round(int(value) / max(len(output), 1), 6)
            for key, value in split_counts.items()
        },
        "class_split_counts": {
            class_name: {key: int(value) for key, value in row.items()}
            for class_name, row in class_counts.iterrows()
        },
        "duplicate_groups_crossing_splits": int(
            (output.groupby("group_id")["split"].nunique() > 1).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source = pd.read_csv(args.source)
    duplicate_rows = pd.read_csv(args.groups)
    output, quarantine = build_split(source, duplicate_rows)
    report = make_report(source, output, quarantine)

    for path in (args.output, args.quarantine, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    quarantine.to_csv(args.quarantine, index=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote split: {args.output}")
    print(f"Wrote quarantine: {args.quarantine}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
