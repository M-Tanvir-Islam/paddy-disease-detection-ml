"""Audit a dataset split for exact and near-duplicate image leakage.

This script performs data-integrity analysis only. It never loads a model,
computes predictions, or evaluates validation/test metrics.

Checks:
  1. Byte-identical files (SHA-256).
  2. Identical decoded RGB pixels, despite file encoding differences.
  3. Perceptually similar cross-split candidates using pHash and dHash.
  4. Conflicting labels among exact duplicates.
  5. Metadata groups (for example variety + age) spanning splits.

Usage from the repository root:
    python scripts/audit_split_leakage.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CSV = ROOT / "data" / "splits" / "kaggle.csv"
DEFAULT_METADATA_CSV = (
    ROOT / "datasets" / "paddy-disease-classification" / "train.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "data_integrity" / "kaggle_split_audit"
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class ImageRecord:
    index: int
    filepath: str
    display_path: str
    split: str
    class_name: str
    image_id: str


@dataclass(frozen=True)
class ImageFingerprint:
    index: int
    width: int
    height: int
    file_sha256: str
    pixel_sha256: str
    phash: int
    dhash: int
    error: str = ""


class BKNode:
    """Node for exact-radius search in Hamming space."""

    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    """Small dependency-free BK-tree for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return

        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []

        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                matches.extend(node.indices)
            lower = distance - radius
            upper = distance + radius
            stack.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--dhash-threshold", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--contact-sheet-pairs",
        type=int,
        default=100,
        help="Number of highest-priority candidate pairs to render.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic row limit for smoke testing only.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def numeric_image_id(image_id: str) -> int | None:
    match = re.search(r"(\d+)", Path(image_id).stem)
    return int(match.group(1)) if match else None


def load_records(split_csv: Path, limit: int | None) -> list[ImageRecord]:
    frame = pd.read_csv(split_csv)
    required = {"filepath", "class", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")

    invalid_splits = set(frame["split"].unique()) - set(SPLIT_ORDER)
    if invalid_splits:
        raise ValueError(f"Unsupported split values: {sorted(invalid_splits)}")

    if limit is not None:
        frame = frame.head(limit)

    records: list[ImageRecord] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        path = Path(str(row["filepath"]))
        records.append(
            ImageRecord(
                index=index,
                filepath=str(path),
                display_path=repo_relative(path),
                split=str(row["split"]),
                class_name=str(row["class"]),
                image_id=path.name,
            )
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dct_basis(low_frequency_size: int = 8, image_size: int = 32) -> np.ndarray:
    x = np.arange(image_size, dtype=np.float64)
    basis = np.empty((low_frequency_size, image_size), dtype=np.float64)
    for frequency in range(low_frequency_size):
        alpha = math.sqrt(1 / image_size) if frequency == 0 else math.sqrt(2 / image_size)
        basis[frequency] = alpha * np.cos(
            math.pi * (2 * x + 1) * frequency / (2 * image_size)
        )
    return basis


DCT_BASIS = dct_basis()


def bits_to_uint64(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def perceptual_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    coefficients = DCT_BASIS @ pixels @ DCT_BASIS.T
    flattened = coefficients.reshape(-1)
    median = np.median(flattened[1:])
    return bits_to_uint64(coefficients > median)


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return bits_to_uint64(pixels[:, 1:] > pixels[:, :-1])


def fingerprint(record: ImageRecord) -> ImageFingerprint:
    path = Path(record.filepath)
    try:
        file_hash = sha256_file(path)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            pixels = np.ascontiguousarray(np.asarray(image))
            pixel_digest = hashlib.sha256()
            pixel_digest.update(f"{image.width}x{image.height}:RGB".encode("ascii"))
            pixel_digest.update(pixels.tobytes())
            return ImageFingerprint(
                index=record.index,
                width=image.width,
                height=image.height,
                file_sha256=file_hash,
                pixel_sha256=pixel_digest.hexdigest(),
                phash=perceptual_hash(image),
                dhash=difference_hash(image),
            )
    except Exception as exc:  # keep the complete audit running and report failures
        return ImageFingerprint(
            index=record.index,
            width=0,
            height=0,
            file_sha256="",
            pixel_sha256="",
            phash=0,
            dhash=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def cross_split_pairs(indices: Iterable[int], records: list[ImageRecord]) -> set[tuple[int, int]]:
    ordered = sorted(indices)
    return {
        (left, right)
        for position, left in enumerate(ordered)
        for right in ordered[position + 1 :]
        if records[left].split != records[right].split
    }


def grouped_exact_pairs(
    fingerprints: list[ImageFingerprint],
    records: list[ImageRecord],
    attribute: str,
) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for item in fingerprints:
        value = getattr(item, attribute)
        if value and not item.error:
            groups[value].append(item.index)

    pairs: set[tuple[int, int]] = set()
    for indices in groups.values():
        if len(indices) > 1:
            pairs.update(cross_split_pairs(indices, records))
    return pairs


def exact_group_stats(
    fingerprints: list[ImageFingerprint],
    records: list[ImageRecord],
    attribute: str,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for item in fingerprints:
        value = getattr(item, attribute)
        if value and not item.error:
            groups[value].append(item.index)

    cross_split_groups = [
        indices
        for indices in groups.values()
        if len(indices) > 1
        and len({records[index].split for index in indices}) > 1
    ]
    group_sizes = Counter(len(indices) for indices in cross_split_groups)
    unique_by_split = Counter(
        records[index].split
        for indices in cross_split_groups
        for index in indices
    )
    groups_with_train = [
        indices
        for indices in cross_split_groups
        if "train" in {records[index].split for index in indices}
    ]
    return {
        "cross_split_groups": len(cross_split_groups),
        "unique_images_in_groups": sum(len(indices) for indices in cross_split_groups),
        "group_sizes": {str(size): count for size, count in sorted(group_sizes.items())},
        "unique_images_by_split": dict(sorted(unique_by_split.items())),
        "val_images_with_train_duplicate": sum(
            records[index].split == "val"
            for indices in groups_with_train
            for index in indices
        ),
        "test_images_with_train_duplicate": sum(
            records[index].split == "test"
            for indices in groups_with_train
            for index in indices
        ),
    }


def near_duplicate_pairs(
    fingerprints: list[ImageFingerprint],
    records: list[ImageRecord],
    phash_threshold: int,
    dhash_threshold: int,
) -> set[tuple[int, int]]:
    phash_tree = BKTree()
    dhash_tree = BKTree()
    candidates: set[tuple[int, int]] = set()

    for item in tqdm(fingerprints, desc="Searching perceptual hashes"):
        if item.error:
            continue

        prior_indices = set(phash_tree.query(item.phash, phash_threshold))
        prior_indices.update(dhash_tree.query(item.dhash, dhash_threshold))
        for prior in prior_indices:
            if records[prior].split != records[item.index].split:
                candidates.add((min(prior, item.index), max(prior, item.index)))

        phash_tree.add(item.phash, item.index)
        dhash_tree.add(item.dhash, item.index)

    return candidates


def confidence_tier(
    same_file: bool,
    same_pixels: bool,
    phash_distance: int,
    dhash_distance: int,
) -> str:
    if same_file:
        return "confirmed_exact_file"
    if same_pixels:
        return "confirmed_exact_pixels"
    if (phash_distance <= 2 and dhash_distance <= 4) or (
        dhash_distance <= 2 and phash_distance <= 4
    ):
        return "very_likely_near_duplicate"
    if phash_distance <= 4 and dhash_distance <= 6:
        return "likely_near_duplicate"
    return "possible_near_duplicate"


def build_candidate_rows(
    pairs: set[tuple[int, int]],
    records: list[ImageRecord],
    fingerprints: list[ImageFingerprint],
) -> list[dict]:
    by_index = {item.index: item for item in fingerprints}
    rows: list[dict] = []
    for left_index, right_index in pairs:
        left_record = records[left_index]
        right_record = records[right_index]
        left_hash = by_index[left_index]
        right_hash = by_index[right_index]

        same_file = left_hash.file_sha256 == right_hash.file_sha256
        same_pixels = left_hash.pixel_sha256 == right_hash.pixel_sha256
        phash_distance = BKTree.distance(left_hash.phash, right_hash.phash)
        dhash_distance = BKTree.distance(left_hash.dhash, right_hash.dhash)
        left_numeric_id = numeric_image_id(left_record.image_id)
        right_numeric_id = numeric_image_id(right_record.image_id)
        id_delta = (
            abs(left_numeric_id - right_numeric_id)
            if left_numeric_id is not None and right_numeric_id is not None
            else None
        )

        rows.append(
            {
                "confidence": confidence_tier(
                    same_file, same_pixels, phash_distance, dhash_distance
                ),
                "split_a": left_record.split,
                "split_b": right_record.split,
                "class_a": left_record.class_name,
                "class_b": right_record.class_name,
                "same_class": left_record.class_name == right_record.class_name,
                "path_a": left_record.display_path,
                "path_b": right_record.display_path,
                "image_id_a": left_record.image_id,
                "image_id_b": right_record.image_id,
                "image_id_delta": id_delta,
                "width_a": left_hash.width,
                "height_a": left_hash.height,
                "width_b": right_hash.width,
                "height_b": right_hash.height,
                "same_file_sha256": same_file,
                "same_decoded_pixels": same_pixels,
                "phash_distance": phash_distance,
                "dhash_distance": dhash_distance,
                "orb_good_matches": "",
                "orb_homography_inliers": "",
                "orb_inlier_ratio": "",
                "geometric_support": "confirmed_exact" if same_file or same_pixels else "not_scored",
                "_index_a": left_index,
                "_index_b": right_index,
            }
        )

    tier_order = {
        "confirmed_exact_file": 0,
        "confirmed_exact_pixels": 1,
        "very_likely_near_duplicate": 2,
        "likely_near_duplicate": 3,
        "possible_near_duplicate": 4,
    }
    rows.sort(
        key=lambda row: (
            tier_order[row["confidence"]],
            row["phash_distance"] + row["dhash_distance"],
            row["phash_distance"],
            row["dhash_distance"],
            row["path_a"],
            row["path_b"],
        )
    )
    return rows


def add_geometric_verification(
    rows: list[dict], records: list[ImageRecord]
) -> None:
    """Verify high-confidence pHash/dHash candidates with ORB + homography."""
    orb = cv2.ORB_create(nfeatures=750)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache: dict[int, tuple[list, np.ndarray | None]] = {}

    def features(index: int) -> tuple[list, np.ndarray | None]:
        if index not in cache:
            image = cv2.imread(records[index].filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                cache[index] = ([], None)
            else:
                cache[index] = orb.detectAndCompute(image, None)
        return cache[index]

    eligible = [
        row
        for row in rows
        if row["confidence"]
        in {"very_likely_near_duplicate", "likely_near_duplicate"}
    ]
    for row in tqdm(eligible, desc="Geometrically verifying candidates"):
        keypoints_a, descriptors_a = features(row["_index_a"])
        keypoints_b, descriptors_b = features(row["_index_b"])
        if descriptors_a is None or descriptors_b is None:
            row["geometric_support"] = "insufficient_features"
            continue

        nearest = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = [
            first
            for pair in nearest
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        row["orb_good_matches"] = len(good_matches)
        if len(good_matches) < 4:
            row["orb_homography_inliers"] = 0
            row["orb_inlier_ratio"] = 0.0
            row["geometric_support"] = "not_supported"
            continue

        points_a = np.float32(
            [keypoints_a[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        points_b = np.float32(
            [keypoints_b[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        _, inlier_mask = cv2.findHomography(
            points_a, points_b, cv2.RANSAC, ransacReprojThreshold=5.0
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = inliers / len(good_matches)
        row["orb_homography_inliers"] = inliers
        row["orb_inlier_ratio"] = round(inlier_ratio, 4)
        row["geometric_support"] = (
            "supported"
            if inliers >= 12 and inlier_ratio >= 0.5
            else "not_supported"
        )


def build_duplicate_groups(
    rows: list[dict], records: list[ImageRecord]
) -> tuple[dict, list[dict]]:
    """Build conservative connected groups from exact or ORB-supported pairs."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    accepted_rows = [
        row
        for row in rows
        if row["geometric_support"] in {"confirmed_exact", "supported"}
    ]
    for row in accepted_rows:
        union(row["_index_a"], row["_index_b"])

    components: dict[int, set[int]] = defaultdict(set)
    for row in accepted_rows:
        components[find(row["_index_a"])].update(
            {row["_index_a"], row["_index_b"]}
        )

    ordered_components = sorted(
        components.values(), key=lambda indices: (min(indices), len(indices))
    )
    member_rows: list[dict] = []
    cross_label_groups = 0
    groups_with_exact = 0
    images_by_split: Counter = Counter()
    val_with_train = 0
    test_with_train = 0
    group_sizes: Counter = Counter()

    for number, indices in enumerate(ordered_components, start=1):
        group_id = f"dup_{number:04d}"
        splits = {records[index].split for index in indices}
        classes = {records[index].class_name for index in indices}
        related_pairs = [
            row
            for row in accepted_rows
            if row["_index_a"] in indices and row["_index_b"] in indices
        ]
        has_exact = any(
            row["geometric_support"] == "confirmed_exact"
            for row in related_pairs
        )
        cross_label = len(classes) > 1
        cross_label_groups += int(cross_label)
        groups_with_exact += int(has_exact)
        group_sizes[len(indices)] += 1

        for index in sorted(indices):
            record = records[index]
            images_by_split[record.split] += 1
            if "train" in splits and record.split == "val":
                val_with_train += 1
            if "train" in splits and record.split == "test":
                test_with_train += 1
            member_rows.append(
                {
                    "group_id": group_id,
                    "group_size": len(indices),
                    "has_exact_pair": has_exact,
                    "cross_label_group": cross_label,
                    "splits_in_group": ";".join(
                        sorted(splits, key=lambda split: SPLIT_ORDER[split])
                    ),
                    "classes_in_group": ";".join(sorted(classes)),
                    "split": record.split,
                    "class": record.class_name,
                    "image_id": record.image_id,
                    "path": record.display_path,
                }
            )

    summary = {
        "groups": len(ordered_components),
        "groups_with_exact_pair": groups_with_exact,
        "cross_label_groups": cross_label_groups,
        "unique_images_in_groups": len(member_rows),
        "group_sizes": {str(size): count for size, count in sorted(group_sizes.items())},
        "unique_images_by_split": dict(sorted(images_by_split.items())),
        "val_images_grouped_with_train": val_with_train,
        "test_images_grouped_with_train": test_with_train,
        "basis": (
            "Confirmed exact pairs plus high-confidence perceptual pairs with "
            "at least 12 ORB homography inliers and inlier ratio >= 0.5."
        ),
    }
    return summary, member_rows


def metadata_summary(
    records: list[ImageRecord], metadata_csv: Path
) -> tuple[dict, list[dict]]:
    if not metadata_csv.exists():
        return {"available": False, "reason": "metadata CSV not found"}, []

    metadata = pd.read_csv(metadata_csv)
    required = {"image_id", "variety", "age"}
    missing = required - set(metadata.columns)
    if missing:
        return {
            "available": False,
            "reason": f"metadata columns missing: {sorted(missing)}",
        }, []

    split_frame = pd.DataFrame(
        {
            "image_id": [record.image_id for record in records],
            "split": [record.split for record in records],
            "class": [record.class_name for record in records],
        }
    )
    merged = split_frame.merge(metadata, on="image_id", how="left")
    missing_rows = int(merged["variety"].isna().sum())
    groups: list[dict] = []

    for (class_name, variety, age), group in merged.dropna(
        subset=["variety", "age"]
    ).groupby(["class", "variety", "age"], dropna=False):
        counts = group["split"].value_counts().to_dict()
        groups.append(
            {
                "class": str(class_name),
                "variety": str(variety),
                "age": str(age),
                "total": int(len(group)),
                "train": int(counts.get("train", 0)),
                "val": int(counts.get("val", 0)),
                "test": int(counts.get("test", 0)),
                "spans_multiple_splits": len(counts) > 1,
            }
        )

    groups.sort(key=lambda row: (-row["total"], row["class"], row["variety"], row["age"]))
    spanning = sum(group["spans_multiple_splits"] for group in groups)
    summary = {
        "available": True,
        "matched_rows": int(len(merged) - missing_rows),
        "missing_metadata_rows": missing_rows,
        "group_definition": ["class", "variety", "age"],
        "total_groups": len(groups),
        "groups_spanning_multiple_splits": spanning,
        "note": (
            "Variety and age are proxies, not capture-session identifiers. "
            "A group spanning splits is a distribution-risk flag, not proof of leakage."
        ),
    }
    return summary, groups


def save_csv(path: Path, rows: list[dict], excluded_keys: set[str] | None = None) -> None:
    excluded_keys = excluded_keys or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [key for key in rows[0] if key not in excluded_keys]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = ImageOps.exif_transpose(image).convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    left = (width - copy.width) // 2
    top = (height - copy.height) // 2
    canvas.paste(copy, (left, top))
    return canvas


def render_contact_sheets(
    rows: list[dict],
    records: list[ImageRecord],
    output_dir: Path,
    pair_limit: int,
    prefix: str = "candidates",
) -> list[str]:
    if pair_limit <= 0 or not rows:
        return []

    selected = rows[:pair_limit]
    rows_per_sheet = 10
    sheet_paths: list[str] = []
    font = ImageFont.load_default()
    tile_width = 360
    image_height = 220
    row_height = 290

    for sheet_number, offset in enumerate(range(0, len(selected), rows_per_sheet), start=1):
        batch = selected[offset : offset + rows_per_sheet]
        sheet = Image.new("RGB", (tile_width * 2, row_height * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)

        for row_number, candidate in enumerate(batch):
            top = row_number * row_height
            left_record = records[candidate["_index_a"]]
            right_record = records[candidate["_index_b"]]
            try:
                with Image.open(left_record.filepath) as image_a:
                    sheet.paste(fit_image(image_a, tile_width, image_height), (0, top))
                with Image.open(right_record.filepath) as image_b:
                    sheet.paste(
                        fit_image(image_b, tile_width, image_height),
                        (tile_width, top),
                    )
            except Exception as exc:
                draw.text((5, top + 5), f"Render error: {exc}", fill="red", font=font)

            label = (
                f"#{offset + row_number + 1} {candidate['confidence']} | "
                f"p={candidate['phash_distance']} d={candidate['dhash_distance']} | "
                f"{candidate['split_a']}:{candidate['class_a']} <> "
                f"{candidate['split_b']}:{candidate['class_b']}"
            )
            draw.rectangle(
                (0, top + image_height, tile_width * 2, top + row_height),
                fill="white",
            )
            draw.text((6, top + image_height + 6), label, fill="black", font=font)
            draw.text(
                (6, top + image_height + 24),
                f"{candidate['image_id_a']} <> {candidate['image_id_b']}",
                fill="black",
                font=font,
            )

        path = output_dir / "contact_sheets" / f"{prefix}_{sheet_number:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=92)
        sheet_paths.append(repo_relative(path))

    return sheet_paths


def markdown_report(summary: dict) -> str:
    tier_counts = summary["candidate_counts_by_confidence"]
    split_pair_counts = summary["candidate_counts_by_split_pair"]
    metadata = summary["metadata_audit"]
    exact_groups = summary["exact_file_group_stats"]
    conservative_groups = summary["conservative_duplicate_group_stats"]
    errors = summary["read_errors"]

    lines = [
        "# Kaggle Split Leakage Audit",
        "",
        "This is a data-integrity audit. No model predictions or test metrics were computed.",
        "The existing split CSV was not modified.",
        "",
        "## Dataset",
        "",
        f"- Images audited: `{summary['images_audited']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Classes: `{summary['class_count']}`",
        f"- Image read errors: `{len(errors)}`",
        "",
        "## Cross-Split Findings",
        "",
        f"- Confirmed byte-identical pairs: `{summary['confirmed_exact_file_pairs']}`",
        f"- Confirmed decoded-pixel-identical pairs: `{summary['confirmed_exact_pixel_pairs']}`",
        f"- Exact duplicate label conflicts: `{summary['exact_duplicate_label_conflicts']}`",
        f"- Cross-split exact duplicate groups: `{exact_groups['cross_split_groups']}`",
        f"- Unique images in those groups: `{exact_groups['unique_images_in_groups']}`",
        "- Validation images with a training duplicate: "
        f"`{exact_groups['val_images_with_train_duplicate']}`",
        "- Test images with a training duplicate: "
        f"`{exact_groups['test_images_with_train_duplicate']}`",
        f"- Total perceptual candidates (including exact pairs): `{summary['total_candidate_pairs']}`",
        "- Geometrically supported high-confidence near-duplicate pairs: "
        f"`{summary['geometrically_supported_near_duplicate_pairs']}`",
        "- Supported high-confidence cross-label pairs: "
        f"`{summary['geometrically_supported_cross_label_pairs']}`",
        "",
        "Candidate confidence counts:",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in tier_counts.items())
    lines.extend(["", "Candidate split-pair counts:", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in split_pair_counts.items())

    lines.extend(
        [
            "",
            "## Conservative Duplicate Groups",
            "",
            f"- Groups: `{conservative_groups['groups']}`",
            f"- Unique images in groups: `{conservative_groups['unique_images_in_groups']}`",
            f"- Cross-label groups: `{conservative_groups['cross_label_groups']}`",
            "- Validation images grouped with training images: "
            f"`{conservative_groups['val_images_grouped_with_train']}`",
            "- Test images grouped with training images: "
            f"`{conservative_groups['test_images_grouped_with_train']}`",
            f"- Grouping basis: {conservative_groups['basis']}",
        ]
    )

    lines.extend(["", "## Metadata Distribution Risk", ""])
    if metadata.get("available"):
        lines.extend(
            [
                f"- Metadata rows matched: `{metadata['matched_rows']}`",
                f"- Metadata rows missing: `{metadata['missing_metadata_rows']}`",
                f"- Class/variety/age groups: `{metadata['total_groups']}`",
                "- Groups spanning more than one split: "
                f"`{metadata['groups_spanning_multiple_splits']}`",
                f"- Interpretation: {metadata['note']}",
            ]
        )
    else:
        lines.append(f"Metadata audit unavailable: {metadata.get('reason', 'unknown reason')}")

    lines.extend(
        [
            "",
            "## Review Guidance",
            "",
            "- `confirmed_exact_file` and `confirmed_exact_pixels` are definitive duplicates.",
            "- Perceptual candidates require visual review; similar disease symptoms are not automatically duplicates.",
            "- Prioritize cross-label candidates and train-to-test candidates.",
            "- Do not change the split until candidate pairs have been reviewed and grouped.",
            "",
            "## Outputs",
            "",
            "- `candidate_pairs.csv`: ranked cross-split exact and perceptual candidates.",
            "- `hash_manifest.csv`: reproducible fingerprints for every audited image.",
            "- `metadata_groups.csv`: class/variety/age distribution across splits.",
            "- `duplicate_groups.csv`: conservative exact/near-duplicate group membership.",
            "- `audit_summary.json`: machine-readable summary and parameters.",
            "- `contact_sheets/`: visual review sheets for the highest-priority pairs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    split_csv = args.split_csv.resolve()
    metadata_csv = args.metadata_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not 0 <= args.phash_threshold <= 64:
        raise ValueError("--phash-threshold must be between 0 and 64")
    if not 0 <= args.dhash_threshold <= 64:
        raise ValueError("--dhash-threshold must be between 0 and 64")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    records = load_records(split_csv, args.limit)
    if not records:
        raise ValueError("No image records found")

    missing_paths = [record.display_path for record in records if not Path(record.filepath).is_file()]
    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} image paths do not exist. First entries:\n{preview}"
        )

    print(f"Auditing {len(records)} images from {split_csv}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        fingerprints = list(
            tqdm(
                executor.map(fingerprint, records),
                total=len(records),
                desc="Fingerprinting images",
            )
        )

    exact_file_pairs = grouped_exact_pairs(
        fingerprints, records, attribute="file_sha256"
    )
    exact_pixel_pairs = grouped_exact_pairs(
        fingerprints, records, attribute="pixel_sha256"
    )
    candidate_pairs = near_duplicate_pairs(
        fingerprints,
        records,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    candidate_pairs.update(exact_file_pairs)
    candidate_pairs.update(exact_pixel_pairs)

    candidate_rows = build_candidate_rows(candidate_pairs, records, fingerprints)
    cv2.setRNGSeed(42)
    add_geometric_verification(candidate_rows, records)
    metadata_audit, metadata_groups = metadata_summary(records, metadata_csv)
    read_errors = [
        {
            "path": records[item.index].display_path,
            "split": records[item.index].split,
            "class": records[item.index].class_name,
            "error": item.error,
        }
        for item in fingerprints
        if item.error
    ]

    tier_counts = Counter(row["confidence"] for row in candidate_rows)
    split_pair_counts = Counter(
        "-".join(
            sorted(
                (row["split_a"], row["split_b"]),
                key=lambda split: SPLIT_ORDER[split],
            )
        )
        for row in candidate_rows
    )
    exact_label_conflicts = sum(
        (row["same_file_sha256"] or row["same_decoded_pixels"])
        and not row["same_class"]
        for row in candidate_rows
    )
    geometrically_supported = [
        row
        for row in candidate_rows
        if row["geometric_support"] == "supported"
    ]
    supported_cross_label = [
        row for row in geometrically_supported if not row["same_class"]
    ]
    conservative_group_stats, duplicate_group_rows = build_duplicate_groups(
        candidate_rows, records
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_dir = output_dir / "contact_sheets"
    if contact_sheet_dir.exists():
        for stale_sheet in contact_sheet_dir.glob("*.jpg"):
            stale_sheet.unlink()
    ranked_contact_sheets = render_contact_sheets(
        candidate_rows,
        records,
        output_dir,
        pair_limit=args.contact_sheet_pairs,
    )
    cross_label_contact_sheets = render_contact_sheets(
        supported_cross_label,
        records,
        output_dir,
        pair_limit=len(supported_cross_label),
        prefix="cross_label_high_confidence",
    )

    manifest_rows = []
    for record, item in zip(records, fingerprints, strict=True):
        manifest_rows.append(
            {
                "path": record.display_path,
                "split": record.split,
                "class": record.class_name,
                "image_id": record.image_id,
                "width": item.width,
                "height": item.height,
                "file_sha256": item.file_sha256,
                "pixel_sha256": item.pixel_sha256,
                "phash": f"{item.phash:016x}" if not item.error else "",
                "dhash": f"{item.dhash:016x}" if not item.error else "",
                "error": item.error,
            }
        )

    save_csv(
        output_dir / "candidate_pairs.csv",
        candidate_rows,
        excluded_keys={"_index_a", "_index_b"},
    )
    save_csv(output_dir / "hash_manifest.csv", manifest_rows)
    save_csv(output_dir / "metadata_groups.csv", metadata_groups)
    save_csv(output_dir / "duplicate_groups.csv", duplicate_group_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "data_integrity_only_no_model_evaluation",
        "split_csv": repo_relative(split_csv),
        "metadata_csv": repo_relative(metadata_csv),
        "parameters": {
            "phash_threshold": args.phash_threshold,
            "dhash_threshold": args.dhash_threshold,
            "workers": args.workers,
            "contact_sheet_pairs": args.contact_sheet_pairs,
            "limit": args.limit,
        },
        "images_audited": len(records),
        "split_counts": dict(Counter(record.split for record in records)),
        "class_count": len({record.class_name for record in records}),
        "read_errors": read_errors,
        "confirmed_exact_file_pairs": len(exact_file_pairs),
        "confirmed_exact_pixel_pairs": len(exact_pixel_pairs),
        "exact_file_group_stats": exact_group_stats(
            fingerprints, records, attribute="file_sha256"
        ),
        "exact_duplicate_label_conflicts": int(exact_label_conflicts),
        "total_candidate_pairs": len(candidate_rows),
        "geometrically_supported_near_duplicate_pairs": len(
            geometrically_supported
        ),
        "geometrically_supported_cross_label_pairs": len(
            supported_cross_label
        ),
        "conservative_duplicate_group_stats": conservative_group_stats,
        "candidate_counts_by_confidence": dict(sorted(tier_counts.items())),
        "candidate_counts_by_split_pair": dict(sorted(split_pair_counts.items())),
        "metadata_audit": metadata_audit,
        "contact_sheets": {
            "ranked": ranked_contact_sheets,
            "cross_label_high_confidence": cross_label_contact_sheets,
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "audit_report.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )

    print(f"Confirmed exact file pairs: {len(exact_file_pairs)}")
    print(f"Confirmed exact pixel pairs: {len(exact_pixel_pairs)}")
    print(f"Perceptual candidates: {len(candidate_rows)}")
    print(f"Geometrically supported near-duplicate pairs: {len(geometrically_supported)}")
    print(f"Supported cross-label pairs: {len(supported_cross_label)}")
    print(f"Label conflicts among exact duplicates: {exact_label_conflicts}")
    print(f"Read errors: {len(read_errors)}")
    print(f"Report: {output_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
