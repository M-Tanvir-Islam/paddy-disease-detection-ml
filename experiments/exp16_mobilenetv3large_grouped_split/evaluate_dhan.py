"""Evaluate exp16 on compatible Dhan-Shomadhan classes without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.augment import get_eval_transforms  # noqa: E402
from src.classes import CLASS_NAMES_EN, INPUT_SIZE, NUM_CLASSES  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "dhan_supported_eval.csv"
DEFAULT_OUTPUT_DIR = EXP_DIR / "results" / "dhan_deployment_eval"
SUPPORTED_CLASSES = ["blast", "brown_spot", "tungro"]


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES_EN)}
        self.labels = [class_to_idx[name] for name in self.frame["class"]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        with Image.open(self.frame.iloc[index]["filepath"]) as source:
            image = np.array(ImageOps.exif_transpose(source).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, self.labels[index], index


def build_model() -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def metrics_for_slice(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    supported_ids = np.array([CLASS_NAMES_EN.index(name) for name in SUPPORTED_CLASSES])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=supported_ids, zero_division=0
    )
    restricted_preds = supported_ids[np.argmax(probs[:, supported_ids], axis=1)]
    r_precision, r_recall, r_f1, _ = precision_recall_fscore_support(
        labels, restricted_preds, labels=supported_ids, zero_division=0
    )
    top2 = np.argsort(probs, axis=1)[:, -2:]
    other_mask = ~np.isin(preds, supported_ids)

    return {
        "images": int(len(labels)),
        "full_10_class_accuracy": float(np.mean(preds == labels)),
        "supported_class_macro_f1": float(np.mean(f1)),
        "top_2_accuracy": float(np.mean([label in row for label, row in zip(labels, top2)])),
        "out_of_scope_prediction_rate": float(np.mean(other_mask)),
        "out_of_scope_prediction_counts": {
            CLASS_NAMES_EN[index]: int(np.sum(preds[other_mask] == index))
            for index in sorted(set(preds[other_mask].tolist()))
        },
        "per_class": {
            name: {
                "precision": float(precision[position]),
                "recall": float(recall[position]),
                "f1": float(f1[position]),
                "support": int(support[position]),
            }
            for position, name in enumerate(SUPPORTED_CLASSES)
        },
        "restricted_3_class_accuracy": float(np.mean(restricted_preds == labels)),
        "restricted_3_class_macro_f1": float(np.mean(r_f1)),
        "restricted_3_class_per_class": {
            name: {
                "precision": float(r_precision[position]),
                "recall": float(r_recall[position]),
                "f1": float(r_f1[position]),
            }
            for position, name in enumerate(SUPPORTED_CLASSES)
        },
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = ROOT / config["output"]["checkpoint_path"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    frame = pd.read_csv(args.manifest)
    unknown = set(frame["class"]) - set(SUPPORTED_CLASSES)
    if unknown:
        raise ValueError(f"Manifest contains unsupported evaluation labels: {sorted(unknown)}")
    if set(frame["split"]) != {"dhan_eval"}:
        raise ValueError("Expected only split=dhan_eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ManifestDataset(frame, get_eval_transforms(INPUT_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=device == "cuda",
    )
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    for images, batch_labels, batch_indices in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy().tolist())
        indices.extend(batch_indices.numpy().tolist())

    probs = np.vstack(probabilities)
    labels_array = np.asarray(labels)
    preds = probs.argmax(axis=1)
    supported_ids = np.array([CLASS_NAMES_EN.index(name) for name in SUPPORTED_CLASSES])
    restricted_preds = supported_ids[np.argmax(probs[:, supported_ids], axis=1)]

    ordered = frame.iloc[indices].reset_index(drop=True).copy()
    ordered["predicted_class"] = [CLASS_NAMES_EN[index] for index in preds]
    ordered["confidence"] = probs.max(axis=1)
    ordered["correct"] = preds == labels_array
    ordered["restricted_3_class_prediction"] = [
        CLASS_NAMES_EN[index] for index in restricted_preds
    ]
    ordered["restricted_3_class_correct"] = restricted_preds == labels_array

    report = {
        "scope": "external_deployment_evaluation_no_training",
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "manifest": args.manifest.resolve().relative_to(ROOT).as_posix(),
        "supported_classes": SUPPORTED_CLASSES,
        "excluded_dhan_classes": ["leaf_scald", "sheath_blight"],
        "overall": metrics_for_slice(labels_array, preds, probs),
        "by_background": {},
    }
    for background, rows in ordered.groupby("background", sort=True):
        positions = rows.index.to_numpy()
        report["by_background"][background] = metrics_for_slice(
            labels_array[positions], preds[positions], probs[positions]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    ordered.to_csv(args.output_dir / "predictions.csv", index=False)

    true_ids = [CLASS_NAMES_EN.index(name) for name in SUPPORTED_CLASSES]
    matrix = confusion_matrix(labels_array, preds, labels=list(range(NUM_CLASSES)))[true_ids]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES_EN,
        yticklabels=SUPPORTED_CLASSES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted class (full 10-class model)")
    ax.set_ylabel("Dhan true class")
    ax.set_title("Exp16 on compatible Dhan-Shomadhan classes")
    plt.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
