"""
exp17 - Validation calibration and latency comparison against exp16.

Loads the exp16 and exp17 checkpoints, runs both on the grouped validation
split, and reports macro-F1, negative log-likelihood, expected calibration
error (15 equal-width bins), and mean confidence on correct and incorrect
predictions. CPU latency is measured for both models back to back in the same
process so the comparison is not distorted by machine load during training.

Run from repo root:
    python experiments/exp17_mobilenetv3large_smoothing_ema/calibration_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.augment import get_eval_transforms  # noqa: E402
from src.classes import INPUT_SIZE, NUM_CLASSES  # noqa: E402
from src.dataset import PaddyDataset  # noqa: E402
from src.evaluate import compute_metrics, measure_inference_latency  # noqa: E402
from src.utils import set_seed  # noqa: E402

SPLIT_CSV = ROOT / "data/splits/kaggle_grouped_v2.csv"
CHECKPOINTS = {
    "exp16": ROOT / "checkpoints/exp16_grouped_split_best.pt",
    "exp17": ROOT / "checkpoints/exp17_smoothing_ema_best.pt",
}
OUTPUT = Path(__file__).parent / "results/calibration_check.json"
N_BINS = 15


def build_model() -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray) -> float:
    confidences = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def main() -> None:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PaddyDataset(csv_path=str(SPLIT_CSV), split="val",
                          transform=get_eval_transforms(INPUT_SIZE))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    report = {}
    for name, ckpt in CHECKPOINTS.items():
        model = build_model()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)

        metrics = compute_metrics(model, val_loader, device)
        probs = metrics["probs"]
        labels = np.array(metrics["labels"])
        correct = probs.argmax(axis=1) == labels
        confidences = probs.max(axis=1)
        nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))))

        latency = measure_inference_latency(model, INPUT_SIZE, device="cpu", num_runs=200)
        report[name] = {
            "macro_f1": round(metrics["macro_f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "errors": int((~correct).sum()),
            "nll": round(nll, 4),
            "ece_15bin": round(expected_calibration_error(probs, labels), 4),
            "mean_confidence_correct": round(float(confidences[correct].mean()), 4),
            "mean_confidence_incorrect": round(float(confidences[~correct].mean()), 4),
            "cpu_latency_mean_ms": latency["mean_ms"],
            "cpu_latency_p95_ms": latency["p95_ms"],
        }
        print(name, report[name])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
