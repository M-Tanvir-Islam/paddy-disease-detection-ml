"""Apply reproducible proxy capture gates to the compatible Dhan manifest.

The repository documents an approximate blur threshold of 80 and minimum
green-pixel coverage of 60%, but it does not contain the product browser's
exact implementation. This script therefore records its algorithm as
``capture_gate_proxy_v1`` and preserves scores for later synchronization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "splits" / "dhan_supported_eval.csv"
DEFAULT_ELIGIBLE = ROOT / "data" / "splits" / "dhan_capture_eligible_eval.csv"
DEFAULT_REJECTED = ROOT / "data" / "splits" / "dhan_capture_rejected.csv"
DEFAULT_SCORES = (
    ROOT / "results" / "data_integrity" / "dhan_capture_gate_proxy" / "scores.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "results" / "data_integrity" / "dhan_capture_gate_proxy" / "summary.json"
)
GATE_VERSION = "capture_gate_proxy_v1"


def score_image(path: str, canvas_size: int) -> tuple[float, float]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        resized = oriented.resize((canvas_size, canvas_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # OpenCV hue uses [0, 179]. This broad plant-green range also retains some
    # yellow-green disease discoloration while excluding white/gray background.
    green_mask = (
        (hsv[:, :, 0] >= 20)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 40)
    )
    return blur_score, float(green_mask.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--eligible", type=Path, default=DEFAULT_ELIGIBLE)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--green-ratio-threshold", type=float, default=0.60)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    blur_scores: list[float] = []
    green_ratios: list[float] = []
    for path in frame["filepath"]:
        blur_score, green_ratio = score_image(path, args.canvas_size)
        blur_scores.append(blur_score)
        green_ratios.append(green_ratio)

    scored = frame.copy()
    scored["gate_version"] = GATE_VERSION
    scored["gate_canvas_size"] = args.canvas_size
    scored["blur_score"] = blur_scores
    scored["green_ratio"] = green_ratios
    scored["passes_blur"] = scored["blur_score"] >= args.blur_threshold
    scored["passes_green_coverage"] = (
        scored["green_ratio"] >= args.green_ratio_threshold
    )
    scored["capture_eligible"] = scored["passes_blur"] & scored["passes_green_coverage"]
    scored["rejection_reason"] = np.select(
        [
            ~scored["passes_blur"] & ~scored["passes_green_coverage"],
            ~scored["passes_blur"],
            ~scored["passes_green_coverage"],
        ],
        ["blur_and_leaf_coverage", "blur", "leaf_coverage"],
        default="",
    )

    eligible = scored[scored["capture_eligible"]].copy()
    rejected = scored[~scored["capture_eligible"]].copy()
    summary = {
        "gate_version": GATE_VERSION,
        "warning": "Proxy implementation; synchronize with exact product browser code before deployment.",
        "canvas_size": args.canvas_size,
        "blur_threshold": args.blur_threshold,
        "green_ratio_threshold": args.green_ratio_threshold,
        "hsv_green_mask_opencv": {"hue": [20, 100], "saturation_min": 40, "value_min": 40},
        "input_images": int(len(scored)),
        "eligible_images": int(len(eligible)),
        "rejected_images": int(len(rejected)),
        "eligible_rate": float(len(eligible) / max(len(scored), 1)),
        "eligible_by_background": eligible["background"].value_counts().sort_index().to_dict(),
        "eligible_by_class": eligible["class"].value_counts().sort_index().to_dict(),
        "rejection_reasons": rejected["rejection_reason"].value_counts().sort_index().to_dict(),
    }

    for path in (args.eligible, args.rejected, args.scores, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.eligible, index=False)
    rejected.to_csv(args.rejected, index=False)
    scored.to_csv(args.scores, index=False)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
