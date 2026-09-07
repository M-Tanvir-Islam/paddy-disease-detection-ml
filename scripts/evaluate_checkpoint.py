"""
Re-evaluate a saved checkpoint without retraining.

Computes everything the training script reports at end (metrics, cost,
latency, confusion matrix, classification report), but from an existing
.pt file. Use this when you want to add new metrics to an old experiment.

Usage:
    python scripts/evaluate_checkpoint.py <experiment_dir>

Example:
    python scripts/evaluate_checkpoint.py experiments/exp01_mobilenetv3small_baseline
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import get_eval_transforms  # noqa: E402
from src.classes import CLASS_NAMES_EN, INPUT_SIZE, NUM_CLASSES  # noqa: E402
from src.dataset import PaddyDataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    compute_metrics,
    measure_inference_latency,
    measure_model_costs,
    print_classification_report,
    save_confusion_matrix,
    save_metrics_json,
)


def build_model_from_config(config: dict) -> nn.Module:
    """Construct the model architecture matching the experiment's config."""
    name = config["model"]["name"]
    source = config["model"].get("source", "torchvision.models")

    if source == "torchvision.models":
        import torchvision.models as m
        model = getattr(m, name)(weights=None)
        # Match the head-replacement done at train time.
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        else:
            raise ValueError(f"Unknown head structure for torchvision model: {name}")
    elif source == "timm":
        import timm
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model source: {source}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        help="Path to experiment folder (e.g. experiments/exp01_...)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on. Use 'test' only for the final winner.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; useful for a non-destructive validation dry run.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        sys.exit(f"ERROR: {config_path} not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ckpt_path = ROOT / config["output"]["checkpoint_path"]
    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint {ckpt_path} not found")

    if args.split == "test":
        confirm = input(
            "\n⚠️  You are evaluating on the TEST set. Per protocol, only the\n"
            "    final winning model should ever be tested. Continue? [y/N]: "
        )
        if confirm.strip().lower() != "y":
            sys.exit("Cancelled.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Experiment: {exp_dir.name}")
    print(f"Checkpoint: {ckpt_path}")

    # --- Model ---
    model = build_model_from_config(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    # --- Data ---
    csv_path = ROOT / config["data"]["split_csv"]
    eval_tf = get_eval_transforms(INPUT_SIZE)
    ds = PaddyDataset(csv_path=str(csv_path), split=args.split, transform=eval_tf)
    print(f"{args.split} set size: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    # --- Metrics ---
    metrics = compute_metrics(model, loader, device)
    print(f"\nmacro-F1: {metrics['macro_f1']:.4f}  |  accuracy: {metrics['accuracy']:.4f}")

    # --- Cost & latency ---
    print("\nMeasuring model cost and inference latency...")
    cost = measure_model_costs(model, INPUT_SIZE, checkpoint_path=str(ckpt_path))
    lat_cpu = measure_inference_latency(model, INPUT_SIZE, device="cpu", num_runs=50)
    lat_gpu = (measure_inference_latency(model, INPUT_SIZE, device="cuda", num_runs=50)
               if device == "cuda" else None)
    print(f"  params: {cost['total_params_M']} M  |  MACs: {cost['macs_G']} G  |  ckpt: {cost['checkpoint_size_MB']} MB")
    print(f"  CPU latency (mean/p95): {lat_cpu['mean_ms']} / {lat_cpu['p95_ms']} ms")
    if lat_gpu:
        print(f"  GPU latency (mean/p95): {lat_gpu['mean_ms']} / {lat_gpu['p95_ms']} ms")

    # --- Save ---
    results_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / config["output"]["results_dir"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"metrics_{args.split}.json"
    cm_path = results_dir / f"confusion_matrix_{args.split}.png"

    extra = {
        "model_cost": cost,
        "inference_latency_cpu": lat_cpu,
        "inference_latency_gpu": lat_gpu,
        "training": {"note": "training-time stats not available (re-evaluated from checkpoint)"},
    }
    save_metrics_json(metrics, CLASS_NAMES_EN, str(json_path), extra=extra)
    save_confusion_matrix(metrics["labels"], metrics["preds"], CLASS_NAMES_EN, str(cm_path))
    print_classification_report(metrics["labels"], metrics["preds"], CLASS_NAMES_EN)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
