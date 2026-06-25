"""
exp03 - EfficientNet-B0 baseline.

Run from repo root:
    python experiments/exp03_efficientnetb0_baseline/train.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import wandb
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.augment import get_eval_transforms, get_train_transforms  # noqa: E402
from src.classes import CLASS_NAMES_EN, INPUT_SIZE, NUM_CLASSES  # noqa: E402
from src.dataset import PaddyDataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    compute_metrics,
    measure_inference_latency,
    measure_model_costs,
    print_classification_report,
    save_confusion_matrix,
    save_metrics_json,
    save_training_curves,
)
from src.loss import get_weighted_criterion  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import set_seed, tee_to_file  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    results_dir = ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "training.log"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path = ROOT / config["data"]["split_csv"]
    train_tf = get_train_transforms(INPUT_SIZE, augment=config["data"]["augmentation"])
    eval_tf = get_eval_transforms(INPUT_SIZE)

    train_ds = PaddyDataset(csv_path=str(csv_path), split="train", transform=train_tf)
    val_ds = PaddyDataset(csv_path=str(csv_path), split="val", transform=eval_tf)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    nw = config["hardware"]["num_workers"]
    bs = config["training"]["batch_size"]
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_model(NUM_CLASSES).to(device)

    if config["data"]["weighted_loss"]:
        criterion = get_weighted_criterion(train_ds.labels, NUM_CLASSES, device)
    else:
        criterion = nn.CrossEntropyLoss()

    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_run_name"],
        config=config,
        mode=config["logging"].get("wandb_mode", "disabled"),
    )

    ckpt_path = ROOT / config["output"]["checkpoint_path"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    with tee_to_file(str(log_path)):
        best_f1, history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_epochs=config["training"]["epochs"],
            warmup_epochs=config["training"]["warmup_epochs"],
            lr_head=config["training"]["lr_head"],
            lr_full=config["training"]["lr_full"],
            device=device,
            checkpoint_path=str(ckpt_path),
            weight_decay=config["training"]["weight_decay"],
        )

        print("\nLoading best checkpoint for final val evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        val_metrics = compute_metrics(model, val_loader, device)

        save_confusion_matrix(
            val_metrics["labels"], val_metrics["preds"], CLASS_NAMES_EN,
            str(results_dir / "confusion_matrix_val.png"),
        )
        save_training_curves(history, str(results_dir / "training_curves.png"))

        print("\nMeasuring model cost and inference latency...")
        cost = measure_model_costs(model, INPUT_SIZE, checkpoint_path=str(ckpt_path))
        lat_cpu = measure_inference_latency(model, INPUT_SIZE, device="cpu", num_runs=50)
        lat_gpu = (measure_inference_latency(model, INPUT_SIZE, device="cuda", num_runs=50)
                   if device == "cuda" else None)
        print(f"  params: {cost['total_params_M']} M  |  MACs: {cost['macs_G']} G  |  ckpt: {cost['checkpoint_size_MB']} MB")
        print(f"  CPU latency (mean/p95): {lat_cpu['mean_ms']} / {lat_cpu['p95_ms']} ms")
        if lat_gpu:
            print(f"  GPU latency (mean/p95): {lat_gpu['mean_ms']} / {lat_gpu['p95_ms']} ms")

        extra = {
            "model_cost": cost,
            "inference_latency_cpu": lat_cpu,
            "inference_latency_gpu": lat_gpu,
            "training": {
                "total_seconds": history.get("total_training_seconds"),
                "mean_epoch_seconds": history.get("mean_epoch_seconds"),
                "peak_vram_MB": history.get("peak_vram_MB"),
                "best_epoch": history.get("best_epoch"),
                "num_epochs": config["training"]["epochs"],
            },
        }
        save_metrics_json(val_metrics, CLASS_NAMES_EN, str(results_dir / "metrics_val.json"), extra=extra)
        print_classification_report(val_metrics["labels"], val_metrics["preds"], CLASS_NAMES_EN)

        print(f"\nBest val macro-F1: {best_f1:.4f} @ epoch {history['best_epoch']}")
        print(f"Results saved to: {results_dir}")
        print(f"Full log:         {log_path}")
        print("\nNote: test set was NOT evaluated. Per protocol, test is locked until")
        print("the final winning experiment.")

    wandb.finish()


if __name__ == "__main__":
    main()
