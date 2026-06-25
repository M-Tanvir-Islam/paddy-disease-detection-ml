"""
exp08 - MobileNetV3-Large with augmentation + weighted loss + OneCycleLR.

Run from repo root:
    python experiments/exp08_mobilenetv3large_onecycle/train.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import wandb
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.augment import get_train_transforms, get_val_transforms  # noqa: E402
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
from src.utils import set_seed, tee_to_file  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(k in name for k in ("classifier", "fc", "head")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training_onecycle(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    config: dict,
    device: str,
    checkpoint_path: str,
) -> tuple[float, dict]:
    use_amp = (device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_epochs = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
    weight_decay = config["training"]["weight_decay"]

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr_head"],
    )
    scheduler = None
    best_f1 = 0.0
    best_epoch = 0

    history: dict = {
        "epoch": [], "train_loss": [], "val_macro_f1": [],
        "val_accuracy": [], "lr": [], "epoch_seconds": [],
    }

    total_start = time.perf_counter()
    for epoch in range(num_epochs):
        if epoch == warmup_epochs:
            print(
                "--- Phase B: unfreeze all, OneCycleLR "
                f"max_lr={config['training']['onecycle_max_lr']} ---"
            )
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["training"]["onecycle_max_lr"],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["training"]["onecycle_max_lr"],
                steps_per_epoch=len(train_loader),
                epochs=max(num_epochs - warmup_epochs, 1),
                pct_start=config["training"].get("onecycle_pct_start", 0.1),
                div_factor=config["training"].get("onecycle_div_factor", 25.0),
                final_div_factor=config["training"].get("onecycle_final_div_factor", 10000.0),
            )

        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, scheduler
        )
        metrics = compute_metrics(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "lr": current_lr,
            "epoch_seconds": round(epoch_seconds, 2),
        }
        wandb.log(log)
        for k, v in log.items():
            history[k].append(v)

        print(
            f"Epoch {epoch+1:03d} | loss {train_loss:.4f} | "
            f"val macro-F1 {metrics['macro_f1']:.4f} | "
            f"val acc {metrics['accuracy']:.4f} | "
            f"lr {current_lr:.2e} | {epoch_seconds:.1f}s"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved best checkpoint (val macro-F1: {best_f1:.4f})")

    total_seconds = time.perf_counter() - total_start
    peak_vram_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_f1
    history["total_training_seconds"] = round(total_seconds, 1)
    history["mean_epoch_seconds"] = round(total_seconds / max(num_epochs, 1), 2)
    history["peak_vram_MB"] = peak_vram_mb
    print(
        f"\nTraining complete. Best val macro-F1: {best_f1:.4f} @ epoch {best_epoch}. "
        f"Total: {total_seconds/60:.1f} min"
        + (f", peak VRAM {peak_vram_mb} MB" if peak_vram_mb else "")
    )
    return best_f1, history


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
    val_tf = get_val_transforms(INPUT_SIZE)

    train_ds = PaddyDataset(csv_path=str(csv_path), split="train", transform=train_tf)
    val_ds = PaddyDataset(csv_path=str(csv_path), split="val", transform=val_tf)
    test_ds = PaddyDataset(csv_path=str(csv_path), split="test", transform=val_tf)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    nw = config["hardware"]["num_workers"]
    bs = config["training"]["batch_size"]
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

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
        best_f1, history = run_training_onecycle(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            device=device,
            checkpoint_path=str(ckpt_path),
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
                "scheduler": "OneCycleLR",
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