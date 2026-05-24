"""
Two-phase fine-tuning loop.

Phase A (epochs 0..warmup_epochs-1): freeze backbone, train head only at lr_head.
Phase B (epochs warmup_epochs..end): unfreeze all, train at lr_full with
                                     cosine annealing.

Logs to W&B every epoch and saves the best checkpoint (highest val macro-F1).
Captures per-epoch wall-clock time and peak VRAM into the returned history.
"""

import time
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader

from .evaluate import compute_metrics


def _freeze_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(k in name for k in ("classifier", "fc", "head")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def _unfreeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
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
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    num_epochs: int,
    warmup_epochs: int,
    lr_head: float,
    lr_full: float,
    device: str,
    checkpoint_path: str,
    weight_decay: float = 1e-2,
) -> tuple[float, dict]:
    use_amp = (device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Phase A
    _freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
    )
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    best_f1 = 0.0
    best_epoch = 0

    history: dict = {
        "epoch": [], "train_loss": [], "val_macro_f1": [],
        "val_accuracy": [], "lr": [], "epoch_seconds": [],
    }

    total_start = time.perf_counter()
    for epoch in range(num_epochs):
        if epoch == warmup_epochs:
            print(f"--- Phase B: unfreeze all, lr={lr_full}, cosine annealing ---")
            _unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr_full, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=1e-6
            )

        epoch_start = time.perf_counter()
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        metrics = compute_metrics(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start

        if scheduler is not None:
            scheduler.step()

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 2),
        }
        wandb.log(log)
        for k, v in log.items():
            history[k].append(v)

        print(
            f"Epoch {epoch+1:03d} | loss {train_loss:.4f} | "
            f"val macro-F1 {metrics['macro_f1']:.4f} | "
            f"val acc {metrics['accuracy']:.4f} | "
            f"{epoch_seconds:.1f}s"
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
