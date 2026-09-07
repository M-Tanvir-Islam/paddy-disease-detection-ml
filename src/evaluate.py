"""Evaluation: compute val/test metrics, save confusion matrix, measure model cost."""

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


@torch.no_grad()
def compute_metrics(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    return {
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "preds": all_preds,
        "labels": all_labels,
        "probs": np.vstack(all_probs),
    }


def save_confusion_matrix(labels, preds, class_names, save_path: str) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_json(
    metrics: dict,
    class_names: list[str],
    save_path: str,
    extra: dict | None = None,
) -> None:
    labels = metrics["labels"]
    preds = metrics["preds"]
    probs = metrics["probs"]
    label_ids = list(range(len(class_names)))

    per_class_f1 = f1_score(labels, preds, average=None, labels=label_ids)
    per_class_p = precision_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    per_class_r = recall_score(labels, preds, average=None, labels=label_ids, zero_division=0)

    payload: dict = {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "top_2_accuracy": float(top_k_accuracy_score(labels, probs, k=2, labels=label_ids)),
        "top_3_accuracy": float(top_k_accuracy_score(labels, probs, k=3, labels=label_ids)),
        "macro_auc_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=label_ids)),
        "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
        "per_class_precision": dict(zip(class_names, per_class_p.tolist())),
        "per_class_recall": dict(zip(class_names, per_class_r.tolist())),
    }
    if extra:
        payload.update(extra)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)


def measure_model_costs(
    model: torch.nn.Module,
    input_size: int,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Returns parameter counts, FLOPs/MACs, and disk size for the model.
    Independent of training — works on any model instance.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs = None
    flops = None
    try:
        from thop import profile
        from copy import deepcopy
        # thop mutates the model — operate on a copy
        m = deepcopy(model).cpu().eval()
        dummy = torch.randn(1, 3, input_size, input_size)
        macs, _ = profile(m, inputs=(dummy,), verbose=False)
        flops = 2 * macs  # one MAC = one multiply + one add
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed: {e}")

    size_bytes = os.path.getsize(checkpoint_path) if checkpoint_path and os.path.exists(checkpoint_path) else None

    return {
        "total_params_M": round(total_params / 1e6, 3),
        "trainable_params_M": round(trainable_params / 1e6, 3),
        "macs_G": round(macs / 1e9, 3) if macs is not None else None,
        "flops_G": round(flops / 1e9, 3) if flops is not None else None,
        "checkpoint_size_MB": round(size_bytes / 1e6, 2) if size_bytes else None,
    }


@torch.no_grad()
def measure_inference_latency(
    model: torch.nn.Module,
    input_size: int,
    device: str,
    num_runs: int = 50,
    warmup: int = 10,
) -> dict:
    """
    Measures forward-pass latency in milliseconds at batch size 1.
    Includes warmup runs to exclude JIT compilation / cudnn autotuning cost.
    """
    model = model.to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    return {
        "device": device,
        "batch_size": 1,
        "input_size": input_size,
        "num_runs": num_runs,
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def print_classification_report(labels, preds, class_names) -> None:
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def save_training_curves(history: dict, save_path: str) -> None:
    """
    history: dict with keys 'epoch', 'train_loss', 'val_macro_f1', 'val_accuracy'
             each mapping to a list of values, one per epoch.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history["epoch"]
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["val_macro_f1"], label="Val macro-F1", color="tab:green")
    ax2.plot(epochs, history["val_accuracy"], label="Val accuracy", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
