# Paddy Disease Detection — ML Research Repository

> **Repo name:** `krishidoc-ml`
> **Purpose:** Standalone ML research — train, experiment, evaluate, and export
> the disease classification model that will be integrated into the KrishiDoc
> production application (`krishidoc` repo).
> **Status:** Pre-experiment — dataset not yet finalised, model not yet chosen.
> Update the marked sections as experiments progress.

---

## Table of Contents

1. [Context — How This Repo Fits the Bigger Project](#1-context--how-this-repo-fits-the-bigger-project)
2. [Research Goals](#2-research-goals)
3. [Folder Structure](#3-folder-structure)
4. [The Experiment Workflow](#4-the-experiment-workflow)
5. [src — Shared Training Code](#5-src--shared-training-code)
6. [Experiment Folder Convention](#6-experiment-folder-convention)
7. [Dataset Strategy](#7-dataset-strategy)
8. [Image Constraints from the Product](#8-image-constraints-from-the-product)
9. [Model Candidates](#9-model-candidates)
10. [Training Strategy](#10-training-strategy)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [ONNX Export](#12-onnx-export)
13. [Weights & Biases Setup](#13-weights--biases-setup)
14. [Hardware Plan](#14-hardware-plan)
15. [Integration Handoff to Main Repo](#15-integration-handoff-to-main-repo)
16. [GitHub Workflow](#16-github-workflow)
17. [AI Agent Context](#17-ai-agent-context)

---

## 1. Context — How This Repo Fits the Bigger Project

This repo is one of two repositories for the KrishiDoc project:

```
krishidoc-ml   (this repo)
  │
  │  Produces: model.onnx  +  MODEL_CARD.md
  │  Hosted on: Hugging Face Hub
  │             your-username/paddy-disease-model
  │
  ▼
krishidoc   (product repo)
  │
  ├── services/api/        ← FastAPI backend downloads model.onnx at startup
  ├── apps/web/            ← Vite + React + Tailwind frontend
  ├── rag/                 ← RAG ingestion, pgvector on Neon DB
  └── docs/ARCHITECTURE.md ← full system architecture
```

**This repo's only production output is `model.onnx`.**
Everything else — notebooks, experiment logs, training scripts, checkpoints —
stays here permanently for reproducibility and research purposes.

**The product repo does not contain training code.**
It contains only `ml/export_onnx.py` (the conversion script) and
`ml/MODEL_CARD.md` (which links back here).

### What the product needs from this repo

The FastAPI backend (`services/api/core/inference.py`) expects:

```
Input  name : "input"
Input  shape: float32 (1, 3, H, W)   ← H and W depend on chosen model
Output name : "logits"
Output shape: float32 (1, NUM_CLASSES)  ← raw logits, softmax applied server-side

NUM_CLASSES = number of disease classes in your final dataset
              (currently unknown — update after EDA)
```

The backend also needs the class name lists in order:

```python
CLASS_NAMES_EN = [...]   # English disease names, index matches class_id
CLASS_NAMES_BN = [...]   # Same in Bangla script
```

Both lists must be committed to this repo in `src/classes.py` and copied
to the product repo's `services/api/core/classes.py` when integrating.

---

## 2. Research Goals

### Primary goal
Find the best lightweight classification model for paddy disease detection
that satisfies all three constraints simultaneously:

1. **Accuracy** — val macro-F1 > 85% on the chosen dataset
2. **Speed** — CPU inference < 500ms on ONNX Runtime (HF Spaces CPU Basic tier)
3. **Size** — ONNX file < 50MB (fast startup, fits HF Hub free tier easily)

### Secondary goal
Produce a clean, reproducible experiment log that can be:
- Used as the methodology section of a research paper
- Linked from a LinkedIn post and GitHub profile
- Referenced by the production repo as the source of the deployed model

### What this repo deliberately does NOT do
- Build or test the FastAPI backend
- Build or test the React frontend
- Handle RAG, LLM calls, or database logic
- Deploy anything to production

All of that lives in `krishidoc`. This repo ends at `model.onnx`.

---

## 3. Folder Structure

```
krishidoc-ml/
│
├── README.md                        ← experiment results table + reproduce instructions
│
├── data/
│   ├── README.md                    ← dataset sources, licenses, class mappings, stats
│   ├── download_kaggle.py           ← reproducible Kaggle dataset download
│   ├── download_dhan.py             ← Dhan-Shomadhan download (update when access granted)
│   └── verify_dataset.py           ← check splits, class counts, image dimensions
│
├── notebooks/
│   ├── 01_eda.ipynb                 ← exploratory data analysis (run first, always)
│   ├── 02_baseline_check.ipynb     ← sanity check: can any model overfit one batch?
│   └── 03_error_analysis.ipynb     ← confusion matrix deep-dive after each experiment
│
├── src/                             ← shared code imported by all experiments
│   ├── __init__.py
│   ├── classes.py                   ← CLASS_NAMES_EN, CLASS_NAMES_BN  ← update after EDA
│   ├── dataset.py                   ← PyTorch Dataset class (generic, works for any folder)
│   ├── augment.py                   ← Albumentations pipeline (shared across experiments)
│   ├── train.py                     ← training loop (generic, accepts any model + config)
│   ├── evaluate.py                  ← F1, confusion matrix, ROC-AUC functions
│   ├── loss.py                      ← weighted CrossEntropyLoss + class weight computation
│   └── utils.py                     ← seed, checkpoint save/load, timer, image preview
│
├── experiments/
│   │
│   ├── exp01_[model_name]_baseline/      ← one folder per experiment, named on creation
│   │   ├── config.yaml                   ← ALL hyperparameters for this run
│   │   ├── train.py                      ← thin script: loads config, calls src/train.py
│   │   ├── results/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── training_curves.png
│   │   │   └── metrics.json              ← val F1, test F1, per-class scores
│   │   └── README.md                     ← what you tried, what happened, conclusion
│   │
│   ├── exp02_[model_name]_augment/
│   │   └── ...
│   │
│   ├── exp03_[model_name]_weighted_loss/
│   │   └── ...
│   │
│   └── expN_final_winner/
│       ├── config.yaml
│       ├── train.py
│       ├── export_onnx.py               ← only the winner gets this
│       ├── results/
│       │   ├── confusion_matrix.png
│       │   ├── training_curves.png
│       │   ├── metrics.json
│       │   └── sample_predictions.png   ← grid of correct + wrong predictions
│       └── README.md
│
├── results/
│   ├── experiment_log.md            ← running table: all experiments, one row each
│   └── figures/                     ← exported plots for paper / blog / LinkedIn
│
├── docs/
│   ├── ML_EXPERIMENTS.md           ← detailed methodology notes (paper-ready)
│   └── ARCHITECTURE.md             ← this file
│
├── requirements.txt                 ← pinned versions, pip installable
├── requirements_kaggle.txt          ← lighter version for Kaggle notebook installs
├── .env.example                     ← shows which env vars are needed (no secrets)
├── .gitignore
└── MODEL_CARD.md                    ← published to HF Hub alongside model.onnx
```

### What goes in `.gitignore`

```gitignore
# Dataset files — download locally, never commit raw images
data/raw/
data/processed/

# Model checkpoints — large binary files, hosted on HF Hub
checkpoints/
*.pt
*.pth
*.onnx

# Notebook outputs — commit notebooks without outputs
.ipynb_checkpoints/

# Env and secrets
.env
wandb/

# Python
__pycache__/
*.pyc
.venv/
```

---

## 4. The Experiment Workflow

Every experiment follows this exact sequence. No shortcuts.

```
Step 1  Create branch:  git checkout -b exp/01-mobilenet-small-baseline
        Create folder:  experiments/exp01_mobilenetv3small_baseline/

Step 2  Write config.yaml with ALL hyperparameters
        (model name, lr, epochs, batch size, augmentation on/off,
         weighted loss on/off, input size, seed)

Step 3  Run training:
        python experiments/exp01_.../train.py
        (logs automatically to W&B)

Step 4  Evaluate on val set:
        - Record val macro-F1 in experiment_log.md
        - Save confusion_matrix.png and training_curves.png to results/

Step 5  Write README.md for this experiment BEFORE merging:
        - What you changed vs previous experiment
        - What you expected
        - What actually happened
        - Conclusion and next step

Step 6  Merge to main:
        git checkout main
        git merge exp/01-mobilenet-small-baseline
        (only after README.md is written)

Step 7  Only touch test set for the FINAL winner.
        Never evaluate on test during experimentation.
```

**The test set is locked until you have a winner.**
All iteration happens on the validation set.
Test set evaluation happens exactly once — on the final chosen model.

---

## 5. src — Shared Training Code

These files are shared across all experiments. Changing them affects all
future experiments. If you need to change core logic for one experiment,
copy it into that experiment's folder instead of modifying `src/`.

### src/dataset.py

```python
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PaddyDataset(Dataset):
    """
    Expects folder structure:
        root/
          train/
            disease_class_name/
              image1.jpg
              image2.jpg
          val/
            disease_class_name/
              ...
          test/
            disease_class_name/
              ...

    Class indices are sorted alphabetically from folder names.
    Update src/classes.py after EDA to lock in the class order.
    """

    def __init__(self, root: str, split: str, transform=None):
        self.root      = Path(root) / split
        self.transform = transform
        self.samples   = []
        self.classes   = sorted([
            d.name for d in self.root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for img_path in (self.root / cls).glob("*.[jJpP][pPnN][gG]"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

### src/augment.py

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(input_size: int = 224):
    """
    Augmentation pipeline for training.
    Designed for close-up paddy leaf images (per product UX constraint).
    Simulates lighting variation, angle, and partial occlusion.

    DO NOT apply to val or test sets.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=25,
            val_shift_limit=10,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=input_size // 8,
            max_width=input_size // 8,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = 224):
    """
    Validation and test transforms — resize and normalize only.
    No augmentation.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
```

### src/train.py

```python
import torch
import wandb
from torch.utils.data import DataLoader
from src.evaluate import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_training(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str,
    checkpoint_path: str,
    experiment_name: str,
):
    scaler  = torch.cuda.amp.GradScaler()
    best_f1 = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        metrics = compute_metrics(model, val_loader, device)
        scheduler.step()

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
        })

        print(
            f"Epoch {epoch+1:03d} | "
            f"loss {train_loss:.4f} | "
            f"val F1 {metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best checkpoint (val macro-F1: {best_f1:.4f})")

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    return best_f1
```

### src/evaluate.py

```python
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.vstack(all_probs)
    macro_f1   = f1_score(all_labels, all_preds, average="macro")
    accuracy   = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "preds":    all_preds,
        "labels":   all_labels,
        "probs":    all_probs,
    }


def save_confusion_matrix(labels, preds, class_names, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def print_classification_report(labels, preds, class_names):
    print(classification_report(labels, preds, target_names=class_names))
```

### src/loss.py

```python
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_weighted_criterion(labels: list, num_classes: int, device: str):
    """
    Computes per-class weights to handle class imbalance.
    Pass all training labels as a flat list.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    return torch.nn.CrossEntropyLoss(weight=weight_tensor)
```

### src/classes.py

```python
# ── UPDATE THIS FILE AFTER EDA ──────────────────────────────────────────────
# Class order must match the alphabetical sort of your dataset folder names.
# This file is the source of truth for class indices across all experiments.
# Copy to krishidoc/services/api/core/classes.py when integrating.

# Placeholder — fill in after dataset EDA
CLASS_NAMES_EN: list[str] = [
    # "Bacterial Leaf Blight",
    # "Brown Spot",
    # "Healthy",
    # "Leaf Scald",
    # "Rice Blast",
    # "Rice Tungro",
    # "Sheath Blight",
]

CLASS_NAMES_BN: list[str] = [
    # "ব্যাকটেরিয়াল লিফ ব্লাইট",
    # "বাদামী দাগ",
    # "সুস্থ",
    # "পাতার পোড়া রোগ",
    # "ধানের ব্লাস্ট",
    # "টুংরো ভাইরাস",
    # "শিথ ব্লাইট",
]

NUM_CLASSES: int = len(CLASS_NAMES_EN)  # set after EDA

# Input size for the chosen model — update after winner is selected
# Common values: 224 (MobileNetV3, EfficientNet-B0, ResNet), 240 (EfficientNet-B1)
INPUT_SIZE: int = 224  # placeholder
```

---

## 6. Experiment Folder Convention

Every experiment folder follows the same pattern.
This makes it scannable and makes every experiment reproducible by anyone
(including AI agents) without asking you for context.

### config.yaml (fill in before training)

```yaml
# experiments/expNN_modelname_variant/config.yaml

experiment:
  id: "expNN"
  name: "modelname_variant"          # e.g. mobilenetv3small_augment
  description: "One sentence: what this run tests"

model:
  name: "TBD"                        # e.g. mobilenet_v3_small
  source: "torchvision.models"       # or timm, HuggingFace, custom
  pretrained: true
  pretrained_weights: "IMAGENET1K_V1"
  input_size: 224                    # H and W
  num_classes: null                  # fill after EDA

data:
  dataset: "TBD"                     # e.g. kaggle_paddy_2022
  root: "data/processed/"
  split_ratios: [0.70, 0.15, 0.15]
  augmentation: false                # true/false — toggle per experiment
  weighted_loss: false               # true/false — toggle per experiment

training:
  epochs: 50
  batch_size: 64
  optimizer: "AdamW"
  lr_head: 3.0e-4                   # Phase A — head only
  lr_full: 5.0e-5                   # Phase B — full network
  warmup_epochs: 5                  # Phase A duration
  scheduler: "CosineAnnealingLR"
  mixed_precision: true
  seed: 42

hardware:
  device: "cuda"                     # cuda / cpu / mps
  num_workers: 4

logging:
  wandb_project: "krishidoc"
  wandb_run_name: "expNN_modelname_variant"

output:
  checkpoint_path: "checkpoints/expNN_best.pt"
  results_dir: "experiments/expNN_modelname_variant/results/"
```

### README.md (write before merging to main)

```markdown
# Exp NN — [Model Name] — [Variant]

## What I changed
Compared to previous experiment: [one sentence]

## What I expected
[one sentence about your hypothesis]

## Results

| Metric | Value |
|---|---|
| Val macro-F1 | X.XX |
| Val accuracy | X.XX% |
| Best epoch | XX |
| Training time | XX min |

## Confusion matrix
![confusion matrix](results/confusion_matrix.png)

## What actually happened
[2–3 sentences: did it match expectation? Where did it fail?]

## Key insight
[One takeaway that informs the next experiment]

## Next experiment
[What you will try next and why]
```

---

## 7. Dataset Strategy

### Datasets to obtain (in priority order)

| Priority | Dataset | Images | Classes | Access | Action |
|---|---|---|---|---|---|
| 1 | Kaggle Paddy Disease 2022 | 10,407 | 10 | Public | Download now |
| 2 | Dhan-Shomadhan | ~5,000+ | 5 | Email authors | Contact now |
| 3 | BRRI Dataset | 19,000 | Bangladesh-specific | Email BRRI | Contact now |

Start with Kaggle Paddy 2022 — it is immediately available. Run your first
experiments on it while waiting for the others.

### After obtaining dataset — run EDA first

Before any training, run `notebooks/01_eda.ipynb` and record:

```
Total images:         ___
Number of classes:    ___
Images per class:     [list]
Min images in class:  ___  ← drives your augmentation strategy
Max images in class:  ___  ← drives your class weight calculation
Image dimensions:     ___  ← check for consistency
Mean image size:      ___
Are images close-up or wide-angle? ___   ← important for augmentation design
Are disease lesions large or small? ___  ← important for model choice
Any duplicate images? ___
Any corrupt files?    ___
```

**Update `src/classes.py` immediately after EDA.**
The class list must be locked before any training run.

### Dataset split

```python
# data/verify_dataset.py — run after split to confirm
# Stratified split — same class ratio in all three sets

from sklearn.model_selection import train_test_split

# 70% train / 15% val / 15% test
# Stratify by class label
train, temp   = train_test_split(data, test_size=0.30, stratify=labels, random_state=42)
val,   test   = train_test_split(temp, test_size=0.50, stratify=temp_labels, random_state=42)
```

**The test set is locked after split. Do not evaluate on it until final model.**

### Handling class imbalance

Three tools — use in this order, layer them:

1. Weighted CrossEntropyLoss (always use this — one line, no downside)
2. `WeightedRandomSampler` (over-samples minority classes per batch)
3. Extra augmentation on minority classes (only if still imbalanced after 1+2)

---

## 8. Image Constraints from the Product

This section documents the product-side UX decisions that directly affect
what the ML model needs to handle. Do not train on images that violate
these constraints.

### What the app enforces before sending to the model

The KrishiDoc mobile web app (in `krishidoc/apps/web/`) enforces two
client-side gates before the image is uploaded to the API:

**Gate 1 — Blur detection**
Laplacian variance check on the canvas. Images below threshold (≈80)
trigger a "please retake" prompt. Blurry images never reach the model.

**Gate 2 — Leaf coverage check**
Green pixel ratio check on the canvas. The app requires the leaf to fill
at least 60% of the frame. Wide-angle field shots are rejected at this gate.
Farmers are directed to hold the phone 15–20cm from the affected leaf.

### What this means for training data

- Training images should be close-up, single-leaf or small-cluster shots
- Avoid training on field-wide panorama images — the model will never see those
- Disease lesions should be clearly visible, not tiny specks in the distance
- If a dataset image would fail the blur or coverage gate, exclude it from training

### Image input specification (update after model selected)

```
Input size:    224×224 (placeholder — update after model winner chosen)
Colour space:  RGB
Normalization: mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]
               (ImageNet stats — used for all ImageNet-pretrained models)
Format at inference: float32 numpy array, shape (1, 3, H, W)
```

---

## 9. Model Candidates

> Model names are candidates only. Do not treat any as the decision.
> Run experiments in the order listed. Stop when macro-F1 > 85%.

### Classification candidates (Phase 1 — start here)

| Order | Model | torchvision name | Params | Est. ONNX | Est. CPU ms | Try if |
|---|---|---|---|---|---|---|
| 1st | MobileNetV3-Small | `mobilenet_v3_small` | 2.5M | ~8MB | 150–300ms | Always — fastest baseline |
| 2nd | EfficientNet-B0 | `efficientnet_b0` | 5.3M | ~20MB | 600–900ms | Small underperforms |
| 3rd | ResNet18 | `resnet18` | 11.7M | ~45MB | 200–400ms | Good debug reference |
| 4th | MobileNetV3-Large | `mobilenet_v3_large` | 5.4M | ~17MB | 300–500ms | Small underfits, want speed |

### How to swap models in the experiment script

```python
# experiments/expNN_.../train.py
# Replace ONLY this block per experiment — everything else stays identical

import torchvision.models as models
import torch.nn as nn

# ── SWAP THIS BLOCK ──────────────────────────────────────────────────────
# MobileNetV3-Small
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[-1] = nn.Linear(1024, NUM_CLASSES)

# EfficientNet-B0
# model = models.efficientnet_b0(weights="IMAGENET1K_V1")
# model.classifier[-1] = nn.Linear(1280, NUM_CLASSES)

# ResNet18
# model = models.resnet18(weights="IMAGENET1K_V1")
# model.fc = nn.Linear(512, NUM_CLASSES)

# MobileNetV3-Large
# model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
# model.classifier[-1] = nn.Linear(1280, NUM_CLASSES)
# ── END SWAP BLOCK ───────────────────────────────────────────────────────

model = model.to(device)
```

### Segmentation candidates (Phase 2 — conditional)

> Build segmentation only if, after deploying the classifier, user feedback
> shows that severity percentage is needed for accurate treatment advice.
> Do not build it speculatively.

If Phase 2 is triggered:
- Encoder: the Phase 1 winning backbone (reuse pretrained weights)
- Decoder: standard UNet skip-connection decoder
- Loss: 0.5 × Dice + 0.5 × CrossEntropy
- Dataset: requires pixel-level annotated masks — Li et al. 2022 is the
  best available option (DOI:10.3390/plants11223174)

---

## 10. Training Strategy

### Two-phase fine-tuning (apply to every experiment)

```
Phase A — Head warm-up (5 epochs)
  Freeze all backbone layers
  Train only the final classification head
  lr = 3e-4
  Purpose: prevent random head weights from destroying pretrained features

Phase B — Full fine-tune (remaining epochs)
  Unfreeze all layers
  Train end-to-end
  lr = 5e-5  (lower — backbone needs gentle updates)
  Purpose: adapt all features to paddy disease domain
```

```python
# Phase A — freeze backbone
for name, param in model.named_parameters():
    if "classifier" not in name and "fc" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config["training"]["lr_head"]
)

# Phase B — unfreeze all (after warmup_epochs)
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["lr_full"],
    weight_decay=1e-2
)
```

### Mixed precision (always on)

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(images)
    loss    = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Roughly halves VRAM usage. No accuracy cost. Always enable.

### Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs - warmup_epochs,
    eta_min=1e-6
)
```

### Batch size guide for RTX 3060 (12GB VRAM)

| Model | Input size | Batch size | VRAM usage |
|---|---|---|---|
| MobileNetV3-Small | 224×224 | 128 | ~4GB |
| MobileNetV3-Small | 224×224 | 64 | ~2.5GB |
| EfficientNet-B0 | 224×224 | 64 | ~5GB |
| ResNet18 | 224×224 | 128 | ~6GB |
| ResNet34 | 224×224 | 64 | ~7GB |

If CUDA OOM: halve batch size, add `gradient_accumulation_steps=2` to
keep effective batch size the same.

### Seed everything for reproducibility

```python
# src/utils.py
import torch, numpy as np, random, os

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
```

Call `set_seed(config["training"]["seed"])` at the top of every train.py.

---

## 11. Evaluation Protocol

### During experimentation (val set only)

Primary metric: **macro-averaged F1**
- Use macro (not weighted) because class imbalance is expected
- A model that only predicts the majority class will show low macro-F1
- Do not optimise for accuracy — it is misleading on imbalanced data

Secondary metrics logged to W&B each epoch:
- Per-class F1, Precision, Recall
- Overall accuracy
- Training loss

### After selecting the winner (test set — one time only)

```python
# Run evaluate.py on test set with best checkpoint loaded
model.load_state_dict(torch.load("checkpoints/expNN_best.pt"))
metrics = compute_metrics(model, test_loader, device)

# Record in results/metrics.json:
{
  "test_macro_f1":    X.XX,
  "test_accuracy":    X.XX,
  "per_class_f1":     { "class_name": X.XX, ... },
  "per_class_precision": { ... },
  "per_class_recall":    { ... }
}
```

Save confusion matrix, classification report, and a grid of
sample correct + incorrect predictions to `expN_final_winner/results/`.

### Minimum bar to proceed to integration

```
test macro-F1  > 0.82    (hard minimum)
test macro-F1  > 0.85    (target)
ONNX file size < 50MB
CPU inference  < 500ms   (measure on CPU, not GPU)
```

If any of these fail, do not integrate. Improve the model first.

---

## 12. ONNX Export

Run this only on the final winning experiment.

```python
# experiments/expN_final_winner/export_onnx.py
import torch
import onnxruntime as ort
import numpy as np
import os
from src.classes import INPUT_SIZE, NUM_CLASSES

# Load best checkpoint
model.load_state_dict(torch.load("checkpoints/expN_best.pt", map_location="cpu"))
model.eval()

# Export
dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names        = ["input"],
    output_names       = ["logits"],
    dynamic_axes       = {"input": {0: "batch_size"}},
    opset_version      = 17,
    do_constant_folding = True,
)

# Verify — MUST pass before uploading
sess    = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
out     = sess.run(None, {"input": dummy.numpy()})
assert out[0].shape == (1, NUM_CLASSES), f"Wrong output shape: {out[0].shape}"
size_mb = os.path.getsize("model.onnx") / 1e6
print(f"Export verified. Shape: {out[0].shape}. Size: {size_mb:.1f} MB")

# Measure CPU inference time
import time
times = []
for _ in range(50):
    t0 = time.perf_counter()
    sess.run(None, {"input": dummy.numpy()})
    times.append((time.perf_counter() - t0) * 1000)
print(f"CPU inference: {np.mean(times):.1f}ms avg, {np.max(times):.1f}ms max")
```

### Upload to HF Hub

```bash
# Install
pip install huggingface-hub

# Login
huggingface-cli login

# Upload model + card
huggingface-cli upload your-username/paddy-disease-model \
    experiments/expN_final_winner/model.onnx model.onnx

huggingface-cli upload your-username/paddy-disease-model \
    MODEL_CARD.md README.md
```

---

## 13. Weights & Biases Setup

Every experiment logs to W&B automatically. Set up once, forget about it.

```bash
pip install wandb
wandb login   # paste API key from wandb.ai/settings
```

```python
# At the top of every experiment's train.py
import wandb
import yaml

config = yaml.safe_load(open("config.yaml"))

wandb.init(
    project = config["logging"]["wandb_project"],   # "krishidoc"
    name    = config["logging"]["wandb_run_name"],  # "exp01_mobilenet_baseline"
    config  = config,                               # logs all hyperparams
)

# At the end
wandb.finish()
```

What W&B gives you:
- Training curves for every experiment on one dashboard
- Hyperparameter comparison across runs
- Confusion matrix images logged as artifacts
- Shareable URL for your research paper and LinkedIn post

**Link your W&B project in the repo README.md.**
Recruiters and reviewers can see all your experiments without cloning the repo.

---

## 14. Hardware Plan

| Task | Where | Why |
|---|---|---|
| EDA, debugging, small experiments | RTX 3060 local | Fast iteration, no quota |
| Full training runs (all candidates) | RTX 3060 local | 12GB VRAM sufficient for all candidates at 224×224 |
| Segmentation training (Phase 2, if triggered) | Kaggle T4 free | More VRAM headroom for UNet decoder |
| ONNX export and CPU inference timing | CPU (local) | Simulate production server conditions |

### Kaggle setup

```python
# requirements_kaggle.txt — lighter install for Kaggle notebooks
torch
torchvision
albumentations
scikit-learn
wandb
onnx
onnxruntime
matplotlib
seaborn
pyyaml
```

Push your `src/` folder and experiment script to a Kaggle dataset,
then import it in the notebook. Do not re-write training code in the notebook.

---

## 15. Integration Handoff to Main Repo

When the model is ready, these are the exact steps to integrate it into
`krishidoc` (the product repo). This section is instructions for you
or for an AI agent doing the integration.

### Files to copy from this repo to krishidoc

```
krishidoc-ml/src/classes.py
    →  krishidoc/services/api/core/classes.py

krishidoc-ml/experiments/expN_final_winner/results/metrics.json
    →  krishidoc/ml/MODEL_METRICS.json

krishidoc-ml/MODEL_CARD.md
    →  krishidoc/ml/MODEL_CARD.md
```

### What the product repo's inference.py needs

```python
# krishidoc/services/api/core/inference.py

# 1. Input shape — update INPUT_SIZE from src/classes.py
INPUT_SIZE = 224   # or whatever the winner uses

# 2. Preprocessing — must match training exactly
import albumentations as A
transform = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Apply transform, convert to float32, add batch dim → (1, 3, H, W)

# 3. ONNX session
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
logits  = session.run(None, {"input": tensor})[0]   # (1, NUM_CLASSES)

# 4. Class names — import from classes.py copy
from core.classes import CLASS_NAMES_EN, CLASS_NAMES_BN
```

### HF Hub model info to pass to the product repo

```
Repo ID:   your-username/paddy-disease-model
Filename:  model.onnx
Input:     float32 (1, 3, INPUT_SIZE, INPUT_SIZE)
Output:    float32 (1, NUM_CLASSES) — raw logits
Classes:   see src/classes.py
```

---

## 16. GitHub Workflow

### Branch naming

```
main                                 ← clean, merged experiments only
exp/01-mobilenetv3small-baseline     ← active experiment branches
exp/02-mobilenetv3small-augment
exp/03-efficientnet-b0
exp/N-final-winner
```

### Commit message format

```
[exp01] Add MobileNetV3-Small baseline config and train script
[exp01] Training complete — val macro-F1: 0.79
[exp01] Add confusion matrix and experiment README
[exp02] Add Albumentations augmentation pipeline
[exp02] Training complete — val macro-F1: 0.84 (+0.05 vs baseline)
[src]   Fix class weight computation for minority classes
[data]  Add Kaggle Paddy 2022 download script
[onnx]  Export expN winner — 8.2MB, 180ms CPU inference
```

### What to commit vs not commit

```
COMMIT:
  src/*.py              ← all shared training code
  experiments/*/config.yaml
  experiments/*/train.py
  experiments/*/README.md
  experiments/*/results/*.png
  experiments/*/results/metrics.json
  notebooks/*.ipynb     ← clear outputs before committing
  results/experiment_log.md
  MODEL_CARD.md
  requirements.txt

DO NOT COMMIT:
  data/raw/             ← too large, in .gitignore
  checkpoints/*.pt      ← too large, hosted on HF Hub
  model.onnx            ← hosted on HF Hub
  .env                  ← secrets
  wandb/                ← in .gitignore
```

### experiment_log.md — update after every merged experiment

```markdown
# Experiment Log

| ID | Model | Augment | Weighted Loss | Val macro-F1 | Notes |
|---|---|---|---|---|---|
| exp01 | MobileNetV3-Small | No | No | 0.XX | baseline |
| exp02 | MobileNetV3-Small | Yes | No | 0.XX | +X.XX vs exp01 |
| exp03 | MobileNetV3-Small | Yes | Yes | 0.XX | +X.XX vs exp02 |
| exp04 | EfficientNet-B0 | Yes | Yes | 0.XX | compare vs exp03 |
| expN | [winner] | Yes | Yes | 0.XX | **selected for production** |
```

---

## 17. AI Agent Context

This section is written specifically for AI coding agents that will work
on tasks in this repository. Read this section before starting any task.

### What this repo is

A standalone ML research repository for paddy (rice) disease classification.
It produces one output: `model.onnx` — a trained classification model that
gets uploaded to Hugging Face Hub and consumed by a separate FastAPI backend
in the `krishidoc` product repository.

### What this repo is NOT

It is not the web app. It is not the API. It is not the RAG system.
Do not add FastAPI, Streamlit, database, or deployment code here.

### Core facts for every task

```
Framework:        PyTorch (training) + ONNX Runtime (export verification only)
Python version:   3.11
GPU:              NVIDIA RTX 3060, 12GB VRAM (local training)
Mixed precision:  always enabled (torch.cuda.amp)
Experiment track: Weights & Biases
Model registry:   Hugging Face Hub (your-username/paddy-disease-model)
Shared code:      src/ — do not duplicate, import from here
Config format:    YAML (one config.yaml per experiment)
Primary metric:   val macro-F1 (not accuracy, not weighted F1)
```

### File to read before any task

```
docs/ARCHITECTURE.md        ← this file
src/classes.py              ← class names and NUM_CLASSES
results/experiment_log.md   ← what has been tried so far
```

### Task-specific context

| Task | Files to read first | Key constraint |
|---|---|---|
| Add a new experiment | `src/train.py`, `src/dataset.py`, any existing `config.yaml` | Copy config.yaml from nearest experiment, change only what differs |
| Modify augmentation | `src/augment.py`, Section 8 of this doc | Must match product UX constraints (close-up images only) |
| Add a new model candidate | Section 9 of this doc, any existing `experiments/*/train.py` | Only swap the model block — keep all other training code identical |
| Run evaluation | `src/evaluate.py`, Section 11 of this doc | Val set only during experiments. Test set only for final winner. |
| Export to ONNX | Section 12 of this doc | Verify shape AND measure CPU inference time. Both must pass. |
| Update class list | `src/classes.py` | Must run EDA first. Class order = alphabetical folder sort. |

### Current state (update this as work progresses)

```
Dataset:           [ ] Not yet obtained — download Kaggle Paddy 2022 first
EDA:               [ ] Not yet run
Classes:           [ ] Not yet defined — src/classes.py has placeholders
Experiments run:   0
Best val macro-F1: N/A
Winner selected:   No
ONNX exported:     No
HF Hub uploaded:   No
```

### When the model is ready for integration

Notify the human. Do not open pull requests to `krishidoc` or modify
the product repo. The integration checklist (Section 15) covers what
needs to happen, but a human should initiate it after reviewing
the final test metrics.
