"""
PaddyDataset — loads paddy disease images for training or evaluation.

Two construction modes:

1. CSV-based (preferred): pass `csv_path` to a file with columns
   `filepath, class, split`. No file copying needed — references images
   at their original location on disk. Use `split=` to filter.

2. Folder-based: pass `root` and `split`; expects `root/split/<class>/*.jpg`.

Albumentations transforms expect numpy arrays — we convert PIL → numpy here.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .classes import CLASS_NAMES_EN


class PaddyDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform=None,
    ):
        if csv_path is None and (root is None or split is None):
            raise ValueError("Provide either csv_path OR (root + split)")

        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES_EN)}

        if csv_path is not None:
            df = pd.read_csv(csv_path)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split].reset_index(drop=True)
            unknown = set(df["class"]) - set(self.class_to_idx)
            if unknown:
                raise ValueError(
                    f"CSV contains classes not in CLASS_NAMES_EN: {sorted(unknown)}"
                )
            self.samples = [
                (row["filepath"], self.class_to_idx[row["class"]])
                for _, row in df.iterrows()
            ]
        else:
            split_root = Path(root) / split
            self.samples = []
            for cls in CLASS_NAMES_EN:
                cls_dir = split_root / cls
                if not cls_dir.is_dir():
                    continue
                for img_path in cls_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]
