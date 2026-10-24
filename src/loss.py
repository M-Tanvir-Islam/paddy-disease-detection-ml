"""Loss functions for handling class imbalance."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def get_weighted_criterion(
    labels: list[int],
    num_classes: int,
    device: str,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Returns CrossEntropyLoss with per-class weights computed from training labels.
    Pass the flat list of integer labels from the training set.

    `label_smoothing` defaults to 0.0 so every experiment before exp17 is
    unchanged; exp17 is the first to pass a non-zero value.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
