"""Loss functions for handling class imbalance."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def get_weighted_criterion(labels: list[int], num_classes: int, device: str) -> nn.Module:
    """
    Returns CrossEntropyLoss with per-class weights computed from training labels.
    Pass the flat list of integer labels from the training set.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)
