"""Reproducibility + small helpers shared across experiments."""

import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - t0:.2f}s")


class _Tee:
    """Duplicates writes to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_to_file(log_path: str):
    """
    Redirects stdout + stderr to BOTH the terminal AND a log file.
    Use to capture full training output without manual shell redirection.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        sys.stdout = _Tee(orig_out, log_file)
        sys.stderr = _Tee(orig_err, log_file)
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()
