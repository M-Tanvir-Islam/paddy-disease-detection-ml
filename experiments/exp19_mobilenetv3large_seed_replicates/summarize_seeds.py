"""
exp19 - Summarise exp16 (seed 42) and the exp19 seed replicates (seeds 1, 2).

Reads each run's `metrics_val.json` and reports the mean and sample standard
deviation of validation macro-F1, accuracy, and per-class F1 across the three
seeds. Writes `results/seed_summary.json`.

Run from repo root after both exp19 runs have finished:
    python experiments/exp19_mobilenetv3large_seed_replicates/summarize_seeds.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    42: ROOT / "experiments/exp16_mobilenetv3large_grouped_split/results/metrics_val.json",
    1: ROOT / "experiments/exp19_mobilenetv3large_seed_replicates/results/seed_1/metrics_val.json",
    2: ROOT / "experiments/exp19_mobilenetv3large_seed_replicates/results/seed_2/metrics_val.json",
}
OUTPUT = Path(__file__).parent / "results/seed_summary.json"


def mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=1)), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4)}


def main() -> None:
    runs = {seed: json.loads(path.read_text()) for seed, path in RUNS.items()}
    classes = list(runs[42]["per_class_f1"])

    summary = {
        "seeds": list(runs),
        "per_seed": {
            seed: {
                "macro_f1": round(m["macro_f1"], 4),
                "accuracy": round(m["accuracy"], 4),
                "best_epoch": m["training"]["best_epoch"],
            }
            for seed, m in runs.items()
        },
        "macro_f1": mean_std([m["macro_f1"] for m in runs.values()]),
        "accuracy": mean_std([m["accuracy"] for m in runs.values()]),
        "per_class_f1": {
            c: mean_std([m["per_class_f1"][c] for m in runs.values()]) for c in classes
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))
    for seed, row in summary["per_seed"].items():
        print(f"seed {seed:>2}: macro-F1 {row['macro_f1']:.4f}  acc {row['accuracy']:.4f}  best epoch {row['best_epoch']}")
    mf = summary["macro_f1"]
    print(f"macro-F1 mean {mf['mean']:.4f} +/- {mf['std']:.4f} (range {mf['min']:.4f}-{mf['max']:.4f})")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
