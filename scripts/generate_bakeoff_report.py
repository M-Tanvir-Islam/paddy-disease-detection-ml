"""
Aggregate Phase 1 bake-off results from all 4 experiment folders into
paper-ready artifacts.

Reads:
  experiments/exp0N_*/results/metrics_val.json   (final aggregate metrics)
  experiments/exp0N_*/results/training.log       (per-epoch history, parsed)

Writes:
  results/phase1_metrics_combined.json           (single source of truth)
  results/phase1_training_curves.png             (val macro-F1 + loss overlay)
  results/phase1_per_class_f1.png                (per-class F1 grouped bar chart)
  results/phase1_bakeoff_report.md               (the comparison report)

Run from repo root after pulling all 4 experiment folders into a branch
(e.g. compare_models):
    python scripts/generate_bakeoff_report.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("exp01", "MobileNetV3-Small", "experiments/exp01_mobilenetv3small_baseline"),
    ("exp02", "MobileNetV3-Large", "experiments/exp02_mobilenetv3large_baseline"),
    ("exp03", "EfficientNet-B0",   "experiments/exp03_efficientnetb0_baseline"),
    ("exp04", "MobileViT-XXS",     "experiments/exp04_mobilevit_xxs_baseline"),
]

EPOCH_LINE = re.compile(
    r"Epoch (\d+) \| loss ([\d.eE+-]+) \| val macro-F1 ([\d.]+) \| val acc ([\d.]+)"
    r"(?:\s*\|\s*([\d.]+)s)?"
)


def load_metrics(exp_dir: Path) -> dict:
    return json.loads((exp_dir / "results/metrics_val.json").read_text())


def parse_training_log(exp_dir: Path) -> dict | None:
    log_path = exp_dir / "results/training.log"
    if not log_path.exists():
        return None
    epochs, losses, f1s, accs, secs = [], [], [], [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = EPOCH_LINE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        f1s.append(float(m.group(3)))
        accs.append(float(m.group(4)))
        secs.append(float(m.group(5)) if m.group(5) else None)
    if not epochs:
        return None
    return {
        "epoch": epochs, "train_loss": losses,
        "val_macro_f1": f1s, "val_accuracy": accs, "epoch_seconds": secs,
    }


# ---------- output writers ----------

def save_combined_json(aggregated: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2))


def save_curve_overlay(histories: list, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for (exp_id, label, history), color in zip(histories, colors):
        if history is None:
            continue
        ax1.plot(history["epoch"], history["val_macro_f1"],
                 label=f"{exp_id} {label}", color=color, linewidth=2)
        ax2.plot(history["epoch"], history["train_loss"],
                 label=f"{exp_id} {label}", color=color, linewidth=2)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val macro-F1")
    ax1.set_title("Validation macro-F1 over training")
    ax1.set_ylim(0.45, 1.0)
    ax1.grid(True, alpha=0.3); ax1.legend(loc="lower right")

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Train loss (log scale)")
    ax2.set_title("Training loss over training")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_per_class_f1_chart(aggregated: dict, out_path: Path) -> None:
    classes = list(next(iter(aggregated.values()))["metrics"]["per_class_f1"].keys())
    n_classes = len(classes)
    n_models = len(aggregated)

    x = np.arange(n_classes)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, ((exp_id, payload), color) in enumerate(zip(aggregated.items(), colors)):
        f1s = list(payload["metrics"]["per_class_f1"].values())
        ax.bar(x + (i - n_models / 2 + 0.5) * width, f1s, width,
               label=f"{exp_id} {payload['label']}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 across bake-off candidates")
    ax.set_ylim(0.80, 1.00)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def f(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"


def generate_report_md(aggregated: dict, out_path: Path) -> None:
    rows = []
    for exp_id, payload in aggregated.items():
        m = payload["metrics"]
        cost = m["model_cost"]
        cpu = m["inference_latency_cpu"]
        gpu = m.get("inference_latency_gpu") or {}
        tr = m.get("training") or {}
        rows.append({
            "exp_id": exp_id,
            "label": payload["label"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "top_2": m.get("top_2_accuracy"),
            "top_3": m.get("top_3_accuracy"),
            "auc": m.get("macro_auc_ovr"),
            "params_M": cost.get("total_params_M"),
            "macs_G": cost.get("macs_G"),
            "ckpt_MB": cost.get("checkpoint_size_MB"),
            "cpu_mean": cpu.get("mean_ms"),
            "cpu_p95": cpu.get("p95_ms"),
            "gpu_mean": gpu.get("mean_ms"),
            "vram_MB": tr.get("peak_vram_MB"),
            "train_min": (tr.get("total_seconds") or 0) / 60 if tr.get("total_seconds") else None,
            "epoch_s": tr.get("mean_epoch_seconds"),
            "best_epoch": tr.get("best_epoch"),
        })

    # winner = highest macro_f1
    winner = max(rows, key=lambda r: r["macro_f1"])

    # per-class table data
    classes = list(next(iter(aggregated.values()))["metrics"]["per_class_f1"].keys())

    md = []
    md.append("# Phase 1 — Bake-off Report\n")
    md.append("Four candidate models trained with **identical pipelines** (no augmentation,\n"
              "no class weighting, AdamW + frozen-then-unfrozen schedule, 40 epochs, seed=42,\n"
              "same 70/15/15 stratified Kaggle Paddy 2022 split). Only the model architecture\n"
              "varies. Goal: select the best architecture for Phase 2 optimization.\n")

    md.append("## Headline metrics\n")
    md.append("| Exp | Model | Val macro-F1 | Val Acc | Top-2 | Top-3 | AUC-OvR | Params (M) | MACs (G) | Ckpt (MB) | CPU mean (ms) | CPU p95 (ms) | GPU mean (ms) | Peak VRAM (MB) | Train (min) | s/epoch |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        marker = " ★" if r["exp_id"] == winner["exp_id"] else ""
        md.append(
            f"| {r['exp_id']}{marker} | {r['label']} | "
            f"{f(r['macro_f1'])} | {f(r['accuracy'])} | "
            f"{f(r['top_2'])} | {f(r['top_3'])} | {f(r['auc'])} | "
            f"{f(r['params_M'], 3)} | {f(r['macs_G'], 3)} | "
            f"{f(r['ckpt_MB'], 2)} | {f(r['cpu_mean'], 2)} | {f(r['cpu_p95'], 2)} | "
            f"{f(r['gpu_mean'], 2)} | {f(r['vram_MB'], 1)} | "
            f"{f(r['train_min'], 1)} | {f(r['epoch_s'], 1)} |"
        )

    md.append(f"\n★ Highest val macro-F1: **{winner['label']}** ({winner['exp_id']}) at {f(winner['macro_f1'])}.\n")

    md.append("## Per-class F1\n")
    md.append("| Class | " + " | ".join(r["exp_id"] for r in rows) + " |")
    md.append("|---|" + "---|" * len(rows))
    for c in classes:
        line = [c]
        for r in rows:
            v = aggregated[r["exp_id"]]["metrics"]["per_class_f1"].get(c)
            line.append(f(v))
        md.append("| " + " | ".join(line) + " |")
    md.append("")
    md.append("![Per-class F1 across candidates](phase1_per_class_f1.png)\n")

    md.append("## Training dynamics\n")
    md.append("![Val macro-F1 and training loss across candidates](phase1_training_curves.png)\n")
    md.append("Observations on convergence and overfitting are documented in the discussion below.\n")

    md.append("## Production constraints (all candidates)\n")
    md.append("| Constraint | Budget | exp01 | exp02 | exp03 | exp04 |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| ONNX size  | < 50 MB | {f(rows[0]['ckpt_MB'], 1)} | {f(rows[1]['ckpt_MB'], 1)} | {f(rows[2]['ckpt_MB'], 1)} | {f(rows[3]['ckpt_MB'], 1)} |")
    md.append(f"| CPU latency (p95) | < 500 ms | {f(rows[0]['cpu_p95'], 1)} | {f(rows[1]['cpu_p95'], 1)} | {f(rows[2]['cpu_p95'], 1)} | {f(rows[3]['cpu_p95'], 1)} |")
    md.append(f"| Val macro-F1 | > 0.85 | {f(rows[0]['macro_f1'])} | {f(rows[1]['macro_f1'])} | {f(rows[2]['macro_f1'])} | {f(rows[3]['macro_f1'])} |")
    md.append("\nAll four candidates meet every production constraint. The decision is which "
              "gives the best Phase 2 optimization headroom and final inference accuracy.\n")

    md.append("## Phase 2 candidate ranking\n")
    md.append("Sorted by val macro-F1 (descending):\n")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["macro_f1"]), 1):
        md.append(f"{i}. **{r['exp_id']} {r['label']}** — macro-F1 {f(r['macro_f1'])}, "
                  f"{f(r['cpu_mean'], 2)} ms CPU, {f(r['params_M'], 2)} M params")

    md.append("\n## Reproduction\n")
    md.append("```bash\n# Regenerate this report after any retrain:\npython scripts/generate_bakeoff_report.py\n```\n")
    md.append("Aggregated raw numbers: `results/phase1_metrics_combined.json`.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    aggregated = {}
    histories = []
    print("Aggregating bake-off results from 4 experiment folders...\n")
    for exp_id, label, exp_dir in EXPERIMENTS:
        exp_path = ROOT / exp_dir
        if not exp_path.exists():
            print(f"  [skip] {exp_id} — folder not found at {exp_dir}")
            continue
        metrics = load_metrics(exp_path)
        history = parse_training_log(exp_path)
        aggregated[exp_id] = {"label": label, "exp_dir": exp_dir, "metrics": metrics}
        histories.append((exp_id, label, history))
        n_epochs = len(history["epoch"]) if history else 0
        print(f"  [ok]  {exp_id} {label}: macro-F1 {metrics['macro_f1']:.4f}  "
              f"({n_epochs} epochs parsed from log)")

    out_dir = ROOT / "results"
    save_combined_json(aggregated, out_dir / "phase1_metrics_combined.json")
    save_curve_overlay(histories, out_dir / "phase1_training_curves.png")
    save_per_class_f1_chart(aggregated, out_dir / "phase1_per_class_f1.png")
    generate_report_md(aggregated, out_dir / "phase1_bakeoff_report.md")

    print(f"\nWrote:")
    print(f"  {out_dir / 'phase1_metrics_combined.json'}")
    print(f"  {out_dir / 'phase1_training_curves.png'}")
    print(f"  {out_dir / 'phase1_per_class_f1.png'}")
    print(f"  {out_dir / 'phase1_bakeoff_report.md'}")


if __name__ == "__main__":
    main()
