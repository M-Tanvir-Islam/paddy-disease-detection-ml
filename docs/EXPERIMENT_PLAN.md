# Experiment Plan

The full campaign that produces the deployed model. Three phases:
**bake-off**, **optimize the winners**, **finalize**. Every experiment is
one variable change against the prior, so attribution stays clean.

---

## Phase 1 â€” Bake-off (vanilla baselines) â€” **COMPLETE**

Goal: pick the best architectures under the production budget. Every run
used the same training pipeline (no augmentation, no class weighting, same
hyperparameters, same data split, same seed). Only the model varied.

| ID       | Model             | Source      | Params (M) | Val macro-F1 | CPU mean (ms) | Branch                         |
| -------- | ----------------- | ----------- | ---------- | ------------ | ------------- | ------------------------------ |
| exp01    | MobileNetV3-Small | torchvision | 1.528      | 0.9251       | 8.28          | `exp01_mobilenetv3small`       |
| exp02 â˜…  | MobileNetV3-Large | torchvision | 4.215      | **0.9518**   | 15.54         | `exp02_mobilenetv3large`       |
| exp03 â˜…â˜… | EfficientNet-B0   | torchvision | 4.020      | **0.9609**   | 25.79         | `exp03_efficientnetb0`         |
| exp04    | MobileViT-XXS     | timm        | 0.954      | 0.9227       | 21.57         | `exp04_mobilevit_xxs_baseline` |

Aggregated comparison (per-class F1, training curves, full cost table):
[`results/phase1_bakeoff_report.md`](../results/phase1_bakeoff_report.md)
on the `compare_models` branch.

**Phase 2 candidates selected:** exp02 (best Pareto trade-off) and
exp03 (highest absolute accuracy). Running both in parallel tracks so
the data decides the production winner, not a guess.

---

## Phase 2 â€” Optimize the two winners (parallel tracks)

Each experiment isolates a single improvement on top of the prior. We run
the same recipe on each track so the comparison stays clean.

### Track A â€” MobileNetV3-Large (built on exp02)

| ID    | Change vs prior                    | Hypothesis                                      | Branch                                  |
| ----- | ---------------------------------- | ----------------------------------------------- | --------------------------------------- |
| exp05 | + augmentation pipeline            | Reduces overfitting; lifts overall macro-F1     | `exp05_mobilenetv3large_augment`        |
| exp06 | exp05 + weighted CrossEntropy      | Lifts minority-class recall (BLB, downy mildew) | `exp06_mobilenetv3large_weighted`       |
| exp07 | exp06 + LR sweep                  | Tests whether LR tuning improves the current best MobileNet setup | `exp07_mobilenetv3large_lr_sweep`       |
| exp08 | best LR + Dhan-Shomadhan eval/data decision | Tests deployment-style robustness after LR tuning | `exp08_mobilenetv3large_dhan`           |

### Track B â€” EfficientNet-B0 (built on exp03)

| ID    | Change vs prior                    | Hypothesis                                                                   | Branch                                |
| ----- | ---------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------- |
| exp09 | + augmentation pipeline            | Same as exp05; B0 likely gains less (built-in regularization via SE + Swish) | `exp09_efficientnetb0_augment`        |
| exp10 | exp09 + weighted CrossEntropy      | Same as exp06                                                                | `exp10_efficientnetb0_weighted`       |
| exp11 | exp10 + Dhan-Shomadhan field added | Same as exp07                                                                | `exp11_efficientnetb0_dhan`           |
| exp12 | exp11 + label smoothing (0.1)      | Optional, only if exp11 hasn't diverged from Track A                         | `exp12_efficientnetb0_labelsmoothing` |


**LR sweep insertion:** exp07 is a diagnostic sweep on the current best MobileNetV3-Large setup (exp06). It compares high/default/low LR recipes before adding Dhan-Shomadhan or label smoothing. Keep the dataset fixed for exp07.
Stop a track early if it hits diminishing returns (e.g. < +0.005 macro-F1
on a step) or starts to overfit hard despite augmentation.

Both tracks merge into `compare_models` again after Phase 2 to build the
Phase 2 comparison report (same pattern as Phase 1).

---

## Phase 3 â€” Finalize

Run on the **single cross-track winner** chosen from Phase 2.

| Step                            | Action                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------- |
| Final test eval                 | `python scripts/evaluate_checkpoint.py --split test experiments/<winner>` â€” **one time only** |
| Held-out "deployment-like" eval | Reserved ~100 Dhan-Shomadhan field images â€” never trained on                                  |
| ONNX export                     | `experiments/<winner>/export_onnx.py`, verify shape + measure CPU latency                     |
| `MODEL_CARD.md`                 | Write the public model card for HF Hub                                                        |
| HF Hub upload                   | `huggingface-cli upload your-username/paddy-disease-model model.onnx`                         |
| Notify integration              | Point the `krishidoc` product repo at the new model artifact                                  |

---

## Branch convention

- `main` â€” infrastructure only (`src/`, `data/`, `docs/`, `scripts/`, `results/`). No `experiments/` folders.
- `expNN_<model>_<variant>` â€” one branch per experiment. Branch's `.gitignore` whitelists ONLY that experiment's folder.
- `compare_models` â€” aggregated multi-experiment view. `.gitignore` whitelists all experiment folders pulled in.

When creating a new experiment branch:

```bash
git checkout main
git checkout -b expNN_<model>_<variant>
# Update .gitignore on the new branch to whitelist experiments/expNN_<model>_<variant>/
# Scaffold config.yaml, train.py, README.md
# Run training, commit, push
```

---

## Design notes

- **Input size 224Ã—224 for all models.** MobileViT natively prefers 256 â€” we override to 224 for fairness.
- **Test set locked until Phase 3.** Every preview makes the final test number less trustworthy as an unbiased estimate.
- **Each experiment auto-reports** params, MACs/FLOPs, checkpoint size, CPU + GPU inference latency, peak VRAM, total training time â€” captured in `metrics_val.json`. No manual benchmarking needed.
- **Weights & Biases** is wired into every experiment. Flip `wandb_mode: "online"` in `config.yaml` after `wandb login` to enable live dashboards.
- **Reproducibility:** every experiment uses `seed=42`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Retraining with the same toolchain reproduces results to ~0.001 F1.
