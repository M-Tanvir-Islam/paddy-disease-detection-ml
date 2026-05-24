# Experiment Plan

The full campaign that produces the deployed model. Three phases:
**bake-off**, **optimize the winner**, **finalize**. Every experiment is
one variable change against the prior, so attribution stays clean.

---

## Phase 1 — Bake-off (vanilla baselines)

Goal: pick the best architecture under the production budget. Every run
uses the same training pipeline (no augmentation, no class weighting,
same hyperparameters, same data split, same seed). Only the model varies.

| ID    | Model             | Source      | Est. params | Status  | Val macro-F1 |
| ----- | ----------------- | ----------- | ----------- | ------- | ------------ |
| exp01 | MobileNetV3-Small | torchvision | 1.5 M       | done    | **0.9251**   |
| exp02 | MobileNetV3-Large | torchvision | ~4.2 M      | pending | —            |
| exp03 | EfficientNet-B0   | torchvision | ~4.0 M      | pending | —            |
| exp04 | MobileViT-XXS     | timm        | ~1.0 M      | pending | —            |

**Winner selection criteria** (in priority order):

1. Highest val macro-F1
2. Subject to: ONNX size < 50 MB, CPU inference < 500 ms (production budget)
3. Tiebreaker: lower CPU latency (better farmer experience on cheap phones)

---

## Phase 2 — Optimize the winner (one variable at a time)

Goal: take the winning architecture and push it. Each experiment isolates
a single improvement so we can attribute gains correctly.

| ID    | Change vs prior                    | Hypothesis                                                        |
| ----- | ---------------------------------- | ----------------------------------------------------------------- |
| exp05 | + augmentation pipeline            | Reduces overfitting; lifts overall macro-F1                       |
| exp06 | exp05 + weighted CrossEntropy      | Lifts per-class recall on minority classes (BLB, downy mildew)    |
| exp07 | exp06 + Dhan-Shomadhan field added | Tests robustness to deployment distribution (close-up natural-bg) |
| exp08 | exp07 + label smoothing (0.1)      | Optional: improves calibration; small accuracy bump               |

We may stop early if any experiment hits diminishing returns.

---

## Phase 3 — Finalize

| Step                     | Action                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------- |
| Final test eval          | Run `evaluate_checkpoint.py --split test` on the winner. **One time only.**        |
| ONNX export              | `experiments/expN_final_winner/export_onnx.py`, verify shape + measure CPU latency |
| Held-out deployment test | ~100 reserved Dhan-Shomadhan field images, never trained on                        |
| `MODEL_CARD.md`          | Write the public model card for HF Hub                                             |
| HF Hub upload            | `huggingface-cli upload ...`                                                       |
| Notify integration       | Point the `krishidoc` product repo at the new model artifact                       |

---

## Design notes

- **Input size 224×224 for all bake-off models.** MobileViT natively prefers
  256 — we override to 224 for fairness. Documented in each `config.yaml`.
- **Test set locked until Phase 3.** Every preview makes the final test
  number less trustworthy as an unbiased estimate.
- **Each experiment auto-reports** params, MACs/FLOPs, checkpoint size,
  CPU + GPU inference latency, peak VRAM, total training time — captured
  in `metrics_val.json`. No manual benchmarking needed.
- **Weights & Biases** is wired into every experiment. Flip
  `wandb_mode: "online"` in `config.yaml` after `wandb login` to enable
  live dashboards.
