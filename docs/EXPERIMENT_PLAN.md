# Experiment Plan

The campaign that produces the deployed model. Three phases: bake-off,
optimize the winners, and finalize. Every experiment changes one main variable
against the prior result so attribution stays clean.

---

## Phase 1 - Bake-off (vanilla baselines) - COMPLETE

Goal: pick the best architectures under the production budget. Every run used
the same training pipeline: no augmentation, no class weighting, same
hyperparameters, same data split, same seed. Only the model varied.

| ID | Model | Source | Params (M) | Val macro-F1 | CPU mean (ms) | Branch |
| --- | --- | --- | ---: | ---: | ---: | --- |
| exp01 | MobileNetV3-Small | torchvision | 1.528 | 0.9251 | 8.28 | `exp01_mobilenetv3small` |
| exp02 | MobileNetV3-Large | torchvision | 4.215 | 0.9518 | 15.54 | `exp02_mobilenetv3large` |
| exp03 | EfficientNet-B0 | torchvision | 4.020 | 0.9609 | 25.79 | `exp03_efficientnetb0` |
| exp04 | MobileViT-XXS | timm | 0.954 | 0.9227 | 21.57 | `exp04_mobilevit_xxs_baseline` |

Phase 2 candidates selected:

- exp02: best Pareto trade-off.
- exp03: highest absolute validation macro-F1.

---

## Phase 2 - Optimize the winners

Each experiment isolates a single improvement on top of the prior. The two
winner tracks should use comparable recipes when practical.

### Track A - MobileNetV3-Large

| ID | Change vs prior | Hypothesis | Branch | Status |
| --- | --- | --- | --- | --- |
| exp05 | exp02 + augmentation | Augmentation improves generalization | `exp05_mobilenetv3large_augmentation` | complete: 0.9573 macro-F1 |
| exp06 | exp05 + weighted CrossEntropy | Weighted loss improves minority/fragile classes | `exp06_mobilenetv3large_weighted` | complete: 0.9670 macro-F1 |
| exp07 | exp06 + LR sweep | LR tuning may improve the current best MobileNet setup | `exp07_mobilenetv3large_lr_sweep` | complete elsewhere: best 0.9676 macro-F1, not meaningful vs exp06 |
| exp08 | exp06 + OneCycleLR | Test whether OneCycleLR beats the conservative exp06 scheduler | `exp08_mobilenetv3large_onecycle` | complete: 0.9746 macro-F1, current best MobileNet |
| exp09 | exp08 + OneCycle max_lr sweep | Tune OneCycleLR max_lr for the best MobileNet setup | `exp09_mobilenetv3large_onecycle_sweep` | complete: best 0.9779 macro-F1 |

LR sweep result: exp07 compared high/default/low LR recipes on the current best
MobileNetV3-Large setup. High LR was numerically best at 0.9676 macro-F1, but
only +0.0006 over exp06 and below the +0.003 meaningful-gain threshold. Use the
exp06 default LR recipe (`head_lr=3e-4`, `full_lr=5e-5`) as the conservative
base unless future reruns confirm high LR consistently.

OneCycleLR result: exp08 kept exp06's augmentation, weighted loss, dataset, and head warmup, then changed only the full fine-tuning scheduler to batch-level OneCycleLR after unfreezing. It reached 0.9746 macro-F1, a meaningful +0.0076 over exp06, and became the best MobileNetV3-Large configuration at that point.

OneCycle max_lr sweep result: exp09 kept exp08 fixed except `onecycle_max_lr`.
The best value was 1.5e-3 with 0.9779 macro-F1, 0.9763 accuracy, and 15.02 ms
CPU latency. This beats exp08 by +0.0033 macro-F1 and becomes the current best
MobileNetV3-Large configuration. Lower max_lr values underfit relative to exp08,
while 2e-3 dropped below the 1.5e-3 result.
### Track B - EfficientNet-B0

| ID | Change vs prior | Hypothesis | Branch | Status |
| --- | --- | --- | --- | --- |
| TBD | EfficientNet-B0 + augmentation | Same question as exp05 for EfficientNet-B0 | branch assigned after MobileNet track | planned |
| TBD | EfficientNet-B0 + weighted CrossEntropy | Same question as exp06 for EfficientNet-B0 | branch assigned after augmentation run | planned |
| TBD | EfficientNet-B0 + Dhan-Shomadhan eval/data decision | Test deployment-style robustness | branch assigned later | planned |
| TBD | EfficientNet-B0 + label smoothing | Optional calibration/accuracy check | branch assigned later | optional |

Stop a track early if it hits diminishing returns, for example less than +0.005
macro-F1 on a step, or starts to overfit hard despite augmentation.

Use Dhan-Shomadhan first as deployment-like validation/evaluation data. Do not
merge it into training until a domain gap is demonstrated or a deliberate new
dataset phase is started.

---

## Phase 3 - Finalize

Run on the single cross-track winner chosen from Phase 2.

| Step | Action |
| --- | --- |
| Final test eval | `python scripts/evaluate_checkpoint.py --split test experiments/<winner>` - one time only |
| Deployment-like eval | Evaluate reserved Dhan-Shomadhan field images; do not train on them unless a new dataset phase is declared |
| ONNX export | `experiments/<winner>/export_onnx.py`, verify shape and measure CPU latency |
| Model card | Write `MODEL_CARD.md` for HF Hub |
| HF Hub upload | Upload `model.onnx` and model card |
| Integration handoff | Point the separate `krishidoc` product repo at the new model artifact |

---

## Branch Convention

- `main` is for infrastructure only: `src/`, `data/`, `docs/`, `scripts/`, and canonical `results/`.
- `expNN_<model>_<variant>` branches own one experiment folder.
- If an experiment builds on the previous experiment in a track, branch from that previous experiment branch.
- `compare_models` is the aggregation branch for reports and multi-experiment comparison.

When creating a new experiment branch:

```bash
git checkout <previous-track-branch>
git checkout -b expNN_<model>_<variant>
# Update .gitignore to whitelist the new experiment folder.
# Scaffold config.yaml, train.py, expNN_README.md, and tmp log.
# Run training, update docs/results, commit, push.
```

---

## Design Notes

- Input size is 224x224 for all current model comparisons.
- Test set is locked until Phase 3.
- Primary metric is validation macro-F1.
- Accuracy, per-class F1/recall, CPU latency, checkpoint size, and curves are supporting metrics.
- Every experiment should auto-report params, MACs/FLOPs, checkpoint size, CPU/GPU inference latency, peak VRAM, and total training time in `metrics_val.json`.
- W&B is available but should stay disabled unless login/setup is confirmed.
- Reproducibility settings: `seed=42`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.
