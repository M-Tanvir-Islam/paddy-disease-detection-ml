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
| exp03 | EfficientNet-B0 | torchvision | 4.020 | 0.9609 | 25.79 | `exp03_efficientnet_b0` |
| exp04 | MobileViT-XXS | timm | 0.954 | 0.9227 | 21.57 | `exp04_mobilevit_xxs_baseline` |

Phase 2 candidates selected:

- exp02: best Pareto trade-off.
- exp03: highest absolute validation macro-F1 baseline.

---

## Phase 2 - Optimize The Winners

Each experiment isolates a single improvement on top of the prior. The two
winner tracks should use comparable recipes when practical.

### Track A - MobileNetV3-Large - CURRENT BEST MOBILE NET

| ID | Change vs prior | Hypothesis | Branch | Status |
| --- | --- | --- | --- | --- |
| exp05 | exp02 + augmentation | Augmentation improves generalization | `exp05_mobilenetv3large_augmentation` | complete: 0.9573 macro-F1 |
| exp06 | exp05 + weighted CrossEntropy | Weighted loss improves minority/fragile classes | `exp06_mobilenetv3large_weighted` | complete: 0.9670 macro-F1 |
| exp07 | exp06 + LR sweep | LR tuning may improve the current best MobileNet setup | `exp07_mobilenetv3large_lr_sweep` | complete: best 0.9676 macro-F1, not meaningful vs exp06 |
| exp08 | exp06 + OneCycleLR | Test whether OneCycleLR beats the conservative scheduler | `exp08_mobilenetv3large_onecycle` | complete: 0.9746 macro-F1 |
| exp09 | exp08 + OneCycle max_lr sweep | Tune OneCycleLR max_lr | `exp09_mobilenetv3large_onecycle_sweep` | complete: best 0.9779 macro-F1 |
| exp10 | exp09 + input resolution sweep | Test whether 256 or 288 input improves small disease cues | `exp10_mobilenetv3large_resolution_sweep` | complete: 256 underperformed; keep 224 |

MobileNet conclusion: keep exp09 `maxlr_15e4` as the best MobileNetV3-Large
configuration: 224 input, augmentation, weighted CrossEntropy, OneCycleLR
`max_lr=1.5e-3`, validation macro-F1 0.9779.

### Track B - EfficientNet-B0 - COMPLETE

Base branch for this track:

```text
exp03_efficientnet_b0
```

Do not branch EfficientNet experiments from the MobileNet branches. The first
EfficientNet optimization branch should start from `exp03_efficientnet_b0`.
After that, each new EfficientNet branch should start from the nearest prior
EfficientNet branch.

| ID | Change vs prior | Hypothesis | Branch | Status |
| --- | --- | --- | --- | --- |
| exp11 | exp03 + augmentation | Test whether the MobileNet augmentation recipe also improves EfficientNet-B0 | `exp11_efficientnet_b0_augmentation` | complete: 0.9469 macro-F1; under exp03 |
| exp12 | exp11 + weighted CrossEntropy | Test whether weighted loss improves fragile/minority classes on top of augmentation | `exp12_efficientnet_b0_weighted` | complete: 0.9329 macro-F1; under exp11 and exp03 |
| exp13 | exp12 + OneCycleLR | Test whether OneCycleLR improves EfficientNet-B0 fine-tuning | `exp13_efficientnet_b0_onecycle` | complete: 0.9701 macro-F1; best EfficientNet so far |
| exp14 | exp13 + OneCycle max_lr sweep | Tune OneCycleLR max_lr for the best EfficientNet-B0 setup | `exp14_efficientnet_b0_onecycle_sweep` | complete: best 0.9713 macro-F1 at max_lr 5e-4 |

EfficientNet augmentation result: exp11 added the MobileNet train-only augmentation recipe to exp03 and dropped to 0.9469 macro-F1, -0.0140 versus the 0.9609 exp03 baseline. Continue to exp12 only to test whether weighted loss recovers weak-class behavior on top of augmentation; if exp12 also underperforms, consider lighter EfficientNet-specific augmentation or stopping the track.

EfficientNet weighted-loss result: exp12 added weighted CrossEntropy on top of exp11 and dropped to 0.9329 macro-F1, -0.0140 versus exp11 and -0.0280 versus exp03. Continue to exp13 only as a scheduler rescue/check. If exp13 also underperforms exp03, stop this EfficientNet recipe or design a lighter EfficientNet-specific augmentation experiment.

EfficientNet OneCycle result: exp13 replaced the conservative full fine-tuning scheduler with OneCycleLR at max_lr 1e-3 and reached 0.9701 macro-F1, +0.0372 versus exp12 and +0.0092 versus exp03. exp14 then swept OneCycle max_lr and selected 5e-4 with 0.9713 macro-F1, 0.9744 accuracy, and 24.01 ms CPU latency. This is the best EfficientNet result, but it still trails MobileNet exp09.

EfficientNet control rules:

- Primary metric: validation macro-F1.
- Do not use the test set during this track.
- Keep input size 224 unless explicitly running a later resolution experiment.
- Keep seed 42 unless the experiment studies randomness.
- Change one variable per experiment.
- Update `results/experiment_log.md` only after meaningful completed runs.
- Use `results/tmp_logs/expNN_tmp_log.md` for scratch notes, partial runs, failed runs, and command notes.

EfficientNet conclusion: use exp14 `maxlr_5e4` as the EfficientNet representative for cross-track comparison. Do not continue EfficientNet tuning unless a new hypothesis is explicitly approved, because MobileNet exp09 is currently better on macro-F1 and CPU latency.

---

## Phase 3 - Finalize

Run on the single cross-track winner chosen from Phase 2. Current validation evidence favors MobileNetV3-Large exp09 over EfficientNet-B0 exp14, but final selection should be documented in the comparison branch before any test-set evaluation.

| Step | Action |
| --- | --- |
| Cross-track comparison | Compare best MobileNet and best EfficientNet by validation macro-F1, weak-class F1, CPU latency, and checkpoint size |
| Final test eval | `python scripts/evaluate_checkpoint.py --split test experiments/<winner>` - one time only |
| Deployment-like eval | Evaluate reserved Dhan-Shomadhan field images; do not train on them unless a new dataset phase is declared |
| ONNX export | Export winner to ONNX, verify output shape `(1, NUM_CLASSES)`, raw logits, file size, and CPU latency |
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

- Input size is 224x224 through the selected MobileNet method.
- Test set is locked until Phase 3.
- Primary metric is validation macro-F1.
- Accuracy, per-class F1/recall, CPU latency, checkpoint size, and curves are supporting metrics.
- Every experiment should auto-report params, MACs/FLOPs, checkpoint size, CPU/GPU inference latency, peak VRAM, and total training time in `metrics_val.json`.
- W&B is available but should stay disabled unless login/setup is confirmed.
- Reproducibility settings: `seed=42`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.