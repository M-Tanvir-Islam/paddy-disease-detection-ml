# EfficientNet-B0 Experiment Handoff

This document tells future agents exactly how to continue the EfficientNet-B0
track. It exists to prevent branch-number confusion and to keep the same
experiment discipline used for the MobileNetV3-Large track.

## Current State

EfficientNet-B0 tuning is complete through exp14.

```text
Best EfficientNet experiment: exp14
Branch:                       exp14_efficientnet_b0_onecycle_sweep
Model:                        EfficientNet-B0
Augmentation:                 yes
Weighted loss:                yes
Input size:                   224
LR config:                    OneCycleLR max_lr 5e-4
Val macro-F1:                 0.9713
Val accuracy:                 0.9744
CPU latency:                  24.01 ms
```

MobileNetV3-Large has already been optimized through exp10. The best MobileNet
method is exp09 `maxlr_15e4` with 0.9779 validation macro-F1 and 15.02 ms CPU
latency. EfficientNet-B0 exp14 is the EfficientNet representative, but it does
not beat the MobileNet candidate.

## Important Branch Rule

EfficientNet work started from:

```powershell
git checkout exp03_efficientnet_b0
```

Do not create EfficientNet branches from MobileNet branches. This completed track used the nearest prior EfficientNet branch for each follow-up experiment.

Correct lineage:

```text
exp03_efficientnet_b0
  -> exp11_efficientnet_b0_augmentation
      -> exp12_efficientnet_b0_weighted
          -> exp13_efficientnet_b0_onecycle
              -> exp14_efficientnet_b0_onecycle_sweep
```

## Completed Experiments

### exp11_efficientnet_b0_augmentation

Question:

```text
Does the MobileNet augmentation recipe improve EfficientNet-B0?
```

Change only:

```text
data.augmentation: false -> true
```

Keep:

```text
weighted_loss: false
input_size: 224
seed: 42
batch_size: use the exp03 baseline unless there is a real memory issue
LR config: head 3e-4 / full 5e-5
```

### exp12_efficientnet_b0_weighted

Branch from exp11.

Question:

```text
Does weighted CrossEntropy add value on top of augmentation?
```

Change only:

```text
data.weighted_loss: false -> true
```

Keep the exp11 augmentation and all other settings fixed.

### exp13_efficientnet_b0_onecycle

Branch from exp12.

Question:

```text
Does OneCycleLR improve EfficientNet-B0 fine-tuning?
```

Change only the full fine-tuning scheduler after warmup.
Use MobileNet exp08 as the implementation reference, but adapt the experiment
folder and output names for EfficientNet-B0.

Start with a conservative OneCycle value. A reasonable first value is:

```text
onecycle_max_lr: 1e-3
pct_start: 0.1
div_factor: 25.0
final_div_factor: 10000.0
```

### exp14_efficientnet_b0_onecycle_sweep

Branch from exp13.

Question:

```text
What OneCycle max_lr works best for EfficientNet-B0?
```

Use the exp09 MobileNet sweep pattern as a template. Good starting candidates:

```text
5e-4
1e-3
1.5e-3
```

Completed result: `5e-4` was best with 0.9713 macro-F1. `1e-3` reached 0.9633 in the repeat, and `1.5e-3` reached 0.9697. Do not add `2e-3` unless a new EfficientNet-specific hypothesis is approved.

## Experiment Rules

- Primary metric is validation macro-F1.
- Accuracy is supporting only; do not select by accuracy alone.
- Test set remains locked until the final cross-track winner.
- Keep input size 224 unless the experiment is explicitly about resolution.
- Keep seed 42 unless testing randomness.
- Change one variable per experiment.
- Keep hyperparameters in the experiment config.
- Use the experiment's own `train.py` as entrypoint.
- Use `results/tmp_logs/expNN_tmp_log.md` for scratch notes and failed runs.
- Use `results/experiment_log.md` only for meaningful completed rows.
- Each completed experiment folder should include `expNN_README.md`.

## What To Copy From MobileNet Experiments

Useful templates:

```text
exp05 MobileNet -> exp11 EfficientNet augmentation structure
exp06 MobileNet -> exp12 weighted loss structure
exp08 MobileNet -> exp13 OneCycleLR implementation
exp09 MobileNet -> exp14 max_lr sweep layout
```

Do not copy MobileNet model construction. EfficientNet-B0 needs:

```python
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
```

## Commands Pattern

Every experiment should be runnable from repo root, for example:

```powershell
conda activate krishidoc_ml
python experiments/exp11_efficientnet_b0_augmentation/train.py
```

For sweep experiments, use config arguments like the MobileNet exp09 pattern:

```powershell
python experiments/exp14_efficientnet_b0_onecycle_sweep/train.py --config configs/maxlr_1e3.yaml
```

## Comparison Target

The EfficientNet-B0 representative must be compared against:

```text
MobileNet exp09 maxlr_15e4
Val macro-F1: 0.9779
CPU latency: 15.02 ms
Checkpoint: 17.07 MB
Input: 224
```

Current conclusion: EfficientNet-B0 exp14 does not beat MobileNet exp09 on validation macro-F1 or CPU latency. Use exp14 only as the EfficientNet representative in comparison reports.