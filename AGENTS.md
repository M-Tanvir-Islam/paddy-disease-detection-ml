# Agent Rule Book

Shared instructions for AI agents working in this repository.

This is the canonical agent guide. Tool-specific files such as `CLAUDE.md`, Copilot instructions, or Cursor rules should point here instead of duplicating the full rule set.

## Repository Purpose

This is a standalone ML research repository for paddy disease classification. Its production output is one artifact: `model.onnx`, plus supporting model documentation.

Do not add FastAPI, React, database, RAG, deployment, or product application code here. Those belong in the separate `krishidoc` product repository.

## Read First

Before changing ML, experiment, data, evaluation, or export code, read:

1. `docs/ML_RESEARCH_ARCHITECTURE.md`
2. `docs/EXPERIMENT_PLAN.md`
3. `results/experiment_log.md`
4. The nearest prior experiment in the same track, when working on an experiment branch.

## Current Experiment Direction

Phase 1 bake-off is complete.

- `exp02` MobileNetV3-Large is the best Pareto trade-off.
- `exp03` EfficientNet-B0 has the highest validation macro-F1.
- Phase 2 optimizes both tracks with the same recipe: augmentation, weighted CrossEntropy, Dhan-Shomadhan field data, then optional label smoothing.

The cross-track winner proceeds to Phase 3 for final test evaluation and ONNX export.

## Experiment Discipline

- Change one variable per experiment.
- Copy from the nearest prior experiment in the same track and change only the intended variable.
- Keep hyperparameters in the experiment's `config.yaml`.
- Use the experiment's own `train.py` as the entrypoint.
- Do not silently change shared training behavior while running an experiment.
- If shared `src/` code changes, call that out clearly because it affects future comparisons.

## Metrics Discipline

- Primary ranking metric is validation macro-F1.
- Accuracy may be logged, but it must not drive model selection.
- `results/experiment_log.md` is the canonical table for meaningful completed experiment results.
- Do not put scratch runs, failed probes, accidental outputs, or non-canonical notes in `results/experiment_log.md`.
- Use a temporary log for branch-local and non-canonical notes: `results/tmp_logs/expNN_tmp_log.md`.
- Use the experiment README for the final branch-local summary: `experiments/expNN_<model>_<variant>/expNN_README.md`.
- Update `results/experiment_log.md` only with completed meaningful experiment rows, not every failed or debug run.
- If an accidental or exploratory test-like output exists, record it only in the matching tmp log and mark it clearly as non-canonical. Do not use it for model selection.

## Test Set Discipline

- The test set is locked until Phase 3.
- During Phase 2, all iteration must use validation data only.
- Do not run test-set evaluation just to preview performance.
- Final test evaluation happens once, only for the selected cross-track winner.

## Branch Discipline

- `main` is for infrastructure, shared code, docs, scripts, and canonical results. It should not track experiment folders.
- `expNN_<model>_<variant>` branches own one experiment folder.
- If an experiment builds on a previous experiment in the same track, create the new branch from that previous experiment branch, not from `main`.
- Example lineage: `exp05_mobilenetv3large_augment` -> `exp06_mobilenetv3large_weighted` -> `exp07_mobilenetv3large_dhan`.
- Each experiment branch `.gitignore` should whitelist only its own `experiments/expNN_*` folder.
- `compare_models` is for aggregated multi-experiment comparisons and phase reports.
- Use `compare_models`, not `main`, to collect selected experiment folders for comparison.
- Do not add unrelated experiment folders to an experiment branch.

## Experiment Documentation

Each completed experiment branch should include a short experiment summary file inside its experiment folder:

```text
experiments/expNN_<model>_<variant>/expNN_README.md
```

The file should cover:

- Hypothesis
- Changed variable
- Result
- Next step

Logging levels:

- `results/tmp_logs/expNN_tmp_log.md` is for branch-local scratch notes, partial runs, failed runs, command notes, and debugging observations.
- `experiments/expNN_<model>_<variant>/expNN_README.md` is for the final branch-local summary.
- `results/experiment_log.md` is for the clean canonical table used later for reports and research writing.

## Reproducibility

- Keep `seed=42` unless the experiment explicitly studies randomness.
- Keep deterministic CuDNN settings unless the experiment explicitly says otherwise.
- Keep input size fixed for fair comparisons unless the model experiment explicitly requires a different size.
- Do not modify train/validation/test split definitions casually.
- Do not train on validation or test images.

## Dataset And Artifact Rules

Do not commit:

- `datasets/`
- raw data
- generated dataset caches
- checkpoints
- `*.pt`
- `*.pth`
- `*.onnx`
- secrets or `.env`
- W&B local run folders

Model binaries should be hosted externally, such as on Hugging Face Hub, when the final model is ready.

## Class Index Safety

Treat `src/classes.py` as API-critical.

- Class order must match the agreed dataset folder order.
- Product inference depends on class-index alignment.
- Do not reorder, rename, remove, or insert classes casually.
- Any class-list change requires an explicit note and product handoff awareness.

## Production Constraints

The final candidate must satisfy:

- ONNX model size below 50 MB.
- CPU inference latency below 500 ms.
- Validation macro-F1 above 0.85, with target at or above 0.95.
- ONNX output shape `(1, NUM_CLASSES)`.
- Raw logits output; softmax is applied server-side.

Before declaring a winner, verify:

- ONNX export succeeds.
- ONNX Runtime CPU inference succeeds.
- Output shape and class count are correct.
- CPU latency and file size are measured.
- Inference preprocessing matches training validation preprocessing.

## Weights And Biases

- Offline or disabled W&B mode is safe.
- Do not enable `wandb_mode: "online"` unless login/setup is confirmed.
- Do not run commands that may hang waiting for W&B login.

## Agent Behavior

- Prefer small, scoped changes that follow the existing repository patterns.
- Do not run long training jobs unless the user explicitly asks.
- Do not touch the separate `krishidoc` product repo from this repository.
- Do not declare an experiment complete without metrics and documentation.
- If a requested action conflicts with this rule book, explain the conflict before making changes.
