# Claude Code — repository context

ML research repository for paddy (rice) disease classification. Produces
one artifact (`model.onnx`) consumed by a separate FastAPI backend in the
`krishidoc` product repo.

## Read these first

1. **`docs/ML_RESEARCH_ARCHITECTURE.md`** — system design, folder structure,
   ML conventions, integration handoff
2. **`docs/EXPERIMENT_PLAN.md`** — full campaign plan (current phase,
   experiment numbering, branch convention, design notes)
3. **`results/experiment_log.md`** — running table of all experiments
   with val macro-F1 results
4. **`results/phase1_bakeoff_report.md`** _(only on `compare_models` branch)_
   — Phase 1 comparison report with per-class breakdown, training curves,
   cost trade-offs

## Current state

- **Phase 1 (bake-off) — COMPLETE.** Four model baselines trained.
  - exp01 MobileNetV3-Small: 0.9251
  - exp02 MobileNetV3-Large: 0.9518
  - exp03 EfficientNet-B0: **0.9609** (highest)
  - exp04 MobileViT-XXS: 0.9227
- **Phase 2 plan: two parallel tracks** — optimize exp02 (Pareto winner)
  AND exp03 (accuracy winner) with the same recipe:
  augmentation → weighted CE → Dhan-Shomadhan dataset → label smoothing.
- **Phase 3 (finalize)** — test-set eval + ONNX export on the cross-track winner.

## Branch convention

- `main` — infrastructure only (`src/`, `data/`, `docs/`, `scripts/`, `results/`).
  No `experiments/` folders are tracked here.
- `expNN_<model>_<variant>` — one branch per experiment. The branch's
  `.gitignore` whitelists ONLY its own experiment folder.
- `compare_models` — aggregated multi-experiment view. `.gitignore` whitelists
  every experiment folder pulled into it.

When checking out an experiment branch, the experiment folder reappears in
the working tree. When switching to `main`, those folders disappear (still
tracked on their respective branches — not data loss).

## Environment

Conda env: `krishidoc_ml` (Python 3.12, PyTorch 2.6.0+cu124).

```powershell
conda activate krishidoc_ml
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce RTX 3060
```

If `conda` isn't recognized in PowerShell, run once:
`& 'C:\ProgramData\Miniconda3\Scripts\conda.exe' init powershell` then
restart the shell.

## Common commands

```powershell
# Activate the env (every new shell)
conda activate krishidoc_ml

# Train an experiment (always pass through the experiment's train.py)
python experiments/exp05_mobilenetv3large_augment/train.py

# Re-evaluate a saved checkpoint without retraining (only changes metrics,
# not the model; useful when new metrics were added to evaluate.py)
python scripts/evaluate_checkpoint.py experiments/exp05_mobilenetv3large_augment

# Regenerate the Phase 1 bake-off report (only on compare_models branch)
python scripts/generate_bakeoff_report.py

# Sanity-check the dataset split
python data/verify_dataset.py
```

## How to start a new experiment

1. `git checkout main`
2. `git checkout -b expNN_<model>_<variant>` (e.g. `exp05_mobilenetv3large_augment`)
3. Update `.gitignore` to whitelist the new experiment folder:
   ```gitignore
   experiments/*
   !experiments/expNN_<model>_<variant>/
   !experiments/expNN_<model>_<variant>/**
   ```
4. Scaffold `experiments/expNN_<model>_<variant>/`:
   - `config.yaml` — based on the prior experiment in the same track, change
     only the variable being tested
   - `train.py` — thin wrapper, same shape as other experiments
   - `README.md` — fill in after training completes
5. Run `python experiments/expNN_<model>_<variant>/train.py`
6. Update the README with results
7. Commit + push the branch

## Production constraints (do not violate)

- ONNX model size < 50 MB
- CPU inference latency < 500 ms (HF Spaces CPU Basic tier)
- Val macro-F1 > 0.85 (target ≥ 0.95 for a paper-grade result)

All Phase 1 candidates were comfortably within these constraints. Phase 2
optimizations should not push CPU latency past 100 ms.

## Test set discipline

Test set is **locked** until Phase 3. Do not run inference on it during
Phase 2 — every preview makes the final test number less trustworthy as
an unbiased estimate. Use the validation split for all iteration.

## What NOT to do

- Don't add experiments/ folders to `main` (they belong on branches).
- Don't commit `*.pt` / `*.onnx` / `*.pth` files (gitignored — host on HF Hub).
- Don't commit the `datasets/` folder (too large — download separately).
- Don't change `src/classes.py` ordering — API indices in the product repo depend on it.
- Don't enable `wandb_mode: "online"` without running `wandb login` first (will hang).
