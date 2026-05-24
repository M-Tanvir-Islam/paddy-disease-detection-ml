# krishidoc-ml

ML research repository for paddy disease classification. Produces one
artifact for the production `krishidoc` repo: `model.onnx`.

See [`docs/ML_RESEARCH_ARCHITECTURE.md`](docs/ML_RESEARCH_ARCHITECTURE.md)
for the full plan: goals, folder structure, experiment workflow, dataset
strategy, and integration handoff.

## Quick start

### Environment (one time)

Python 3.12 + CUDA 12.4 + PyTorch 2.6.0. Using Miniconda:

```powershell
conda create -n krishidoc_ml python=3.12 -y
conda activate krishidoc_ml

# CUDA-enabled PyTorch — must come from the cu124 index, not PyPI
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
wandb login                                    # optional
```

Verify GPU is visible to PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Train

```powershell
python data/prepare_kaggle.py                  # build stratified split CSV
python data/verify_dataset.py                  # sanity-check counts and paths

python experiments/exp01_mobilenetv3small_baseline/train.py
```

Each experiment auto-saves: best checkpoint, confusion matrix PNG, training
curves PNG, per-class metrics JSON, and a full `training.log` of console output.

## Experiment log

See [`results/experiment_log.md`](results/experiment_log.md) for the running
table of all experiments and their val macro-F1.
