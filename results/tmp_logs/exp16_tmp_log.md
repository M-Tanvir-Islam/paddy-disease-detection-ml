# exp16 temporary log

## Setup

- Parent experiment: exp09 winning `maxlr_15e4` configuration
- Changed variable: `data.split_csv` only
- New split: `data/splits/kaggle_grouped_v2.csv`
- Initialization: fresh ImageNet weights; exp09 checkpoint is not reused
- Test evaluation: locked and not run

## Runs

### Completed run

- Validation macro-F1: 0.9800158992
- Validation accuracy: 0.9813384813
- Best epoch: 38 of 40
- Training time: 2,931.4 seconds
- Peak VRAM: 1,586.2 MB
- CPU latency mean / p95: 21.15 / 36.59 ms
- GPU latency mean / p95: 11.19 / 12.26 ms
- Checkpoint size: 17.07 MB
- Test set: not evaluated

Observation: validation performance dropped after the backbone was unfrozen
and the OneCycle learning rate rose, then recovered steadily. The final six
epochs stayed between 0.9789 and 0.9800 macro-F1.

### Pre-test checks

- Standalone checkpoint reload reproduced validation macro-F1 0.9800 and
  accuracy 0.9813.
- Reporting scripts now force the headless Matplotlib `Agg` backend after the
  dry run exposed a missing-Tk error. Predictions were unaffected.
- Candidate checkpoint and data/config SHA-256 hashes were frozen in
  `results/pretest_candidate_manifest.json`.
- Dhan–Kaggle audit: zero exact matches and zero accepted near duplicates.
- Dhan compatible evaluation: 606 images across blast, brown spot, and tungro.
- Dhan overall full-model accuracy: 0.2162.
- Dhan field-background full-model accuracy: 0.3317.
- Dhan white-background full-model accuracy: 0.1597.
- Kaggle test set: not evaluated.
