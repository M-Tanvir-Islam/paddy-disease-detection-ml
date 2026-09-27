# Paddy Disease Detection: ML Research

**Status: active research · classification experiments in progress · no production model released**

This repository develops a lightweight image classifier for paddy diseases. It is the machine learning research component of **KrishiDoc**, a proposed mobile-first tool for farmers in Bangladesh. The intended handoff is a validated ONNX classifier and a documented class mapping. The web app, disease localization, severity estimation, and retrieval-assisted advice are separate planned components; they are **not implemented by this repository**.

The immediate research question is whether a model that performs well on a curated paddy image dataset also works on the types of photographs farmers may submit. Current experiments show strong in-dataset validation performance and a substantial drop on a separate field-image dataset. Closing that gap is the main focus of ongoing work.

## What has been done

- Built a 10-class classification pipeline for the [Paddy Disease Classification 2022 dataset](https://www.kaggle.com/competitions/paddy-disease-classification/data), using validation **macro-F1** as the model-selection metric.
- Compared MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, and MobileViT-XXS under a shared baseline protocol.
- Studied augmentation, class-weighted loss, learning-rate scheduling, and input resolution across follow-up experiments. Training code and reports live on experiment branches; `main` contains shared code, documentation, and the canonical experiment log.
- Audited duplicate images and retrained MobileNetV3-Large with a group-aware split that keeps accepted duplicate groups within one partition.
- Evaluated the group-aware checkpoint on compatible images from the independent Dhan-Shomadhan collection to probe transfer to a different image source.

## Selected experimental results

| Experiment | Model / change | Validation macro-F1 | Mean CPU latency* |
| --- | --- | ---: | ---: |
| [exp01](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp01_mobilenetv3small) | MobileNetV3-Small baseline | 0.9251 | 8.28 ms |
| [exp02](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp02_mobilenetv3large) | MobileNetV3-Large baseline | 0.9518 | 15.54 ms |
| [exp03](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp03_efficientnet_b0) | EfficientNet-B0 baseline | 0.9609 | 25.79 ms |
| [exp09](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp09_mobilenetv3large_onecycle_sweep) | MobileNetV3-Large; augmentation, weighted loss, OneCycleLR sweep | 0.9779 | 15.02 ms |
| [exp14](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp14_efficientnet_b0_onecycle_sweep) | EfficientNet-B0; optimized OneCycleLR | 0.9713 | 24.01 ms |
| [exp16](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/exp16_mobilenetv3large_grouped_split) | MobileNetV3-Large; fresh retrain on group-aware split | **0.9800** | 21.15 ms |

**Interpretation:** These are *validation* results, not final test results. Exp16 uses a revised, leakage-resistant split; its 0.9800 must not be interpreted as a directly comparable improvement over exp09's 0.9779. The grouped split contains 10,349 usable images (7,241 train / 1,554 validation / 1,554 locked test), with 58 images from cross-label duplicate groups quarantined. The locked Kaggle test set has not been used for final evaluation. See the [experiment log](results/experiment_log.md), the [cross-track comparison log](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/blob/compare_models/results/experiment_log.md), and the [exp16 report](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/blob/exp16_mobilenetv3large_grouped_split/experiments/exp16_mobilenetv3large_grouped_split/exp16_README.md) for protocols and further results.

\* Recorded batch-one CPU timings from the experiment reports; they are not end-to-end phone or server response times and will vary by hardware.

### Cross-dataset finding

The exp16 checkpoint was evaluated, without retraining, on **606 Dhan-Shomadhan images** whose labels map to three supported classes: blast, brown spot, and tungro. It reached **0.2162 accuracy** and **0.2777 supported-class macro-F1** overall. On the 199 field-background images, accuracy was **0.3317**. A documented capture-quality proxy did not resolve the gap. These results indicate that strong Kaggle validation scores do **not** establish reliability on images from a different source. The Dhan images have already been used for development analysis and are not an untouched final external test set. Details are in the [exp16 report](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/blob/exp16_mobilenetv3large_grouped_split/experiments/exp16_mobilenetv3large_grouped_split/exp16_README.md).

## Data and evaluation protocol

The current classifier predicts 10 labels: bacterial leaf blight, bacterial leaf streak, bacterial panicle blight, blast, brown spot, dead heart, downy mildew, hispa, normal, and tungro. The exact class order is defined in [`src/classes.py`](src/classes.py).

The dataset images are not stored in this repository. Preparation scripts expect the Kaggle images in `datasets/paddy-disease-classification/train_images/<class>/`. The first experiments used a stratified train/validation/test split with seed 42; exp16 introduced a separate group-aware split after a duplicate audit. Model selection uses validation macro-F1; the final test partition remains locked. See [`data/README.md`](data/README.md) and the [group-aware experiment report](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/blob/exp16_mobilenetv3large_grouped_split/experiments/exp16_mobilenetv3large_grouped_split/exp16_README.md).

## Repository guide

| Location | Purpose |
| --- | --- |
| [`src/`](src/) | Shared dataset, training, augmentation, and evaluation utilities |
| [`data/`](data/) | Dataset preparation and split definitions; raw images are kept out of Git |
| [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) | Experiment design and decision record |
| [`results/experiment_log.md`](results/experiment_log.md) | Results committed to `main` (currently through exp10) |
| [Experiment branches](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/branches) | Per-experiment configuration, training entrypoint, and reports |
| [`compare_models` branch](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/tree/compare_models) | Aggregated comparison of the earlier model tracks |

The attached [system architecture](docs/ML_RESEARCH_ARCHITECTURE.md) describes the wider KrishiDoc plan. Some planning sections predate the recorded experiments; experiment reports and result files are the source for *completed* work.

## Getting started

This repository currently targets **Python 3.12** and records experiments using **PyTorch 2.6.0**. To set up an environment, clone the repository and follow the install notes in [`requirements.txt`](requirements.txt). The original setup uses the CUDA 12.4 PyTorch wheels; choose a PyTorch build appropriate for your own hardware.

```bash
git clone https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml.git
cd paddy-disease-detection-ml
```

Download the [Paddy Disease Classification 2022 dataset](https://www.kaggle.com/competitions/paddy-disease-classification/data) under `datasets/paddy-disease-classification/` as described above. Dataset files and trained weights are not included here. For a specific run, switch to its experiment branch and follow its experiment README; for example:

```bash
git switch exp16_mobilenetv3large_grouped_split
```

The experiment code is intentionally stored on its branch rather than in `main`. Follow the [exp16 run instructions](https://github.com/M-Tanvir-Islam/paddy-disease-detection-ml/blob/exp16_mobilenetv3large_grouped_split/experiments/exp16_mobilenetv3large_grouped_split/exp16_README.md). That branch uses a prepared group-aware split; consult its data preparation scripts and split report before attempting to reproduce the run. The historical split CSV contains source-machine paths, so regenerate or adapt paths on a new machine.

## Next steps

1. Investigate the Kaggle-to-field-image domain gap with group-separated field data and report Kaggle and field validation separately.
2. Choose a final classifier only after checking generalization, class-wise errors, model size, and CPU inference.
3. Run the locked test evaluation under a documented final protocol, then export and verify an ONNX model with its preprocessing and class order.
4. Integrate the validated classifier into the separate KrishiDoc product. Disease localization, severity estimation, and guidance remain later-stage work.

**Research status:** The figures above document experimental progress, not a deployed diagnostic or pesticide-recommendation system.
