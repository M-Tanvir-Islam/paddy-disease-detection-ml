# Experiment Log

Running table of every merged experiment. One row per experiment. Update
after each run; primary metric is **val macro-F1**.

Do not record test scores here — only the final winning experiment evaluates
on the test set. For the full Phase 1 comparison report (per-class breakdown,
training curves, cost trade-offs) see
[`phase1_bakeoff_report.md`](phase1_bakeoff_report.md).

## Phase 1 — Bake-off (vanilla baselines, identical pipeline)

| ID          | Model               | Augment | Weighted Loss | Val macro-F1 | Val Acc    | Params (M) | CPU mean (ms) | Notes                                           |
| ----------- | ------------------- | ------- | ------------- | ------------ | ---------- | ---------- | ------------- | ----------------------------------------------- |
| exp01       | MobileNetV3-Small   | No      | No            | 0.9251       | 0.9347     | 1.528      | 8.28          | baseline; smallest + fastest                    |
| exp02       | MobileNetV3-Large   | No      | No            | 0.9518       | 0.9558     | 4.215      | 15.54         | best Pareto trade-off                           |
| **exp03** ★ | **EfficientNet-B0** | No      | No            | **0.9609**   | **0.9641** | 4.020      | 25.79         | **highest val macro-F1**                        |
| exp04       | MobileViT-XXS       | No      | No            | 0.9227       | 0.9321     | 0.954      | 21.57         | transformer; least overfit; smallest checkpoint |

★ Phase 1 winner. All four pass the production budget (ONNX < 50 MB, CPU latency < 500 ms).

## Phase 2 — Optimize the winners (parallel tracks)

Running both top Phase 1 candidates through the same optimization recipe.
The cross-track winner of Phase 2 proceeds to Phase 3.

### Track A — MobileNetV3-Large (built on exp02 = 0.9518)

| ID    | Change                       | Val macro-F1 | Δ vs prior | Notes |
| ----- | ---------------------------- | ------------ | ---------- | ----- |
| exp05 | + augmentation               | —            | —          | —     |
| exp06 | + weighted CE                | —            | —          | —     |
| exp07 | + Dhan-Shomadhan field       | —            | —          | —     |
| exp08 | + label smoothing (optional) | —            | —          | —     |

### Track B — EfficientNet-B0 (built on exp03 = 0.9609)

| ID    | Change                       | Val macro-F1 | Δ vs prior | Notes |
| ----- | ---------------------------- | ------------ | ---------- | ----- |
| exp09 | + augmentation               | —            | —          | —     |
| exp10 | + weighted CE                | —            | —          | —     |
| exp11 | + Dhan-Shomadhan field       | —            | —          | —     |
| exp12 | + label smoothing (optional) | —            | —          | —     |

## Phase 3 — Final

_Test-set evaluation, ONNX export, model card. One row only._

| Step             | Result |
| ---------------- | ------ |
| Test macro-F1    | —      |
| ONNX size        | —      |
| ONNX CPU latency | —      |
| HF Hub URL       | —      |
