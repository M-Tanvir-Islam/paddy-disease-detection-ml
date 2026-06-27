# Phase 2 Cross-Track Comparison

This report compares the best completed MobileNetV3-Large candidate against the best completed EfficientNet-B0 candidate. It is the decision document before Phase 3. The test set remains locked and was not evaluated for this comparison.

## Candidates

| Track | Representative | Branch | Recipe | Input | Val macro-F1 | Val Acc | CPU mean ms | CPU p95 ms | Params M | MACs G | Checkpoint MB | Peak VRAM MB | Best epoch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-Large | exp09 `maxlr_15e4` | `exp09_mobilenetv3large_onecycle_sweep` | aug + weighted CE + OneCycleLR max_lr 1.5e-3 | 224 | 0.9779 | 0.9763 | 15.02 | 17.55 | 4.215 | 0.234 | 17.07 | 1586.2 | 31 |
| EfficientNet-B0 | exp14 `maxlr_5e4` | `exp14_efficientnet_b0_onecycle_sweep` | aug + weighted CE + OneCycleLR max_lr 5e-4 | 224 | 0.9713 | 0.9744 | 24.01 | 30.39 | 4.020 | 0.414 | 16.38 | 2977.0 | 38 |

## Decision

Select **MobileNetV3-Large exp09 `maxlr_15e4`** as the current cross-track winner for Phase 3 preparation.

Reasons:

- Higher primary metric: +0.0066 validation macro-F1 over EfficientNet exp14.
- Faster CPU inference: 15.02 ms vs 24.01 ms mean latency.
- Lower compute: 0.234G MACs vs 0.414G MACs.
- Lower training VRAM in these runs: 1586 MB vs 2977 MB.
- Both models satisfy production size and latency constraints, so macro-F1 and latency decide the tie.

EfficientNet-B0 exp14 remains useful as the best EfficientNet reference, but it does not justify replacing MobileNetV3-Large.

## Per-Class F1 Comparison

| Class | MobileNet exp09 | EfficientNet exp14 | Winner | Delta MobileNet - EfficientNet |
| --- | ---: | ---: | --- | ---: |
| bacterial_leaf_blight | 0.9722 | 0.9444 | MobileNet | +0.0278 |
| bacterial_leaf_streak | 1.0000 | 0.9913 | MobileNet | +0.0087 |
| bacterial_panicle_blight | 0.9804 | 0.9709 | MobileNet | +0.0095 |
| blast | 0.9735 | 0.9697 | MobileNet | +0.0038 |
| brown_spot | 0.9754 | 0.9648 | MobileNet | +0.0106 |
| dead_heart | 0.9977 | 0.9883 | MobileNet | +0.0093 |
| downy_mildew | 0.9622 | 0.9412 | MobileNet | +0.0210 |
| hispa | 0.9617 | 0.9767 | EfficientNet | -0.0150 |
| normal | 0.9681 | 0.9812 | EfficientNet | -0.0131 |
| tungro | 0.9879 | 0.9847 | MobileNet | +0.0032 |

MobileNet wins 8 of 10 classes. EfficientNet is better for `hispa` and `normal`, but not by enough to offset MobileNet's stronger macro-F1, lower latency, and lower compute.

## What Is Prepared In This Branch

This `compare_models` branch now contains:

- Phase 1 bake-off evidence for exp01-exp04.
- MobileNet selected-track evidence from exp09 and rejected resolution check from exp10.
- EfficientNet selected-track evidence from exp14.
- Updated `results/experiment_log.md` with completed exp14 numbers.
- Handoff docs for MobileNet and EfficientNet decisions.

## Next Step

Prepare Phase 3 for MobileNetV3-Large exp09. Do not run the test set until the user explicitly confirms final evaluation.

Recommended Phase 3 order:

1. Confirm MobileNet exp09 `maxlr_15e4` is the final validation-selected candidate.
2. Run one locked test-set evaluation for that checkpoint only.
3. Run deployment-style Dhan-Shomadhan field validation as external/domain validation, not as training data unless a new dataset phase is declared.
4. Export the selected checkpoint to ONNX.
5. Verify ONNX output shape `(1, NUM_CLASSES)`, raw logits, CPU latency, preprocessing alignment, and file size below 50 MB.
6. Write the model card and upload `model.onnx` externally.