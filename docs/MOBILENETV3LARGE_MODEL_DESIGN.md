# MobileNetV3-Large Model Design And Learning Guide

This document explains the selected MobileNetV3-Large model design for the
KrishiDoc paddy disease classifier. It is written as both a project record and a
learning guide, so every major decision is explained in plain language.

Current preferred MobileNet method:

```text
Model:              MobileNetV3-Large, ImageNet pretrained
Input size:         224x224 RGB
Classes:            10 Kaggle Paddy 2022 classes
Training data:      Kaggle Paddy 2022 train split
Selection data:     validation split only
Test data:          locked until final model selection
Augmentation:       enabled for train only
Loss:               weighted CrossEntropyLoss
Fine-tuning:        5-epoch frozen-head warmup, then full fine-tuning
Scheduler:          OneCycleLR
OneCycle max_lr:    1.5e-3
Batch size:         64
Epochs:             40
Seed:               42
Best branch:        exp09_mobilenetv3large_onecycle_sweep
Best variant:       maxlr_15e4
Best val macro-F1:  0.9779
Validation acc:     0.9763
CPU latency:        15.02 ms mean, 17.55 ms p95
Checkpoint size:    17.07 MB
```

The selected MobileNet method is not the final production model yet. It is the
best MobileNetV3-Large configuration found so far. The final project winner
should still be compared against the EfficientNet-B0 track, then evaluated once
on the locked test set and deployment-like Dhan-Shomadhan data.

---

## 1. What Problem Are We Solving?

This is an image classification problem.

Input:

```text
A paddy leaf image
```

Output:

```text
One disease class from 10 possible classes
```

Classes, in fixed index order:

```text
0 bacterial_leaf_blight
1 bacterial_leaf_streak
2 bacterial_panicle_blight
3 blast
4 brown_spot
5 dead_heart
6 downy_mildew
7 hispa
8 normal
9 tungro
```

This class order is API-critical. The model output is a vector of 10 raw scores
called logits. Index 0 always means `bacterial_leaf_blight`, index 1 always
means `bacterial_leaf_streak`, and so on. If this order changes without updating
the product API, the app can show the wrong disease even when the model is
technically predicting correctly.

---

## 2. The Final MobileNet Architecture

### 2.1 Backbone

The selected backbone is `torchvision.models.mobilenet_v3_large` with ImageNet
pretrained weights.

A backbone is the feature extractor part of a neural network. It learns visual
patterns such as edges, textures, colors, shapes, spots, and object parts.

ImageNet pretraining means the model already learned useful general visual
features from a very large image dataset before seeing our paddy disease data.
This matters because our dataset is not huge. Starting from pretrained weights
usually trains faster and generalizes better than starting from random weights.

### 2.2 Classification Head

MobileNetV3-Large originally predicts 1000 ImageNet classes. We replace its last
classification layer so it predicts 10 paddy classes instead.

Conceptually:

```python
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
```

The backbone keeps useful visual features. The final layer is replaced because
our label space is different.

### 2.3 Input And Output

Input tensor shape during training and inference:

```text
(batch_size, 3, 224, 224)
```

For one image during production inference:

```text
(1, 3, 224, 224)
```

Output shape:

```text
(batch_size, 10)
```

For one image:

```text
(1, 10)
```

The output is raw logits, not probabilities. Softmax should be applied outside
the model when probabilities are needed.

---

## 3. Why MobileNetV3-Large?

The project has three constraints:

1. High validation macro-F1
2. Low CPU latency
3. Small enough model artifact for practical deployment

Phase 1 compared several model families under the same baseline conditions.

| Experiment | Model | Val macro-F1 | CPU ms | Decision |
| --- | --- | ---: | ---: | --- |
| exp01 | MobileNetV3-Small | 0.9251 | 8.28 | Very fast, but lower accuracy |
| exp02 | MobileNetV3-Large | 0.9518 | 15.54 | Best speed/accuracy trade-off |
| exp03 | EfficientNet-B0 | 0.9609 | 25.79 | Highest baseline F1, slower |
| exp04 | MobileViT-XXS | 0.9227 | 21.57 | Lower accuracy, not worth continuing |

MobileNetV3-Large was chosen for the MobileNet track because it gave much better
accuracy than MobileNetV3-Small while staying very fast on CPU. EfficientNet-B0
was also kept alive because it had the highest Phase 1 macro-F1, but it is a
separate track.

The practical decision:

```text
MobileNetV3-Large is the best Pareto MobileNet candidate.
EfficientNet-B0 still deserves comparison later.
```

Pareto trade-off means no other tested model was clearly better on both accuracy
and speed at the same time.

---

## 4. Dataset And Splits

The current experiments use Kaggle Paddy 2022 through `data/splits/kaggle.csv`.
The CSV stores each image path, class, and split.

Training uses:

```text
split = train
```

Model selection uses:

```text
split = val
```

The test split is locked until the final winner.

### Why Not Use Test During Experiments?

Validation is for choosing models and hyperparameters. Test is for the final
unbiased estimate.

If you check test performance after every idea, the test set becomes part of the
training decision process. That makes the final test score too optimistic. It is
like practicing with the exam answer key.

Correct discipline:

```text
Train set: learn model weights
Validation set: choose experiments and hyperparameters
Test set: final one-time evaluation of selected winner
Deployment-like field set: real-world robustness check
```

---

## 5. Preprocessing And Augmentation

### 5.1 Evaluation Preprocessing

Validation and test use deterministic evaluation preprocessing:

```python
A.Resize(224, 224)
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
ToTensorV2()
```

This is not augmentation. It is required preprocessing.

Resize makes all images the same size. Neural networks process tensors of fixed
shape inside a batch.

Normalize converts pixel values into the scale expected by ImageNet-pretrained
models. ImageNet mean and standard deviation are used because the backbone was
originally trained with that normalization.

ToTensorV2 converts the image into a PyTorch tensor with channel-first shape:

```text
HWC image -> CHW tensor
```

### 5.2 Training Augmentation

Training uses random augmentation:

```python
A.Resize(224, 224)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.3)
A.RandomRotate90(p=0.5)
A.RandomBrightnessContrast(0.2, 0.2, p=0.6)
A.HueSaturationValue(10, 25, 10, p=0.5)
A.GaussNoise(std_range=(0.012, 0.028), p=0.3)
A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(1, 28), hole_width_range=(1, 28), p=0.3)
A.Normalize(...)
ToTensorV2()
```

Augmentation means changing training images randomly while preserving their
label. The goal is to teach the model that the disease class should not change
just because the image is flipped, rotated, slightly brighter, slightly noisier,
or partly occluded.

Important rule:

```text
Augmentation is for training only.
Validation and test must be deterministic.
```

If validation or test are randomly augmented, scores become noisy and unfair.
Our code now uses `get_eval_transforms()` to make this clear.

### 5.3 Why These Augmentations?

Horizontal and vertical flips:

Paddy leaves can be photographed from different directions. Flipping should not
change disease identity.

RandomRotate90:

Phone orientation and leaf direction vary. Rotation helps the model learn
direction-invariant disease cues.

Brightness and contrast:

Farmers will take images under different lighting: sunlight, shadow, indoor
light, cloudy days. This augmentation helps the model avoid depending too much
on exact lighting.

Hue and saturation:

Phone cameras and lighting can shift colors. Paddy diseases often involve color
changes, so this must be moderate. Too much color shift could damage disease
signals.

Gaussian noise:

Camera sensors and compression can introduce noise. Small noise makes the model
more robust.

Coarse dropout:

Some leaf areas may be hidden, damaged, or out of frame. Dropping small patches
prevents the model from relying on only one tiny region.

### 5.4 Why Augmentation Helped

Experiment result:

```text
exp02 no augmentation: 0.9518 macro-F1
exp05 augmentation:    0.9573 macro-F1
Gain:                 +0.0055
```

This showed augmentation improved generalization. The gain was not huge, but it
was meaningful enough to keep.

---

## 6. Loss Function And Weighted Loss

### 6.1 CrossEntropyLoss

For multi-class classification, the standard loss is CrossEntropyLoss.

The model outputs logits. CrossEntropyLoss internally compares those logits
against the true class label. Lower loss means the model assigns higher score to
the correct class.

### 6.2 Why Weighted CrossEntropy?

Datasets are often imbalanced. Some diseases have more images than others.
Without weighting, a model can get good accuracy by doing well on common classes
while ignoring rare classes.

Weighted CrossEntropy gives more importance to classes with fewer samples.

Simple idea:

```text
Rare class mistake:    higher penalty
Common class mistake:  lower penalty
```

This aligns with macro-F1, where every class matters equally.

### 6.3 Result

```text
exp05 augmentation, no weighted loss: 0.9573 macro-F1
exp06 augmentation, weighted loss:    0.9670 macro-F1
Gain:                                +0.0097
```

Weighted loss gave a strong gain on top of augmentation, so it became part of
the preferred recipe.

---

## 7. Fine-Tuning Strategy

Fine-tuning means taking a pretrained model and adapting it to a new dataset.

Our strategy has two phases.

### 7.1 Phase A: Head Warmup

For the first 5 epochs:

```text
Freeze backbone
Train only the classification head
Learning rate: 3e-4
```

Why?

The new classification head starts with random weights. If we immediately train
the whole network with a random head, noisy gradients can disturb the pretrained
backbone. Head warmup lets the new final layer become reasonable first.

### 7.2 Phase B: Full Fine-Tuning

After warmup:

```text
Unfreeze all layers
Train the full model end-to-end
Use OneCycleLR scheduler
```

Why?

The backbone already knows general image features, but paddy disease patterns
are domain-specific. Full fine-tuning adapts lower and middle-level features to
leaf textures, lesions, spots, and color changes.

---

## 8. Learning Rate

### 8.1 What Is Learning Rate?

Learning rate controls how big each optimizer update is.

If learning rate is too high:

```text
Training can jump around and miss good solutions.
The loss may be unstable.
```

If learning rate is too low:

```text
Training is slow.
The model may stop before reaching a good solution.
```

Think of it as step size while walking downhill on a loss surface.

### 8.2 Initial Conservative Learning Rates

Early experiments used:

```text
head_lr = 3e-4
full_lr = 5e-5
```

This is conservative and safe for pretrained models. The head can learn faster
because it is new. The backbone should usually change more gently.

### 8.3 LR Sweep

Experiment exp07 tested different learning rate settings:

```text
head_lr=1e-3 / full_lr=1e-4
head_lr=3e-4 / full_lr=5e-5
head_lr=1e-4 / full_lr=1e-5
```

Best result:

```text
exp07: 0.9676 macro-F1
```

Compared with exp06:

```text
exp06: 0.9670 macro-F1
Gain: +0.0006
```

This gain was too small to trust as a meaningful improvement. We did not select
that LR sweep as the main method.

---

## 9. Scheduler

A scheduler changes the learning rate during training.

### 9.1 CosineAnnealingLR

CosineAnnealingLR gradually lowers the learning rate in a smooth curve. It is a
safe default, especially when fine-tuning pretrained models.

### 9.2 OneCycleLR

OneCycleLR increases the learning rate early, then decreases it strongly later.

In our setup:

```text
Warmup phase: train head only
Full fine-tune phase: OneCycleLR batch-level schedule
pct_start = 0.1
max_lr = tuned value
```

Why it can work:

The temporary high learning rate can help the model escape shallow local
solutions. The later low learning rate helps it settle into a sharper final
solution.

### 9.3 OneCycle Result

```text
exp06 conservative scheduler: 0.9670 macro-F1
exp08 OneCycle max_lr 1e-3:   0.9746 macro-F1
Gain:                        +0.0076
```

This was a meaningful improvement. OneCycleLR became the preferred scheduler.

### 9.4 OneCycle max_lr Sweep

Experiment exp09 tested max learning rate values:

| Variant | max_lr | Val macro-F1 | Decision |
| --- | ---: | ---: | --- |
| maxlr_3e4 | 3e-4 | 0.9686 | Too low |
| maxlr_5e4 | 5e-4 | 0.9683 | Too low |
| maxlr_1e3 | 1e-3 | 0.9749 | Good, close to exp08 |
| maxlr_15e4 | 1.5e-3 | 0.9779 | Best |
| maxlr_2e3 | 2e-3 | 0.9733 | Too high |

Decision:

```text
Use OneCycleLR max_lr = 1.5e-3
```

Why not 2e-3?

Because higher is not always better. Once the learning rate becomes too large,
training can overshoot useful minima and generalize worse.

---

## 10. Batch Size

### 10.1 What Is Batch Size?

Batch size is how many images the model processes before one optimizer update.

For this project:

```text
batch_size = 64
```

### 10.2 Why 64?

Batch size is a trade-off.

Larger batch:

```text
More stable gradient estimate
Better GPU utilization
More VRAM required
Sometimes weaker generalization if too large
```

Smaller batch:

```text
Less VRAM required
Noisier updates
Can generalize well, but training may be less stable
```

Batch size 64 is a practical middle point for MobileNetV3-Large on the available
GPU setup. It is large enough for stable training but small enough to fit memory
comfortably, even with augmentation and mixed precision.

### 10.3 Did We Select Different Batch Sizes?

For the completed MobileNet track, batch size was kept fixed at 64. This is
intentional.

When experimenting, changing too many things at once makes results hard to
interpret. Since the goal was to measure architecture, augmentation, weighted
loss, scheduler, learning rate, and resolution, batch size stayed constant.

If future work studies batch size, it should be a separate experiment.

---

## 11. Epochs And Best Epoch

Epoch means one full pass through the training dataset.

The selected config trains for:

```text
40 epochs
```

The best exp09 variant reached its best validation macro-F1 at:

```text
epoch 31
```

The script saves the best checkpoint whenever validation macro-F1 improves. So
even if later epochs slightly degrade, the final saved checkpoint is the best
validation checkpoint, not necessarily the last epoch.

Why not train forever?

Because after enough epochs, the model may overfit. Overfitting means it keeps
getting better at training images but stops improving, or worsens, on validation
images.

---

## 12. Seed And Reproducibility

### 12.1 What Is A Seed?

A seed controls random number generators.

Training uses randomness in many places:

```text
Weight initialization of the new head
Data shuffling
Random augmentation choices
GPU operation behavior
```

Setting a seed makes runs more reproducible.

### 12.2 Why Seed 42?

`42` is a common conventional seed. The exact number is not magic. What matters
is that we choose one seed and keep it fixed while comparing experiments.

Using the same seed helps answer:

```text
Did this experiment improve because of the changed variable,
or because random chance gave it an easier run?
```

### 12.3 Should We Try Multiple Seeds?

For final scientific confidence, yes. A strong final model can be rerun with
multiple seeds such as:

```text
42, 7, 123
```

Then report mean and standard deviation. But during early experimentation, one
fixed seed is a practical way to move quickly while keeping comparisons fair.

---

## 13. Input Resolution

### 13.1 Why Test Resolution?

Higher input resolution can preserve small visual details. For plant disease,
small spots or lesion texture might matter.

But higher resolution also increases compute.

Approximate compute grows with image area:

```text
224x224 = 50176 pixels
256x256 = 65536 pixels
288x288 = 82944 pixels
```

Going from 224 to 256 increases pixel area by about 31 percent. Going from 224
to 288 increases pixel area by about 65 percent.

### 13.2 Result

Experiment exp10 tested 256 after exp09.

```text
exp09 224: 0.9779 macro-F1, 15.02 ms CPU
exp10 256: 0.9718 macro-F1, 17.90 ms CPU
```

Decision:

```text
Reject higher resolution for this MobileNet track.
Keep 224x224.
```

Because 256 was both worse and slower, 288 was skipped.

Lesson:

Higher resolution is not automatically better. If the model and dataset already
capture the needed visual cues at 224, larger images can add noise, compute, and
overfitting risk without improving generalization.

---

## 14. Metrics

### 14.1 Accuracy

Accuracy is:

```text
correct predictions / total predictions
```

It is easy to understand, but it can be misleading when classes are imbalanced.
A model can score high accuracy by doing well on common classes and poorly on
rare classes.

### 14.2 Precision

Precision asks:

```text
When the model predicts this class, how often is it right?
```

High precision means few false positives.

### 14.3 Recall

Recall asks:

```text
Of all real examples of this class, how many did the model find?
```

High recall means few false negatives.

### 14.4 F1 Score

F1 balances precision and recall:

```text
F1 = harmonic mean of precision and recall
```

It punishes cases where precision is high but recall is low, or recall is high
but precision is low.

### 14.5 Macro-F1

Macro-F1 calculates F1 for each class, then averages all classes equally.

This is our primary metric because every disease class matters. A rare disease
should not be ignored just because it has fewer images.

### 14.6 Why We Do Not Select By Accuracy

The current best exp09 result:

```text
macro-F1: 0.9779
accuracy: 0.9763
```

Both are strong. But if they disagree in future, macro-F1 should drive model
selection because it is more sensitive to weak classes.

---

## 15. Experiment Decisions So Far

### Phase 1: Architecture Bake-Off

Goal: choose candidate backbones.

| ID | Question | Result | Decision |
| --- | --- | --- | --- |
| exp01 | Is MobileNetV3-Small enough? | 0.9251 F1, 8.28 ms | Very fast, but not accurate enough to be main track |
| exp02 | Does MobileNetV3-Large improve accuracy? | 0.9518 F1, 15.54 ms | Yes. Best Pareto MobileNet baseline |
| exp03 | Does EfficientNet-B0 improve accuracy? | 0.9609 F1, 25.79 ms | Yes. Keep as separate high-F1 track |
| exp04 | Is MobileViT-XXS promising? | 0.9227 F1, 21.57 ms | No. Stop this track |

Decision after Phase 1:

```text
Continue MobileNetV3-Large and EfficientNet-B0.
```

### Phase 2 MobileNet Track

| ID | Changed variable | Result | Decision |
| --- | --- | ---: | --- |
| exp05 | Add augmentation | 0.9573 | Keep augmentation |
| exp06 | Add weighted loss | 0.9670 | Keep weighted loss |
| exp07 | LR sweep | 0.9676 | Gain too small; do not replace recipe |
| exp08 | Use OneCycleLR | 0.9746 | Keep OneCycleLR |
| exp09 | Sweep OneCycle max_lr | 0.9779 | Select max_lr 1.5e-3 |
| exp10 | Increase input to 256 | 0.9718 | Reject higher resolution; keep 224 |

Final MobileNet decision:

```text
Use exp09 maxlr_15e4 as the preferred MobileNetV3-Large method.
```

---

## 16. Final Selected MobileNet Configuration

Use this config as the reference for future MobileNet fine-tuning:

```yaml
model:
  name: "mobilenet_v3_large"
  source: "torchvision.models"
  pretrained: true
  pretrained_weights: "IMAGENET1K_V1"

data:
  dataset: "kaggle_paddy_2022"
  split_csv: "data/splits/kaggle.csv"
  augmentation: true
  weighted_loss: true

training:
  epochs: 40
  batch_size: 64
  warmup_epochs: 5
  lr_head: 3.0e-4
  onecycle_max_lr: 1.5e-3
  onecycle_pct_start: 0.1
  onecycle_div_factor: 25.0
  onecycle_final_div_factor: 10000.0
  weight_decay: 1.0e-2
  seed: 42
```

The exact file is:

```text
experiments/exp09_mobilenetv3large_onecycle_sweep/configs/maxlr_15e4.yaml
```

---

## 17. How To Fine-Tune This Model

### 17.1 Start From The Preferred Branch

Use:

```powershell
git checkout exp09_mobilenetv3large_onecycle_sweep
```

If you create a new experiment based on this model, branch from exp09:

```powershell
git checkout -b expNN_mobilenetv3large_<new_variable>
```

Only change one variable at a time.

### 17.2 Run The Preferred Training Recipe

From repo root:

```powershell
conda activate krishidoc_ml
python experiments/exp09_mobilenetv3large_onecycle_sweep/train.py --config configs/maxlr_15e4.yaml
```

The script will:

1. Load config
2. Set seed
3. Build MobileNetV3-Large with ImageNet weights
4. Replace classifier head with 10-class head
5. Build train and validation datasets
6. Apply train augmentation only to train split
7. Apply deterministic eval preprocessing to validation split
8. Use weighted CrossEntropyLoss
9. Warm up the head for 5 epochs
10. Unfreeze the full model
11. Train with OneCycleLR
12. Save the best checkpoint by validation macro-F1
13. Save metrics, curves, and confusion matrix

### 17.3 What To Watch During Training

Watch these values:

```text
train_loss
val_macro_f1
val_accuracy
learning rate
best epoch
per-class F1
```

Healthy signs:

```text
train_loss decreases
val_macro_f1 improves, then stabilizes
weak classes do not collapse
best epoch is not always epoch 1 or 40
```

Warning signs:

```text
train_loss decreases but val_macro_f1 drops: overfitting
val_macro_f1 never improves: learning rate may be wrong or model underfits
one or two classes have very low F1: class imbalance or label confusion
high accuracy but weak macro-F1: majority-class bias
```

### 17.4 How To Fine-Tune On A New Dataset

If you later fine-tune on Dhan-Shomadhan or merged data:

1. Do not overwrite the original Kaggle split.
2. Create a new split CSV.
3. Confirm class names map to `src/classes.py`.
4. If adding new classes, append them only. Do not reorder old classes.
5. Start from the selected MobileNet recipe.
6. Keep test data locked.
7. First evaluate domain gap before training on the new data.

A careful sequence:

```text
Step 1: Evaluate exp09 checkpoint on Dhan-Shomadhan-style validation data.
Step 2: If performance drops, create a deliberate dataset adaptation experiment.
Step 3: Fine-tune with the same recipe on the new training split.
Step 4: Compare validation macro-F1 and per-class F1.
Step 5: Only after choosing final winner, evaluate test once.
```

### 17.5 What Not To Do

Do not:

```text
Run test evaluation after every experiment
Change class order casually
Change augmentation and scheduler in the same experiment
Commit checkpoints or ONNX files
Trust accuracy alone
Assume higher resolution is better
Assume a bigger model is always better
```

---

## 18. How To Read The Current Best Result

Best exp09 metrics:

```text
macro-F1:             0.9779
accuracy:             0.9763
top-2 accuracy:       0.9923
top-3 accuracy:       0.9968
macro AUC OVR:        0.9993
checkpoint size:      17.07 MB
CPU mean latency:     15.02 ms
CPU p95 latency:      17.55 ms
best epoch:           31
training time:        3119.2 seconds
peak VRAM:            1586.2 MB
```

Per-class F1:

| Class | F1 |
| --- | ---: |
| bacterial_leaf_blight | 0.9722 |
| bacterial_leaf_streak | 1.0000 |
| bacterial_panicle_blight | 0.9804 |
| blast | 0.9735 |
| brown_spot | 0.9754 |
| dead_heart | 0.9977 |
| downy_mildew | 0.9622 |
| hispa | 0.9617 |
| normal | 0.9681 |
| tungro | 0.9879 |

The weakest classes are still strong, but `hispa`, `downy_mildew`, and `normal`
are worth watching in future domain validation.

---

## 19. Why We Still Need EfficientNet-B0

EfficientNet-B0 had the highest Phase 1 validation macro-F1:

```text
EfficientNet-B0 baseline: 0.9609
MobileNetV3-Large baseline: 0.9518
```

But MobileNetV3-Large has now been heavily optimized to 0.9779.

We should not assume EfficientNet-B0 will or will not win. It needs the same
recipe:

```text
EfficientNet-B0 + augmentation
EfficientNet-B0 + augmentation + weighted loss
EfficientNet-B0 + best scheduler/tuning if needed
```

Then compare best MobileNet vs best EfficientNet.

---

## 20. Production Readiness Checklist

Before this model can become production:

```text
[ ] Compare against tuned EfficientNet-B0 track
[ ] Evaluate selected cross-track winner on locked test set once
[ ] Evaluate on Dhan-Shomadhan deployment-like field data
[ ] Export ONNX
[ ] Verify ONNX output shape is (1, 10)
[ ] Verify ONNX outputs raw logits
[ ] Verify ONNX Runtime CPU inference works
[ ] Measure ONNX CPU latency
[ ] Confirm ONNX file size below 50 MB
[ ] Write MODEL_CARD.md
[ ] Upload model.onnx and model card to Hugging Face Hub
[ ] Copy class mapping to product repo carefully
[ ] Ensure product preprocessing exactly matches training eval preprocessing
```

---

## 21. Mental Model For Future Experiments

Good ML experimentation is controlled scientific work.

Every experiment should answer one question:

```text
Does changing X improve validation macro-F1, weak-class behavior, or deployment cost?
```

Bad experiment:

```text
Change model, augmentation, learning rate, batch size, and input size together.
```

You might get a better score, but you will not know why.

Good experiment:

```text
Start from the best known config.
Change one variable.
Run training.
Compare macro-F1, weak classes, latency, and training curves.
Record the conclusion.
```

The goal is not just a high score. The goal is a defensible model decision.

---

## 22. Glossary

Backbone:
The feature extractor part of the model.

Classification head:
The final layer that maps extracted features to class logits.

Logits:
Raw model scores before softmax.

Softmax:
Function that converts logits into probabilities.

Epoch:
One full pass through the training dataset.

Batch:
A group of images processed before one optimizer update.

Learning rate:
Step size used by the optimizer.

Scheduler:
Rule that changes learning rate during training.

Augmentation:
Random label-preserving image transformations applied to training data.

Validation set:
Data used to choose models and hyperparameters.

Test set:
Locked data used once for final unbiased evaluation.

Macro-F1:
Average of per-class F1 scores, treating every class equally.

Weighted loss:
Loss that penalizes mistakes on rare classes more strongly.

Fine-tuning:
Adapting a pretrained model to a new task or dataset.

Seed:
Fixed value used to make randomness reproducible.

Overfitting:
When a model learns training data too specifically and generalizes poorly.

Domain shift:
When real-world data differs from training data.

---

## 23. Short Answer: What Should I Use Now?

Use this as the preferred MobileNet model method:

```text
Branch:       exp09_mobilenetv3large_onecycle_sweep
Config:       configs/maxlr_15e4.yaml
Input:        224x224
Augmentation: train only
Loss:         weighted CrossEntropyLoss
Scheduler:    OneCycleLR
max_lr:       1.5e-3
Batch size:   64
Seed:         42
```

Next best learning step:

```text
Run the EfficientNet-B0 track with the same discipline, then compare.
```