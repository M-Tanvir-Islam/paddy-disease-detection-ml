# exp19 temporary log

## Setup

- Parent experiment: exp16 (seed 42)
- Changed variable: `training.seed` only; seeds 1 and 2, fixed in advance
- Initialization: fresh ImageNet weights for each run
- Deployable model unchanged: the seed-42 exp16 checkpoint
- Test evaluation: locked and not run

## Runs

### Completed runs

- Seed 1: validation macro-F1 0.9787, accuracy 0.9807, best epoch 37,
  54.5 minutes, CPU latency 15.59 ms
- Seed 2: validation macro-F1 0.9794, accuracy 0.9813, best epoch 29,
  54.4 minutes, CPU latency 14.94 ms
- Including exp16 (seed 42): macro-F1 0.9794 ± 0.0007, range 0.9787-0.9800
- `downy_mildew` weakest in every run: F1 0.9454 ± 0.0087
- Test set: not evaluated
- Decision: deployable model stays the seed-42 exp16 checkpoint
