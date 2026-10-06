# Data

The original notebook expects two CSV files in the working directory:

- `train.csv` — 2,000 labeled rows with 20 predictors and the `price_range` target.
- `test.csv` — 1,000 unlabeled rows with an `id` column and the same 20 predictors.

The raw CSV files are not included in this public repository. Place them in this folder and update the notebook paths to `data/train.csv` and `data/test.csv`, or run the training script with explicit paths.

Expected target classes: `0`, `1`, `2`, `3`.
