# Mobile Price Classification

A multiclass machine-learning project for predicting mobile-phone **price ranges (0–3)** from hardware and connectivity features. The project compares Decision Trees, Random Forests, and Support Vector Machines (SVM), with exploratory data analysis, rule-based data cleaning, and hyperparameter tuning via `GridSearchCV`.

## Project overview

The original dataset contains **2,000 labeled training rows** and **20 predictive features** such as RAM, battery power, internal memory, CPU cores, camera specifications, screen dimensions, and connectivity indicators. A separate **1,000-row unlabeled test set** is used for final predictions.

The exploratory analysis found **RAM** to be the feature most strongly associated with `price_range`.

## Workflow

1. Exploratory data analysis and feature inspection
2. Rule-based plausibility checks and removal of logically invalid values
3. Train/test split
4. Baseline Decision Tree, Random Forest, and linear SVM models
5. Hyperparameter tuning with 5-fold cross-validation
6. Holdout evaluation and final test-set prediction

## Model selection results

| Model | Best 5-fold CV accuracy | Best configuration |
|---|---:|---|
| Decision Tree | 84.27% | entropy, `min_samples_leaf=2`, `min_samples_split=2` |
| Random Forest | 87.45% | 200 trees, `min_samples_split=5`, `max_features=sqrt` |
| SVM | **97.70%** | `C=10`, linear kernel, `gamma=scale` |

Saved holdout outputs in the notebook show:

- Tuned Random Forest: **90.00% accuracy**, **0.9004 macro-F1**
- Baseline linear SVM: **96.30% accuracy**
- GridSearch-selected SVM: **95.93% holdout accuracy** via the saved `best_svm.score(...)` output

> **Reproducibility note:** the original exploratory notebook had two evaluation cells that referenced a stale prediction variable. The public notebook in this repository corrects those references, and the affected saved outputs were cleared so they can be recomputed with the original CSV files.

## Repository structure

```text
mobile-price-classification/
├── assets/
├── data/
│   └── README.md
├── notebooks/
│   └── mobile_price_classification_analysis.ipynb
├── results/
│   ├── holdout_metrics.csv
│   └── model_selection_summary.csv
├── src/
│   └── train_models.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python src/train_models.py --train data/train.csv
```

The training script applies the same rule-based cleaning logic as the notebook, runs the three model grids, and writes recomputed metrics to `results/recomputed_metrics.json`.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter

## Key takeaway

Among the tested approaches, the SVM achieved the strongest cross-validation performance, while the Random Forest provided a strong nonlinear tree-based baseline. The project also highlights the value of feature-level exploratory analysis and careful model-specific evaluation in multiclass classification.
