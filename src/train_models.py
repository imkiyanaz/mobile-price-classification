"""
Train and compare Decision Tree, Random Forest, and SVM classifiers
for mobile price-range prediction.

Usage:
    python src/train_models.py --train data/train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


TARGET = "price_range"


def clean_logic_noise(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Apply the same rule-based plausibility checks used in the notebook."""
    conditions = {
        "battery_power": lambda x: (x <= 0) | (x > 10000),
        "ram": lambda x: (x <= 0) | (x < 64) | (x > 16384),
        "int_memory": lambda x: (x <= 0) | (x > 2048),
        "n_cores": lambda x: (x <= 0) | (x > 16),
        "clock_speed": lambda x: (x <= 0) | (x > 5),
        "px_height": lambda x: (x <= 0) | (x < 20) | (x > 5000),
        "px_width": lambda x: (x <= 0) | (x < 200) | (x > 5000),
        "sc_h": lambda x: (x <= 0) | (x > 20),
        "sc_w": lambda x: (x <= 0),
        "mobile_wt": lambda x: (x <= 0) | (x < 70) | (x > 350),
        "m_dep": lambda x: (x <= 0),
        "fc": lambda x: (x < 0) | (x > 100),
        "pc": lambda x: (x < 0) | (x > 100),
        "talk_time": lambda x: (x <= 0) | (x > 72),
        "blue": lambda x: ~x.isin([0, 1]),
        "dual_sim": lambda x: ~x.isin([0, 1]),
        "three_g": lambda x: ~x.isin([0, 1]),
        "four_g": lambda x: ~x.isin([0, 1]),
        "touch_screen": lambda x: ~x.isin([0, 1]),
        "wifi": lambda x: ~x.isin([0, 1]),
    }
    if include_target:
        conditions[TARGET] = lambda x: ~x.isin([0, 1, 2, 3])

    invalid = pd.Series(False, index=df.index)
    for col, condition in conditions.items():
        if col in df.columns:
            invalid |= condition(df[col])
    return df.loc[~invalid].reset_index(drop=True)


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to labeled training CSV")
    parser.add_argument("--output", default="results/recomputed_metrics.json")
    args = parser.parse_args()

    df = clean_logic_noise(pd.read_csv(args.train), include_target=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1
    )

    grids = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=1),
            {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=1),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
            },
        ),
        "SVM": (
            SVC(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
        ),
    }

    results = []
    for name, (estimator, param_grid) in grids.items():
        search = GridSearchCV(
            estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate(
            name, search.best_estimator_, X_train, X_test, y_train, y_test
        )
        metrics["best_params"] = search.best_params_
        metrics["best_cv_accuracy"] = float(search.best_score_)
        results.append(metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
