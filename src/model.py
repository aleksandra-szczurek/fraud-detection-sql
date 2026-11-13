"""
Features, time-based split and the two models.

- Logistic regression: simple, interpretable baseline.
- Gradient boosting (HistGradientBoostingClassifier): non-linear model.
Both use class weights, because frauds are only ~0.17% of transactions.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PCA_COLS = [f"V{i}" for i in range(1, 29)]


def make_features(df):
    X = df[PCA_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    return X, df["Class"].to_numpy(), df["Amount"].to_numpy()


def time_split(n, train=0.6, valid=0.2):
    """Chronological split: first 60% train, next 20% validation (to choose the threshold), last 20% test."""
    a, b = int(n * train), int(n * (train + valid))
    return slice(0, a), slice(a, b), slice(b, n)


def build_models(seed=42):
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            class_weight="balanced", random_state=seed,
        ),
    }
