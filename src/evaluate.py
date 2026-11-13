"""
Evaluation for a very imbalanced problem.

Accuracy is useless here (predicting "never fraud" is 99.8% accurate), so we use:
- PR-AUC (average precision): how well the model ranks frauds above normal payments
- precision and recall at the chosen threshold
- a business cost: a missed fraud costs its amount, a false alarm costs a fixed
  review cost (analyst time + friction for the customer).
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REVIEW_COST = 5.0  # EUR per false alarm (assumption, can be changed)


def confusion_at(y, p, threshold):
    flag = p >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    return tp, fp, fn


def business_cost(y, p, amount, threshold, review_cost=REVIEW_COST):
    flag = p >= threshold
    missed = np.sum(amount[(~flag) & (y == 1)])
    alarms = np.sum(flag & (y == 0)) * review_cost
    return float(missed + alarms)


def best_threshold(y, p, amount, review_cost=REVIEW_COST):
    """Threshold with the lowest business cost (chosen on the validation set, never on the test set)."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = np.array([business_cost(y, p, amount, t, review_cost) for t in grid])
    return float(grid[np.argmin(costs)]), grid, costs


def summary(y, p, amount, threshold, review_cost=REVIEW_COST):
    tp, fp, fn = confusion_at(y, p, threshold)
    caught = float(np.sum(amount[(p >= threshold) & (y == 1)]))
    total_fraud = float(np.sum(amount[y == 1]))
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "frauds caught": f"{tp}/{tp + fn}",
        "false alarms": fp,
        "fraud EUR caught": caught,
        "fraud EUR total": total_fraud,
        "cost EUR": business_cost(y, p, amount, threshold, review_cost),
    }
