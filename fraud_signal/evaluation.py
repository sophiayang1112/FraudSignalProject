import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix)


def group_partitions(applications, seed=27):
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev, test = next(first.split(applications, groups=applications.customer_id))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1)
    train_rel, valid_rel = next(second.split(applications.iloc[dev], groups=applications.iloc[dev].customer_id))
    return {"train": dev[train_rel], "validation": dev[valid_rel], "test": test}


def binary_metrics(y, scores, threshold=.5):
    labels = scores >= threshold
    return {"roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier_uncalibrated": float(brier_score_loss(y, scores)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "review_rate": float(np.mean(labels)),
        "confusion_matrix": confusion_matrix(y, labels, labels=[0, 1]).tolist()}


def review_threshold(scores, budget=.1):
    """Validation-only empirical threshold; ties may exceed the target budget."""
    return float(np.quantile(scores, 1-budget, method="higher"))
