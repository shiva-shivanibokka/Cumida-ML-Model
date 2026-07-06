"""Evaluation metrics and plots, shared by the notebooks and the training CLI.

Keeping this logic in one place means Notebook 3, Notebook 4, and ``train.py``
all compute metrics identically, so their numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def evaluate_binary(y_true_binary, y_pred_binary, y_prob_pos) -> dict:
    """Compute the standard metric bundle for the positive (HCC) class.

    All inputs are in binary 1/0 space (1 = HCC). Returns a JSON-serializable dict.
    """
    y_true_binary = np.asarray(y_true_binary)
    y_pred_binary = np.asarray(y_pred_binary)

    # labels=[1, 0] => rows/cols are [HCC, normal]; ravel is [TP, FN, FP, TN].
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
    tp, fn, fp, tn = (int(v) for v in cm.ravel())

    return {
        "f1": float(f1_score(y_true_binary, y_pred_binary)),
        "roc_auc": float(roc_auc_score(y_true_binary, y_prob_pos)),
        "precision": float(precision_score(y_true_binary, y_pred_binary)),
        "recall": float(recall_score(y_true_binary, y_pred_binary)),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def predict_binary_and_proba(fitted_estimator, X):
    """Return (binary predictions, P(HCC)) for a pipeline trained on 1/0 labels."""
    y_pred = fitted_estimator.predict(X)
    # classes_ is [0, 1]; column 1 is P(HCC).
    pos_idx = list(fitted_estimator.classes_).index(1)
    y_prob = fitted_estimator.predict_proba(X)[:, pos_idx]
    return y_pred, y_prob


def roc_points(y_true_binary, y_prob_pos):
    """FPR/TPR arrays for plotting an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_pos, pos_label=1)
    return fpr, tpr


def comparison_frame(lr_metrics: dict, gb_metrics: dict) -> pd.DataFrame:
    """Side-by-side metrics table for the two models."""
    rows = {
        "Logistic Regression": lr_metrics,
        "Gradient Boosting": gb_metrics,
    }
    return pd.DataFrame(
        {
            name: {
                "F1 Score": round(m["f1"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
            }
            for name, m in rows.items()
        }
    ).T


def winner_by_f1(lr_metrics: dict, gb_metrics: dict) -> str:
    return (
        "Gradient Boosting"
        if gb_metrics["f1"] >= lr_metrics["f1"]
        else "Logistic Regression"
    )
