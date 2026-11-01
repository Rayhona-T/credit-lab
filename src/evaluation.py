"""Compute evaluation metrics and confusion matrices from real predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import POSITIVE_CLASS


def _positive_proba(estimator, x) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if POSITIVE_CLASS not in classes:
        # Pipelines expose classes_ on the final step.
        final = estimator
        if hasattr(estimator, "named_steps"):
            final = estimator.named_steps[list(estimator.named_steps)[-1]]
        classes = list(getattr(final, "classes_", [0, 1]))
    try:
        idx = classes.index(POSITIVE_CLASS)
    except ValueError:
        idx = 1 if proba.shape[1] > 1 else 0
    return proba[:, idx]


def score_predictions(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": None,
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def confusion_counts(y_true, y_pred) -> dict[str, Any]:
    """Return a 2x2 matrix with class order [0, 1] (non-default, default)."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "labels": [0, 1],
        "matrix": matrix.astype(int).tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_estimator(estimator, x_test, y_test) -> dict[str, Any]:
    y_pred = estimator.predict(x_test)
    y_score = _positive_proba(estimator, x_test)
    return {
        "metrics": score_predictions(y_test, y_pred, y_score),
        "confusion": confusion_counts(y_test, y_pred),
        "supports_predict_proba": y_score is not None,
        "n_test": int(len(y_test)),
    }


def select_best_model(results: dict[str, dict], metric: str) -> tuple[str, float]:
    ranked = []
    for key, payload in results.items():
        value = payload["metrics"].get(metric)
        if value is not None:
            ranked.append((key, float(value)))
    if not ranked:
        raise ValueError(f"No model reported metric {metric!r}.")
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0]
