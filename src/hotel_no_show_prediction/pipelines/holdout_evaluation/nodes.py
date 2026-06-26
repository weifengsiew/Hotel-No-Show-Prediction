"""Holdout evaluation nodes."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_test_classes(calibrated_pipeline, X_test: pd.DataFrame):
    """Predict test-set classes."""
    return calibrated_pipeline.predict(X_test)


def predict_test_probabilities(calibrated_pipeline, X_test: pd.DataFrame):
    """Predict test-set positive-class probabilities."""
    return calibrated_pipeline.predict_proba(X_test)[:, 1]


def calculate_classification_metrics(
    y_test: pd.Series,
    predicted_class,
    predicted_probability,
) -> dict[str, float]:
    """Calculate test-set classifier metrics."""
    return {
        "test_roc_auc": roc_auc_score(y_test, predicted_probability),
        "test_average_precision": average_precision_score(y_test, predicted_probability),
        "test_brier_score": brier_score_loss(y_test, predicted_probability),
        "test_log_loss": log_loss(y_test, predicted_probability),
        "test_accuracy": accuracy_score(y_test, predicted_class),
        "test_precision": precision_score(y_test, predicted_class, zero_division=0),
        "test_recall": recall_score(y_test, predicted_class, zero_division=0),
        "test_f1": f1_score(y_test, predicted_class, zero_division=0),
    }
