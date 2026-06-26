"""Holdout evaluation pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.holdout_evaluation.nodes import (
    calculate_classification_metrics,
    predict_test_classes,
    predict_test_probabilities,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model evaluation pipeline."""
    return Pipeline(
        [
            node(
                func=predict_test_classes,
                inputs=["calibrated_pipeline", "X_test"],
                outputs="predicted_test_classes",
                name="predict_test_classes",
            ),
            node(
                func=predict_test_probabilities,
                inputs=["calibrated_pipeline", "X_test"],
                outputs="predicted_test_probabilities",
                name="predict_test_probabilities",
            ),
            node(
                func=calculate_classification_metrics,
                inputs=["y_test", "predicted_test_classes", "predicted_test_probabilities"],
                outputs="test_set_metrics",
                name="calculate_classification_metrics",
            ),
        ]
    )
