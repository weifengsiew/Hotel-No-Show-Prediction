"""Train/test split pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.train_test_split.nodes import (
    create_train_test_split,
    split_features_and_target,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model data preparation pipeline."""
    return Pipeline(
        [
            node(
                func=split_features_and_target,
                inputs=["noshow_features", "params:no_show_experiment"],
                outputs=["features", "target"],
                name="split_features_and_target",
            ),
            node(
                func=create_train_test_split,
                inputs=["features", "target", "params:no_show_experiment"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="create_train_test_split",
            ),
        ]
    )
