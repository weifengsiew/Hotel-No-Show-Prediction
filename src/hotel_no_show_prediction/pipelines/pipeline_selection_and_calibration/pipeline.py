"""Pipeline selection and calibration pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.pipeline_selection_and_calibration.nodes import (
    calibrate_pipeline,
    select_best_pipeline,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the pipeline selection and calibration pipeline."""
    return Pipeline(
        [
            node(
                func=select_best_pipeline,
                inputs="grid_search",
                outputs=["selected_pipeline", "best_cv_score", "best_parameters"],
                name="select_best_pipeline",
            ),
            node(
                func=calibrate_pipeline,
                inputs=[
                    "selected_pipeline",
                    "X_train",
                    "y_train",
                    "params:no_show_experiment",
                ],
                outputs="calibrated_pipeline",
                name="calibrate_pipeline",
            ),
        ]
    )
