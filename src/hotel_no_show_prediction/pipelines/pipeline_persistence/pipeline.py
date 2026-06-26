"""Pipeline persistence pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.pipeline_persistence.nodes import (
    persist_calibrated_pipeline,
    summarise_persisted_pipeline,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the pipeline persistence pipeline."""
    return Pipeline(
        [
            node(
                func=persist_calibrated_pipeline,
                inputs=["calibrated_pipeline", "params:no_show_experiment"],
                outputs="model_path",
                name="persist_calibrated_pipeline",
            ),
            node(
                func=summarise_persisted_pipeline,
                inputs=[
                    "params:no_show_experiment",
                    "best_cv_score",
                    "best_parameters",
                    "mlflow_run_id",
                    "model_path",
                    "test_set_metrics",
                ],
                outputs="persisted_pipeline_summary",
                name="summarise_persisted_pipeline",
            ),
        ]
    )
