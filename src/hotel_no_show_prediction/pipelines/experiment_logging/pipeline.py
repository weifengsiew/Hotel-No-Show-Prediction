"""Experiment logging pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.experiment_logging.nodes import (
    format_best_parameters,
    log_mlflow_run,
    prepare_cv_results_table,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the MLflow tracking pipeline."""
    return Pipeline(
        [
            node(
                func=prepare_cv_results_table,
                inputs="grid_search",
                outputs="cv_results_table",
                name="prepare_cv_results_table",
            ),
            node(
                func=format_best_parameters,
                inputs="best_parameters",
                outputs="formatted_best_parameters",
                name="format_best_parameters",
            ),
            node(
                func=log_mlflow_run,
                inputs=[
                    "params:no_show_experiment",
                    "cv_results_table",
                    "best_cv_score",
                    "formatted_best_parameters",
                    "calibrated_pipeline",
                    "test_set_metrics",
                ],
                outputs="mlflow_run_id",
                name="log_mlflow_run",
            ),
        ]
    )
