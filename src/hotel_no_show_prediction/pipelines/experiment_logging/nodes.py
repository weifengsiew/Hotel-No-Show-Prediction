"""Experiment logging nodes."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd

CV_RESULTS_ARTIFACT = "cv_results.json"
MODEL_ARTIFACT_NAME = "model"


def prepare_cv_results_table(grid_search) -> pd.DataFrame:
    """Prepare GridSearchCV results for MLflow logging."""
    cv_results = pd.DataFrame(grid_search.cv_results_)
    for column in cv_results.columns:
        if column.startswith("param_"):
            cv_results[column] = cv_results[column].map(str)
    return cv_results


def format_best_parameters(best_parameters: dict) -> dict[str, str]:
    """Format selected pipeline details for MLflow logging."""
    formatted_parameters = {}
    for key, value in best_parameters.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            formatted_parameters[f"best__{key}"] = str(value)
        else:
            formatted_parameters[f"best__{key}"] = value.__class__.__name__

    return formatted_parameters


def log_mlflow_run(
    config: dict,
    cv_results: pd.DataFrame,
    best_cv_score: float,
    formatted_best_parameters: dict[str, str],
    calibrated_pipeline,
    test_set_metrics: dict[str, float],
) -> str:
    """Log MLflow run."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["experiment"]["name"]) as run:
        mlflow.log_table(cv_results, CV_RESULTS_ARTIFACT)
        mlflow.log_metric("best_cv_score", best_cv_score)
        mlflow.log_params(formatted_best_parameters)
        mlflow.sklearn.log_model(calibrated_pipeline,name=MODEL_ARTIFACT_NAME,)
        mlflow.log_metrics(test_set_metrics)

        return run.info.run_id
