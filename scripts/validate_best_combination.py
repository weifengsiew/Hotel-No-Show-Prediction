"""Evaluate the selected best combination on the untouched test set."""

from pathlib import Path
import sys

import click
import mlflow
import mlflow.models
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import read_yaml, resolve_project_path
from src.model_pipeline import (
    create_model,
    create_pipeline,
    create_preprocessor,
    split_features_and_target,
)


DEFAULT_BEST_COMBINATION_PATH = (
    PROJECT_ROOT / "reports" / "model" / "best_combination_config.yaml"
)
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "noshow_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "noshow_test.csv"


# ---------------------------------------------------------------------------
# Data and model preparation
# ---------------------------------------------------------------------------


def load_selected_config(config_path: str | Path) -> dict:
    """Load the selected best-combination configuration file."""
    config_path = resolve_project_path(config_path)
    config = read_yaml(config_path)

    return config


def load_train_and_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved train and test datasets created by the split step."""
    train_df = pd.read_csv(resolve_project_path(DEFAULT_TRAIN_PATH))
    test_df = pd.read_csv(resolve_project_path(DEFAULT_TEST_PATH))

    return train_df, test_df


def create_selected_pipeline(
    config: dict,
) -> Pipeline:
    """Build the selected preprocessing-plus-model pipeline."""
    # Create preprocessing, then combine it with the selected model.
    processing_config = config["processing"]
    preprocessor = create_preprocessor(processing_config)
    model = create_model(config["model"])
    pipeline = create_pipeline(preprocessor, model)

    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def create_evaluation_data(
    config: dict,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the test dataset format expected by MLflow evaluation."""
    # Keep raw feature columns plus the configured target column.
    data_config = config["data"]
    columns_to_drop = data_config["drop_columns"]
    evaluation_data = test_df.drop(columns=columns_to_drop)

    return evaluation_data


def fit_selected_pipeline(
    config: dict,
    train_df: pd.DataFrame,
) -> Pipeline:
    """Build and fit the selected pipeline on the training data."""
    # Build the selected pipeline and separate train features from target.
    pipeline = create_selected_pipeline(config)
    train_features, train_target = split_features_and_target(
        train_df,
        config["data"],
    )

    # Fit preprocessing and model together.
    pipeline.fit(train_features, train_target)

    return pipeline


# ---------------------------------------------------------------------------
# Output and tracking
# ---------------------------------------------------------------------------


def print_validation_summary(
    config: dict,
    metrics: dict,
    run_id: str,
    model_uri: str,
) -> None:
    """Print the final validation summary."""
    # Extract the most important metrics for command-line feedback.
    run_name = config["tracking"]["run_name"]
    roc_auc = metrics.get("roc_auc", float("nan"))
    f1 = metrics.get("f1_score", metrics.get("f1", float("nan")))
    print(
        f"Validated {run_name}: "
        f"test_roc_auc={roc_auc:.4f}, "
        f"test_f1={f1:.4f}, "
        f"run_id={run_id}, "
        f"model_uri={model_uri}"
    )


@click.command()
@click.option(
    "--config",
    default=str(DEFAULT_BEST_COMBINATION_PATH),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the selected best-combination YAML config.",
)
def main(config: Path) -> None:
    """Run final model validation and save the selected model artifacts."""
    # Load inputs.
    config = load_selected_config(config)
    train_df, test_df = load_train_and_test_data()

    # Track fitting and evaluation with MLflow sklearn autologging.
    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=config["tracking"]["run_name"]) as run:
        # Build and fit the selected pipeline.
        fit_selected_pipeline(config, train_df)
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Evaluate on the untouched test set with MLflow's built-in evaluator.
        evaluation_data = create_evaluation_data(config, test_df)
        evaluation_result = mlflow.models.evaluate(
            model=model_uri,
            data=evaluation_data,
            targets=config["data"]["target_column"],
            model_type="classifier",
            evaluators="default",
        )
        metrics = evaluation_result.metrics

    # Print a concise run summary.
    print_validation_summary(config, metrics, run_id, model_uri)


if __name__ == "__main__":
    main()
