"""Load ML experiment outputs for notebook visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import joblib
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
)


EXPERIMENT_CONFIG_PATH = Path("conf") / "base" / "parameters" / "ml_experiment.yml"
CV_RESULTS_ARTIFACT_PATH = "cv_results.json"
FINISHED_RUN_FILTER = "attributes.status = 'FINISHED'"
LATEST_RUN_ORDER_BY = ["start_time DESC"]
PLOT_COLOR = "#4C78A8"
GUIDE_LINE_COLOR = "#333333"
RECALL_GUIDE_VALUE = 0.60
MODEL_NAME_MARKERS = {
    "RandomForestClassifier": "RandomForest",
    "LGBMClassifier": "LightGBM",
    "XGBClassifier": "XGBoost",
}
PREPROCESSOR_NAME_MARKERS = {
    "ColumnTransformer": "preprocessor_v1",
}


@dataclass(frozen=True)
class ReportContext:
    """Notebook-ready ML experiment report data."""

    config: dict[str, Any]
    data: pd.DataFrame
    latest_run: pd.Series
    cv_results: pd.DataFrame
    y_test: pd.Series
    y_score: Any
    target_column: str


def add_project_source_path(project_root: Path) -> None:
    """Make the local Kedro package importable from a notebook kernel."""
    source_path = str(project_root / "src")
    if source_path not in sys.path:
        sys.path.append(source_path)


def split_train_test_data(dataset: pd.DataFrame, config: dict[str, Any]):
    """Create the same train/test split used by the Kedro pipeline."""
    from hotel_no_show_prediction.pipelines.train_test_split.nodes import (
        create_train_test_split,
        split_features_and_target,
    )

    features, target = split_features_and_target(dataset, config)
    return create_train_test_split(features, target, config)


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve a config path from the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_report_config(project_root: Path) -> dict[str, Any]:
    """Load the Kedro ML experiment config used by the report."""
    add_project_source_path(project_root)
    from hotel_no_show_prediction.sklearn_pipeline_components.config_loaders import (
        load_experiment_config,
    )

    return load_experiment_config(project_root / EXPERIMENT_CONFIG_PATH)


def load_feature_data(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load the feature-engineered dataset used by the report."""
    data_path = resolve_project_path(project_root, config["data"]["path"])
    return pd.read_parquet(data_path)


def load_latest_finished_run(project_root: Path, config: dict[str, Any]) -> pd.Series:
    """Load the latest finished MLflow run for the configured experiment."""
    mlflow.set_tracking_uri(
        str(resolve_project_path(project_root, config["mlflow"]["tracking_uri"]))
    )
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string=FINISHED_RUN_FILTER,
        order_by=LATEST_RUN_ORDER_BY,
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No finished MLflow run found. Run ./run.sh first.")

    return runs.iloc[0]


def load_cv_results(run_id: str) -> pd.DataFrame:
    """Load GridSearchCV results from the latest MLflow run artifact."""
    cv_results_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=CV_RESULTS_ARTIFACT_PATH,
    )
    return pd.read_json(cv_results_path, orient="split")


def predict_no_show_probabilities(
    project_root: Path,
    config: dict[str, Any],
    X_test: pd.DataFrame,
):
    """Predict no-show probabilities for the held-out test rows."""
    model_path = (
        resolve_project_path(project_root, config["experiment"]["output_dir"])
        / config["experiment"]["model_filename"]
    )
    calibrated_pipeline = joblib.load(model_path)
    return calibrated_pipeline.predict_proba(X_test)[:, 1]


def load_report_context(project_root: Path) -> ReportContext:
    """Load and unpack report data into notebook-friendly attributes."""
    config = load_report_config(project_root)
    data = load_feature_data(project_root, config)
    latest_run = load_latest_finished_run(project_root, config)
    cv_results = load_cv_results(latest_run["run_id"])

    _, X_test, _, y_test = split_train_test_data(data, config)
    y_score = predict_no_show_probabilities(project_root, config, X_test)

    return ReportContext(
        config=config,
        data=data,
        latest_run=latest_run,
        cv_results=cv_results,
        y_test=y_test,
        y_score=y_score,
        target_column=config["target"]["column"],
    )


def set_report_theme() -> None:
    """Apply the notebook visual style."""
    sns.set_theme(style="whitegrid")


def summarize_dataset(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Summarize the feature-engineered dataset."""
    return pd.DataFrame(
        {
            "item": ["rows", "columns", "target column"],
            "value": [len(data), data.shape[1], target_column],
        }
    )


def summarize_prediction_features(project_root: Path, data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the feature groups used by the scikit-learn prediction pipeline."""
    add_project_source_path(project_root)
    from hotel_no_show_prediction.sklearn_pipeline_components.preprocessor_registry import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
    )

    feature_groups = [
        ("numeric type", NUMERIC_FEATURES),
        ("string type", CATEGORICAL_FEATURES),
    ]
    rows = []
    for feature_group, features in feature_groups:
        for feature_name in features:
            rows.append(
                {
                    "feature_group": feature_group,
                    "feature": feature_name,
                    "dtype": str(data[feature_name].dtype),
                }
            )

    return pd.DataFrame(rows)


def get_candidate_pipeline_label(cv_result: pd.Series) -> str:
    """Return a readable candidate-pipeline label from one CV result row."""
    preprocessor_text = str(cv_result["param_preprocess"])
    model_text = str(cv_result["param_model"])
    preprocessor_name = next(
        (
            name
            for marker, name in PREPROCESSOR_NAME_MARKERS.items()
            if marker in preprocessor_text
        ),
        preprocessor_text,
    )
    model_name = next(
        (name for marker, name in MODEL_NAME_MARKERS.items() if marker in model_text),
        model_text,
    )
    hyperparameters = []
    for column_name, value in cv_result.items():
        if not column_name.startswith("param_model__"):
            continue

        is_missing_value = pd.isna(value) or str(value).lower() == "nan"
        if is_missing_value:
            continue

        hyperparameter_name = column_name.removeprefix("param_model__")
        hyperparameters.append(f"{hyperparameter_name}={value}")

    if not hyperparameters:
        return f"{preprocessor_name} + {model_name}"

    return f"{preprocessor_name} + {model_name}: {', '.join(hyperparameters)}"


def summarise_candidate_pipelines(cv_results: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Summarise ranked candidate pipelines for notebook display."""
    candidate_pipelines = cv_results.copy()
    candidate_pipelines["candidate_pipeline"] = candidate_pipelines.apply(
        get_candidate_pipeline_label,
        axis=1,
    )
    candidate_pipelines = candidate_pipelines[
        ["rank_test_score", "candidate_pipeline"]
    ].sort_values("rank_test_score")

    return (
        candidate_pipelines
        .style.set_properties(
            subset=["candidate_pipeline"],
            **{
                "text-align": "left",
                "white-space": "normal",
                "min-width": "620px",
                "max-width": "620px",
            },
        )
        .set_properties(
            subset=["rank_test_score"],
            **{"text-align": "center", "width": "110px"},
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("vertical-align", "top")]},
            ]
        )
        .hide(axis="index")
    )


def summarize_selected_pipeline(latest_run: pd.Series) -> pd.DataFrame:
    """Summarize the selected pipeline from the latest MLflow run."""
    rows = [
        {
            "component": "preprocessor",
            "detail": "selected preprocessor",
            "value": latest_run.get("params.best__preprocess"),
        },
        {
            "component": "model",
            "detail": "selected model",
            "value": latest_run.get("params.best__model"),
        },
    ]

    hyperparameter_keys = sorted(
        key for key in latest_run.index if key.startswith("params.best__model__")
    )
    rows.extend(
        {
            "component": "hyperparameter",
            "detail": key.removeprefix("params.best__model__"),
            "value": latest_run.get(key),
        }
        for key in hyperparameter_keys
    )

    return pd.DataFrame(rows)



def plot_target_distribution(data: pd.DataFrame, target_column: str, ax) -> None:
    """Plot the no-show target distribution."""
    target_labels = data[target_column].map({0: "Show", 1: "No-show"})
    target_counts = (
        target_labels.value_counts()
        .reindex(["Show", "No-show"], fill_value=0)
        .rename_axis("booking_outcome")
        .reset_index(name="bookings")
    )
    sns.barplot(
        data=target_counts,
        x="booking_outcome",
        y="bookings",
        color=PLOT_COLOR,
        ax=ax,
    )
    ax.bar_label(ax.containers[0], labels=target_counts["bookings"].map("{:,}".format))
    ax.set_xlabel("Booking outcome")
    ax.set_ylabel("Number of bookings")


def plot_cv_results(cv_results: pd.DataFrame, ax) -> None:
    """Plot mean cross-validation AUROC by candidate ranking."""
    plot_data = cv_results.sort_values("rank_test_score").copy()
    plot_data["rank_label"] = plot_data["rank_test_score"].map(
        lambda rank: f"Rank {int(rank)}"
    )

    sns.barplot(
        data=plot_data,
        x="rank_label",
        y="mean_test_score",
        color=PLOT_COLOR,
        ax=ax,
    )
    ax.bar_label(
        ax.containers[0],
        labels=plot_data["mean_test_score"].map("{:.3f}".format),
        padding=3,
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_title("GridSearchCV Mean Cross-Validation AUROC")
    ax.set_xlabel("Candidate ranking")
    ax.set_ylabel("Mean CV AUROC")


def plot_roc_curve(y_test: pd.Series, y_score, ax) -> None:
    """Plot the test-set ROC curve."""
    display = RocCurveDisplay.from_predictions(y_test, y_score, name="No-show", ax=ax)
    point_index = min(
        range(len(display.tpr)),
        key=lambda index: abs(display.tpr[index] - RECALL_GUIDE_VALUE),
    )
    guide_fpr = display.fpr[point_index]
    guide_recall = display.tpr[point_index]
    ax.hlines(
        guide_recall,
        xmin=0,
        xmax=guide_fpr,
        colors=GUIDE_LINE_COLOR,
        linestyles="--",
        linewidth=1,
    )
    ax.vlines(
        guide_fpr,
        ymin=0,
        ymax=guide_recall,
        colors=GUIDE_LINE_COLOR,
        linestyles="--",
        linewidth=1,
    )
    ax.annotate(
        f"Recall / TPR = {guide_recall:.2f}\nFPR = {guide_fpr:.2f}",
        xy=(guide_fpr, guide_recall),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title(f"ROC Curve (AUROC = {display.roc_auc:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall / TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_precision_recall_curve(y_test: pd.Series, y_score, ax) -> None:
    """Plot the test-set precision-recall curve."""
    auprc = average_precision_score(y_test, y_score)
    display = PrecisionRecallDisplay.from_predictions(
        y_test,
        y_score,
        name="No-show",
        ax=ax,
    )
    point_index = min(
        range(len(display.recall)),
        key=lambda index: abs(display.recall[index] - RECALL_GUIDE_VALUE),
    )
    guide_recall = display.recall[point_index]
    guide_precision = display.precision[point_index]
    ax.hlines(
        guide_precision,
        xmin=0,
        xmax=guide_recall,
        colors=GUIDE_LINE_COLOR,
        linestyles="--",
        linewidth=1,
    )
    ax.vlines(
        guide_recall,
        ymin=0,
        ymax=guide_precision,
        colors=GUIDE_LINE_COLOR,
        linestyles="--",
        linewidth=1,
    )
    ax.annotate(
        f"Recall / TPR = {guide_recall:.2f}\nPrecision = {guide_precision:.2f}",
        xy=(guide_recall, guide_precision),
        xytext=(-102, -30),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title(f"Precision-Recall Curve (AUPRC = {auprc:.3f})")
    ax.set_xlabel("Recall / TPR")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_calibration_curve(y_test: pd.Series, y_score, ax) -> None:
    """Plot the test-set calibration curve."""
    brier_score = brier_score_loss(y_test, y_score)
    CalibrationDisplay.from_predictions(
        y_test,
        y_score,
        n_bins=25,
        strategy="quantile",
        name="No-show",
        ax=ax,
    )
    ax.set_title(f"Calibration Curve (Brier score = {brier_score:.3f})")
    ax.set_xlabel("Predicted No-Show Probability")
    ax.set_ylabel("Observed No-Show Rate")
