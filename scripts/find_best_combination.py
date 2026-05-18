"""Find the best feature-model-hyperparameter combination with GridSearchCV."""

from itertools import product
from pathlib import Path
import sys
from typing import Any

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config, read_yaml, resolve_project_path, write_yaml
from src.model_pipeline import (
    create_model,
    create_pipeline,
    create_preprocessor,
    split_features_and_target,
)


DEFAULT_MATRIX_PATH = PROJECT_ROOT / "configs" / "experiment_matrix.yaml"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "noshow_train.csv"


# ---------------------------------------------------------------------------
# Config and data loading
# ---------------------------------------------------------------------------


def load_experiment_configs(matrix_path: str | Path) -> tuple[dict, dict]:
    """Load the experiment matrix config and its shared settings config."""
    matrix_path = resolve_project_path(matrix_path)
    matrix_config = read_yaml(matrix_path)
    shared_config = load_config(matrix_config["shared_config"])

    return matrix_config, shared_config


def load_training_data(data_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training split and separate model features from target."""
    train_path = resolve_project_path(DEFAULT_TRAIN_PATH)
    train_df = pd.read_csv(train_path)
    train_features, train_target = split_features_and_target(train_df, data_config)

    return train_features, train_target


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------


def create_candidate_grid(
    matrix_config: dict,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    """Create GridSearchCV candidates for all feature and model combinations."""
    param_grid = []
    candidate_lookup = {}
    feature_configs = {
        feature_name: load_config(feature_settings["config"])["processing"]
        for feature_name, feature_settings in matrix_config["features"].items()
    }
    model_configs = {
        model_name: load_config(model_settings["config"])["model"]
        for model_name, model_settings in matrix_config["models"].items()
    }

    for (feature_name, processing_config), (model_name, model_config) in product(
        feature_configs.items(),
        model_configs.items(),
    ):
        preprocessor = create_preprocessor(processing_config)
        model = create_model(model_config)
        grid_entry = {
            "preprocessor": [preprocessor],
            "model": [model],
            **{
                f"model__{parameter_name}": parameter_values
                for parameter_name, parameter_values in model_config[
                    "tuning_grid"
                ].items()
            },
        }
        candidate_lookup[(id(preprocessor), id(model))] = {
            "feature_name": feature_name,
            "model_name": model_name,
            "processing": processing_config,
            "model": model_config,
        }
        param_grid.append(grid_entry)

    return param_grid, candidate_lookup


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def create_grid_search(
    param_grid: list[dict[str, Any]],
    training_config: dict,
) -> GridSearchCV:
    """Build a GridSearchCV object that searches feature, model, and hyperparameter choices."""
    pipeline = create_pipeline("passthrough", RandomForestClassifier())
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=training_config["scoring"],
        cv=training_config["cv"],
        n_jobs=training_config["grid_search_n_jobs"],
        verbose=1,
    )

    return grid_search


# ---------------------------------------------------------------------------
# Selection config
# ---------------------------------------------------------------------------


def get_best_model_parameters(best_params: dict, model_config: dict) -> dict:
    """Combine the model defaults with the best GridSearchCV hyperparameters."""
    tuned_parameters = {
        parameter_name.removeprefix("model__"): parameter_value
        for parameter_name, parameter_value in best_params.items()
        if parameter_name.startswith("model__")
    }
    model_parameters = {**model_config["parameters"], **tuned_parameters}

    return model_parameters


def create_selected_config(
    grid_search: GridSearchCV,
    shared_config: dict,
    matrix_config: dict,
    candidate_lookup: dict[tuple[int, int], dict[str, Any]],
) -> tuple[dict, dict]:
    """Create the final config and summary for the best CV combination."""
    best_params = grid_search.best_params_
    candidate_key = (id(best_params["preprocessor"]), id(best_params["model"]))
    candidate = candidate_lookup[candidate_key]
    experiment_name = f"{candidate['model_name']}_{candidate['feature_name']}"
    selected_model_parameters = get_best_model_parameters(
        best_params,
        candidate["model"],
    )
    selected_config = {
        "data": shared_config["data"],
        "processing": candidate["processing"],
        "model": {
            "name": candidate["model"]["name"],
            "parameters": selected_model_parameters,
        },
        "tracking": {
            "experiment_name": (
                f"{matrix_config['tracking']['experiment_name_prefix']}_"
                f"{experiment_name}"
            ),
            "run_name": experiment_name,
            "tracking_uri": matrix_config["tracking"]["tracking_uri"],
        },
    }
    selected_summary = {
        "experiment_name": experiment_name,
        "best_cv_score": grid_search.best_score_,
    }

    return selected_config, selected_summary


def save_selection_outputs(
    matrix_config: dict,
    selected_config: dict,
) -> Path:
    """Save the selected best-combination config needed for validation."""
    reports_dir = resolve_project_path(matrix_config["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    selected_config_path = reports_dir / "best_combination_config.yaml"

    write_yaml(selected_config, selected_config_path)

    return selected_config_path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config",
    default=str(DEFAULT_MATRIX_PATH),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the YAML experiment matrix.",
)
def main(config: Path) -> None:
    """Run the full feature-model-hyperparameter selection workflow."""
    matrix_config, shared_config = load_experiment_configs(config)
    train_features, train_target = load_training_data(shared_config["data"])
    param_grid, candidate_lookup = create_candidate_grid(matrix_config)
    grid_search = create_grid_search(
        param_grid,
        shared_config["training"],
    )
    grid_search.fit(train_features, train_target)
    selected_config, selected_summary = create_selected_config(
        grid_search,
        shared_config,
        matrix_config,
        candidate_lookup,
    )
    save_selection_outputs(
        matrix_config,
        selected_config,
    )

    print(
        f"Selected {selected_summary['experiment_name']}: "
        f"best_cv_score={selected_summary['best_cv_score']:.4f}"
    )


if __name__ == "__main__":
    main()
