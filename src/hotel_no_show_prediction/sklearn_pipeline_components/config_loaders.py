from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_CANDIDATE_PIPELINE_KEYS = ["name", "preprocessing", "model"]
REQUIRED_TOP_LEVEL_KEYS = [
    "experiment",
    "data",
    "target",
    "calibration",
    "mlflow",
    "search",
    "candidates",
]
REQUIRED_NESTED_KEYS = {
    "experiment": ["name", "random_state", "output_dir", "model_filename"],
    "data": ["path", "test_size"],
    "target": ["column"],
    "calibration": ["method", "cv_folds"],
    "mlflow": ["tracking_uri", "experiment_name"],
    "search": ["scoring", "cv_folds", "n_jobs"],
}


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file as a dictionary."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary-like YAML file: {config_path}")
    return config


def validate_candidate_pipelines(candidate_pipelines: list[dict[str, Any]]) -> None:
    """Check that each candidate pipeline has name, preprocessing, and model keys."""
    if not isinstance(candidate_pipelines, list) or not candidate_pipelines:
        raise ValueError("Config 'candidates' must be a non-empty list.")

    for candidate_pipeline in candidate_pipelines:
        missing_keys = [
            key
            for key in REQUIRED_CANDIDATE_PIPELINE_KEYS
            if key not in candidate_pipeline
        ]
        if missing_keys:
            raise ValueError(f"Missing keys in candidate pipeline: {missing_keys}")


def validate_required_config_keys(
    config: dict[str, Any],
    required_top_level_keys: list[str],
    required_nested_keys: dict[str, list[str]],
) -> None:
    """Check that a YAML config contains required top-level and nested keys."""
    missing_top_level_keys = [key for key in required_top_level_keys if key not in config]
    if missing_top_level_keys:
        raise ValueError(f"Missing top-level config keys: {missing_top_level_keys}")

    for top_level_key, nested_keys in required_nested_keys.items():
        missing_keys = [key for key in nested_keys if key not in config[top_level_key]]
        if missing_keys:
            raise ValueError(f"Missing keys in '{top_level_key}': {missing_keys}")


def load_experiment_config(
    config_path: str | Path,
    parameter_name: str = "no_show_experiment",
) -> dict[str, Any]:
    """Load and validate the experiment config."""
    parameters = load_yaml_config(Path(config_path))
    if parameter_name not in parameters:
        raise ValueError(f"Missing Kedro parameter group: {parameter_name}")

    config = parameters[parameter_name]
    validate_required_config_keys(config, REQUIRED_TOP_LEVEL_KEYS, REQUIRED_NESTED_KEYS)
    validate_candidate_pipelines(config["candidates"])
    return config
