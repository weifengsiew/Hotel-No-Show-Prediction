"""Pipeline persistence nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def persist_calibrated_pipeline(calibrated_pipeline, config: dict[str, Any]) -> str:
    """Persist the calibrated pipeline and return the artifact path."""
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / config["experiment"]["model_filename"]
    joblib.dump(calibrated_pipeline, model_path)

    return str(model_path)


def stringify_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    """Convert selected model details into JSON-safe strings."""
    return {key: str(value) for key, value in mapping.items()}


def summarise_persisted_pipeline(
    config: dict[str, Any],
    best_cv_score: float,
    best_parameters: dict[str, Any],
    mlflow_run_id: str,
    model_path: str,
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    """Create a JSON-safe summary of the persisted pipeline and metrics."""
    return {
        "experiment": config["experiment"]["name"],
        "best_cv_score": float(best_cv_score),
        "best_params": stringify_mapping(best_parameters),
        "mlflow_run_id": mlflow_run_id,
        "model_path": model_path,
        "test_metrics": test_metrics,
    }
