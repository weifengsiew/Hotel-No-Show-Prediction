"""ML experiment nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from hotel_no_show_prediction.sklearn_pipeline_components.model_registry import (
    build_model,
    model_requires_dense_input,
)
from hotel_no_show_prediction.sklearn_pipeline_components.preprocessor_registry import (
    build_preprocessor,
)

PREPROCESS_STEP_NAME = "preprocess"
MODEL_STEP_NAME = "model"
REQUIRED_MODEL_HYPERPARAM_KEYS = ["estimator", "search_grid"]


def build_placeholder_pipeline() -> Pipeline:
    """Build placeholder pipeline that GridSearchCV will replace."""
    return Pipeline(
        steps=[
            (PREPROCESS_STEP_NAME, "passthrough"),
            (MODEL_STEP_NAME, "passthrough"),
        ]
    )


def validate_model_hyperparams(
    model_name: str,
    model_hyperparams: dict[str, Any],
) -> None:
    """Check that one model has estimator and search-grid config."""
    missing_keys = [
        key for key in REQUIRED_MODEL_HYPERPARAM_KEYS if key not in model_hyperparams
    ]
    if missing_keys:
        raise ValueError(
            f"Missing model hyperparameter keys for '{model_name}': {missing_keys}"
        )

    if "type" not in model_hyperparams["estimator"]:
        raise ValueError(f"Missing estimator type for '{model_name}'.")


def select_candidate_model_configs(
    config: dict[str, Any],
    model_hyperparams: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select model-hyperparameter config for each candidate pipeline."""
    candidate_model_configs = []
    for candidate_pipeline in config["candidates"]:
        model_name = candidate_pipeline["model"]
        if model_name not in model_hyperparams:
            raise ValueError(f"Missing model hyperparameters for '{model_name}'.")

        model_config = model_hyperparams[model_name]
        validate_model_hyperparams(model_name, model_config)
        candidate_model_configs.append(model_config)

    return candidate_model_configs


def build_candidate_preprocessors(
    config: dict[str, Any],
    candidate_model_configs: list[dict[str, Any]],
) -> list:
    """Build configured preprocessor for each candidate pipeline."""
    candidate_pipelines = config["candidates"]
    if len(candidate_pipelines) != len(candidate_model_configs):
        raise ValueError("Candidate pipeline and model config counts do not match.")

    preprocessors = []
    for candidate_pipeline, model_config in zip(
        candidate_pipelines,
        candidate_model_configs,
    ):
        model_type = model_config["estimator"]["type"]
        preprocessors.append(
            build_preprocessor(
                preprocessing_config=candidate_pipeline["preprocessing"],
                force_dense_output=model_requires_dense_input(model_type),
            )
        )

    return preprocessors


def build_candidate_models(candidate_model_configs: list[dict[str, Any]]) -> list:
    """Build model for each candidate pipeline."""
    return [
        build_model(candidate_model_config["estimator"])
        for candidate_model_config in candidate_model_configs
    ]


def build_candidate_grid_entries(
    candidate_model_configs: list[dict[str, Any]],
    candidate_preprocessors: list,
    candidate_models: list,
) -> list[dict]:
    """Build GridSearchCV grid entries from prepared candidate components."""
    if not (
        len(candidate_model_configs)
        == len(candidate_preprocessors)
        == len(candidate_models)
    ):
        raise ValueError("Candidate model, preprocessor, and config counts do not match.")

    grid_entries = []
    for model_config, preprocessor, model in zip(
        candidate_model_configs,
        candidate_preprocessors,
        candidate_models,
    ):
        grid_entries.append(
            {
                PREPROCESS_STEP_NAME: [preprocessor],
                MODEL_STEP_NAME: [model],
                **model_config["search_grid"],
            }
        )

    return grid_entries


def grid_search_with_stratifiedkfold_cv(
    placeholder_pipeline: Pipeline,
    candidate_grid_entries: list[dict],
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Run GridSearchCV with StratifiedKFold CV on the training split."""
    stratified_kfold = StratifiedKFold(
        n_splits=config["search"]["cv_folds"],
        shuffle=True,
        random_state=config["experiment"]["random_state"],
    )
    grid_search = GridSearchCV(
        estimator=placeholder_pipeline,
        param_grid=candidate_grid_entries,
        scoring=config["search"]["scoring"],
        cv=stratified_kfold,
        n_jobs=config["search"].get("n_jobs"),
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search
