"""Model search pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.ml_experiment.nodes import (
    build_candidate_models,
    build_candidate_preprocessors,
    build_candidate_grid_entries,
    build_placeholder_pipeline,
    grid_search_with_stratifiedkfold_cv,
    select_candidate_model_configs,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model search pipeline."""
    return Pipeline(
        [
            node(
                func=build_placeholder_pipeline,
                inputs=None,
                outputs="placeholder_pipeline",
                name="build_placeholder_pipeline",
            ),
            node(
                func=select_candidate_model_configs,
                inputs=["params:no_show_experiment", "params:model_hyperparams"],
                outputs="candidate_model_configs",
                name="select_candidate_model_configs",
            ),
            node(
                func=build_candidate_preprocessors,
                inputs=["params:no_show_experiment", "candidate_model_configs"],
                outputs="candidate_preprocessors",
                name="build_candidate_preprocessors",
            ),
            node(
                func=build_candidate_models,
                inputs="candidate_model_configs",
                outputs="candidate_models",
                name="build_candidate_models",
            ),
            node(
                func=build_candidate_grid_entries,
                inputs=[
                    "candidate_model_configs",
                    "candidate_preprocessors",
                    "candidate_models",
                ],
                outputs="candidate_grid_entries",
                name="build_candidate_grid_entries",
            ),
            node(
                func=grid_search_with_stratifiedkfold_cv,
                inputs=[
                    "placeholder_pipeline",
                    "candidate_grid_entries",
                    "params:no_show_experiment",
                    "X_train",
                    "y_train",
                ],
                outputs="grid_search",
                name="grid_search_with_stratifiedkfold_cv",
            ),
        ]
    )
