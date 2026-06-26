"""Project pipeline registry."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline as modular_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's Kedro pipelines."""
    # Discover each pipeline package under hotel_no_show_prediction.pipelines.
    discovered_pipelines = find_pipelines()

    # Namespace each module pipeline so Kedro-Viz shows the project stages,
    # while keeping dataset names shared across module boundaries.
    default_pipeline = sum(
        (
            modular_pipeline(
                module_pipeline,
                namespace=module_name,
                prefix_datasets_with_namespace=False,
            )
            for module_name, module_pipeline in discovered_pipelines.items()
        )
    )

    # Register one end-to-end graph so Kedro run and Kedro-Viz use the same pipeline.
    return {"__default__": default_pipeline}
