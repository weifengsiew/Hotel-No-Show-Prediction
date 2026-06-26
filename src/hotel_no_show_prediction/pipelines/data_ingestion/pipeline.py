"""Data ingestion pipeline"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.data_ingestion.nodes import (
    load_raw_noshow,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data loading pipeline."""
    return Pipeline(
        [
            node(
                func=load_raw_noshow,
                inputs="params:raw_data",
                outputs="raw_noshow",
                name="load_raw_noshow",
            ),
        ]
    )
