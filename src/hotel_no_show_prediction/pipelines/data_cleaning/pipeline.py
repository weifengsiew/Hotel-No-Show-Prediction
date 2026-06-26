"""Data cleaning pipeline"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.data_cleaning.nodes import (
    convert_numeric_column_types,
    drop_duplicate_rows,
    drop_missing_target_rows,
    drop_rows_with_low_missingness,
    num_adults_words_to_numbers,
    set_invalid_day_of_month_to_missing,
    standardize_month_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the cleaning pipeline."""
    return Pipeline(
        [
            node(
                func=drop_duplicate_rows,
                inputs="raw_noshow",
                outputs="noshow_deduplicated",
                name="drop_duplicate_rows",
            ),
            node(
                func=drop_missing_target_rows,
                inputs="noshow_deduplicated",
                outputs="noshow_target_available",
                name="drop_missing_target_rows",
            ),
            node(
                func=drop_rows_with_low_missingness,
                inputs="noshow_target_available",
                outputs="noshow_low_missingness_rows_dropped",
                name="drop_rows_with_low_missingness",
            ),
            node(
                func=num_adults_words_to_numbers,
                inputs="noshow_low_missingness_rows_dropped",
                outputs="noshow_num_adults_standardized",
                name="num_adults_words_to_numbers",
            ),
            node(
                func=convert_numeric_column_types,
                inputs="noshow_num_adults_standardized",
                outputs="noshow_numeric_types_converted",
                name="convert_numeric_column_types",
            ),
            node(
                func=standardize_month_columns,
                inputs="noshow_numeric_types_converted",
                outputs="noshow_month_standardized",
                name="standardize_month_columns",
            ),
            node(
                func=set_invalid_day_of_month_to_missing,
                inputs="noshow_month_standardized",
                outputs="noshow_cleaned",
                name="set_invalid_day_of_month_to_missing",
            ),
        ]
    )
