"""Data validation pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from hotel_no_show_prediction.pipelines.data_validation.nodes import (
    build_expectation_suite,
    build_failed_expectations_report,
    create_dataframe_batch_request,
    create_validation_context,
    create_validator,
    extract_failed_expectation_rows,
    get_validation_table_name,
    run_validation,
)

RAW_VALIDATION_INPUTS = {"raw_noshow", "expectation_suite"}
CLEANED_VALIDATION_INPUTS = {"noshow_cleaned", "expectation_suite"}
VALIDATION_OUTPUT_SUFFIXES = (
    "validation_context",
    "validation_table_name",
    "validation_context_with_batch",
    "validation_batch_request",
    "validator",
    "validation_results",
    "failed_expectation_rows",
    "validation_failures",
)


def validation_outputs(prefix: str) -> set[str]:
    """Return exposed validation output datasets for a raw or cleaned run."""
    return {f"{prefix}_{suffix}" for suffix in VALIDATION_OUTPUT_SUFFIXES}


def named_node(func, inputs, outputs):
    """Create a Kedro node named after its function."""
    return node(func, inputs, outputs, name=func.__name__)


def create_expectation_suite_pipeline() -> Pipeline:
    """Create the reusable no-show expectation suite pipeline."""
    return Pipeline(
        [
            named_node(build_expectation_suite, None, "expectation_suite"),
        ]
    )


def create_validator_setup_pipeline(
    *,
    prefix: str,
    validation_data: str,
) -> Pipeline:
    """Create the shared Great Expectations validator setup pipeline."""
    context = f"{prefix}_validation_context"
    table_name = f"{prefix}_validation_table_name"
    context_with_batch = f"{prefix}_validation_context_with_batch"
    batch_request = f"{prefix}_validation_batch_request"

    return Pipeline(
        [
            named_node(create_validation_context, None, context),
            named_node(get_validation_table_name, "params:validation", table_name),
            named_node(
                create_dataframe_batch_request,
                [context, validation_data, table_name],
                [context_with_batch, batch_request],
            ),
            named_node(
                create_validator,
                [context_with_batch, batch_request, "expectation_suite"],
                f"{prefix}_validator",
            ),
        ]
    )


def create_validation_reporting_pipeline(
    *,
    prefix: str,
) -> Pipeline:
    """Create the shared validation execution and reporting pipeline."""
    validation_results = f"{prefix}_validation_results"
    failed_rows = f"{prefix}_failed_expectation_rows"

    return Pipeline(
        [
            named_node(run_validation, f"{prefix}_validator", validation_results),
            named_node(extract_failed_expectation_rows, validation_results, failed_rows),
            named_node(
                build_failed_expectations_report,
                failed_rows,
                f"{prefix}_validation_failures",
            ),
        ]
    )


def create_raw_data_validation_pipeline(**kwargs) -> Pipeline:
    """Create the raw data validation pipeline."""
    return create_validator_setup_pipeline(
        prefix="raw",
        validation_data="raw_noshow",
    ) + create_validation_reporting_pipeline(
        prefix="raw",
    )


def create_cleaned_data_validation_pipeline(**kwargs) -> Pipeline:
    """Create the cleaned data validation pipeline."""
    return create_validator_setup_pipeline(
        prefix="cleaned",
        validation_data="noshow_cleaned",
    ) + create_validation_reporting_pipeline(
        prefix="cleaned",
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data validation pipeline."""
    return (
        create_expectation_suite_pipeline()
        + modular_pipeline(
            create_raw_data_validation_pipeline(),
            namespace="raw_data_validation",
            inputs=RAW_VALIDATION_INPUTS,
            outputs=validation_outputs("raw"),
            parameters={"validation"},
        )
        + modular_pipeline(
            create_cleaned_data_validation_pipeline(),
            namespace="cleaned_data_validation",
            inputs=CLEANED_VALIDATION_INPUTS,
            outputs=validation_outputs("cleaned"),
            parameters={"validation"},
        )
    )
