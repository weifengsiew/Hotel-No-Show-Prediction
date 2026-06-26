"""Data validation nodes."""

from __future__ import annotations

from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations import ExpectationSuite

from hotel_no_show_prediction.pipelines.data_validation.expectations import (
    add_basic_expectations as add_basic_expectations_to_suite,
)
from hotel_no_show_prediction.pipelines.data_validation.expectations import (
    add_numeric_range_expectations as add_numeric_range_expectations_to_suite,
)
from hotel_no_show_prediction.pipelines.data_validation.expectations import (
    add_string_expectations as add_string_expectations_to_suite,
)
from hotel_no_show_prediction.pipelines.data_validation.expectations import (
    add_type_expectations as add_type_expectations_to_suite,
)
from hotel_no_show_prediction.pipelines.data_validation.reporting import (
    extract_failed_expectations,
    format_failed_expectations_report,
)

Validator = Any
EXPECTATION_SUITE_NAME = "noshow_validation_suite"


def create_validation_context() -> Any:
    """Create an ephemeral Great Expectations context."""
    return gx.get_context(mode="ephemeral")


def get_validation_table_name(
    validation_config: dict[str, str],
) -> str:
    """Get the table name used for the validation batch asset."""
    return validation_config["table_name"]


def create_dataframe_batch_request(
    validation_context: Any,
    validation_data: pd.DataFrame,
    table_name: str,
) -> tuple[Any, Any]:
    """Create a Great Expectations batch request and return the updated context."""
    data_source = validation_context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name=table_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_table")
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": validation_data}
    )
    return validation_context, batch_request


def build_expectation_suite() -> ExpectationSuite:
    """Build the reusable no-show expectation suite."""
    expectation_suite = gx.ExpectationSuite(name=EXPECTATION_SUITE_NAME)
    add_basic_expectations_to_suite(expectation_suite)
    add_type_expectations_to_suite(expectation_suite)
    add_numeric_range_expectations_to_suite(expectation_suite)
    add_string_expectations_to_suite(expectation_suite)
    return expectation_suite


def create_validator(
    validation_context: Any,
    batch_request: Any,
    expectation_suite: ExpectationSuite,
) -> Validator:
    """Create a Great Expectations Validator from prepared validation inputs."""
    return validation_context.get_validator(
        batch_request=batch_request,
        expectation_suite=expectation_suite,
    )


def run_validation(validator: Validator) -> dict[str, Any]:
    """Run validation and return JSON-serializable results."""
    validation_result = validator.validate()
    return validation_result.to_json_dict()


def extract_failed_expectation_rows(
    validation_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract failed expectation rows from validation results."""
    return extract_failed_expectations(validation_results)


def build_failed_expectations_report(
    failed_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build the failed expectations report DataFrame."""
    return format_failed_expectations_report(failed_rows)
