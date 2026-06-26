"""Validation failure reporting helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

CSV_COLUMNS = [
    "expectation_type",
    "affected_field",
    "unexpected_count",
    "unexpected_percent",
    "examples",
]


def get_affected_field(expectation_kwargs: dict[str, Any]) -> str:
    """Get field affected by a failed expectation."""
    if "column" in expectation_kwargs:
        return expectation_kwargs["column"]
    if "column_list" in expectation_kwargs:
        return ", ".join(expectation_kwargs["column_list"])
    return "table"


def get_unexpected_percent(expectation_result: dict[str, Any]) -> float | None:
    """Return the percentage of rows with failed expectations."""
    if "unexpected_percent_total" in expectation_result:
        return expectation_result["unexpected_percent_total"]
    return None


def format_examples(examples: list[Any]) -> str:
    """Format unique failed values as a compact CSV cell."""
    unique_examples = pd.Series(examples, dtype="object").drop_duplicates().head(30)
    return "; ".join(unique_examples.astype(str))


def extract_failed_expectations(validation_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert failed Great Expectations results into CSV-ready rows."""
    failed_rows = []
    for expectation_result in validation_results["results"]:
        if expectation_result["success"]:
            continue

        expectation_config = expectation_result["expectation_config"]
        expectation_kwargs = expectation_config["kwargs"]
        result_details = expectation_result.get("result", {})

        failed_rows.append(
            {
                "expectation_type": expectation_config["type"],
                "affected_field": get_affected_field(expectation_kwargs),
                "unexpected_count": result_details.get("unexpected_count"),
                "unexpected_percent": get_unexpected_percent(result_details),
                "examples": format_examples(
                    result_details.get("partial_unexpected_list", [])
                ),
            }
        )

    return failed_rows


def format_failed_expectations_report(
    failed_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Format failed expectations as the persisted CSV DataFrame."""
    return pd.DataFrame(failed_rows, columns=CSV_COLUMNS)
