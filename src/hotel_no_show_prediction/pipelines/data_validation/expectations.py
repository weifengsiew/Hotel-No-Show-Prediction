"""Great Expectations rules for hotel no-show data."""

from __future__ import annotations

import calendar
from typing import Any

from great_expectations import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToBeUnique,
    ExpectColumnValuesToMatchRegex,
    ExpectColumnValuesToNotBeNull,
    ExpectCompoundColumnsToBeUnique,
    ExpectTableColumnsToMatchOrderedList,
)

Suite = ExpectationSuite | Any

NO_SHOW_COLUMNS = [
    "booking_id",
    "no_show",
    "branch",
    "booking_month",
    "arrival_month",
    "arrival_day",
    "checkout_month",
    "checkout_day",
    "country",
    "first_time",
    "room",
    "price",
    "platform",
    "num_adults",
    "num_children",
]
INTEGER_COLUMNS = [
    "booking_id",
    "no_show",
    "arrival_day",
    "checkout_day",
    "num_adults",
    "num_children",
]
STRING_COLUMNS = [
    "branch",
    "booking_month",
    "arrival_month",
    "checkout_month",
    "country",
    "first_time",
    "room",
    "price",
    "platform",
]
MONTH_VALUES = list(calendar.month_name)[1:]


def add_expectation(expectation_suite: Suite, expectation: Any) -> None:
    """Add an expectation to a suite without requiring an active GX context."""
    expectation_suite.expectations.append(expectation)


def add_basic_expectations(expectation_suite: Suite) -> None:
    """Add basic table, uniqueness, and non-null expectations."""
    add_expectation(
        expectation_suite,
        ExpectTableColumnsToMatchOrderedList(column_list=NO_SHOW_COLUMNS),
    )
    add_expectation(
        expectation_suite,
        ExpectCompoundColumnsToBeUnique(column_list=NO_SHOW_COLUMNS),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeUnique(column="booking_id"),
    )

    for column_name in NO_SHOW_COLUMNS:
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToNotBeNull(column=column_name),
        )


def add_type_expectations(expectation_suite: Suite) -> None:
    """Add column type expectations."""
    for column_name in INTEGER_COLUMNS:
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToBeOfType(column=column_name, type_="int64"),
        )

    for column_name in STRING_COLUMNS:
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToBeOfType(column=column_name, type_="str"),
        )


def add_numeric_range_expectations(expectation_suite: Suite) -> None:
    """Add numeric range expectations."""
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="no_show", value_set=[0.0, 1.0, 0, 1]),
    )
    for column_name in ["num_adults", "num_children"]:
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToBeBetween(
                column=column_name,
                min_value=0,
                max_value=5,
                catch_exceptions=True,
            ),
        )

    for month_number, month_name in enumerate(MONTH_VALUES, start=1):
        max_day = calendar.monthrange(2024, month_number)[1]
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToBeBetween(
                column="arrival_day",
                min_value=1,
                max_value=max_day,
                row_condition=f'arrival_month == "{month_name}"',
                condition_parser="pandas",
                catch_exceptions=True,
            ),
        )
        add_expectation(
            expectation_suite,
            ExpectColumnValuesToBeBetween(
                column="checkout_day",
                min_value=1,
                max_value=max_day,
                row_condition=f'checkout_month == "{month_name}"',
                condition_parser="pandas",
                catch_exceptions=True,
            ),
        )


def add_string_expectations(expectation_suite: Suite) -> None:
    """Add allowed-value and string-format expectations."""
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="branch", value_set=["Changi", "Orchard"]),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="booking_month", value_set=MONTH_VALUES),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="arrival_month", value_set=MONTH_VALUES),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="checkout_month", value_set=MONTH_VALUES),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(
            column="country",
            value_set=[
                "Australia",
                "China",
                "India",
                "Indonesia",
                "Japan",
                "Malaysia",
                "Singapore",
            ],
        ),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(column="first_time", value_set=["Yes", "No"]),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(
            column="room",
            value_set=["Single", "Queen", "King", "President Suite"],
        ),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToBeInSet(
            column="platform",
            value_set=["Website", "Email", "Agent", "Phone"],
        ),
    )
    add_expectation(
        expectation_suite,
        ExpectColumnValuesToMatchRegex(
            column="price",
            regex=r"^(?:SGD|USD)\$\s*\d+(?:\.\d+)?$",
        ),
    )
