"""Great Expectations validation functions for the hotel no-show ETL pipeline."""

import calendar

import great_expectations as gx
from great_expectations import expectations as gxe
import pycountry


MONTH_DAY_LIMITS = {
    month_name: calendar.monthrange(2024, month_number)[1]
    for month_number, month_name in enumerate(calendar.month_name)
    if month_name
}


# =============================================================================
# Expectation definitions
# =============================================================================

def get_null_values_expectation():
    """Return columns allowed to contain null values."""
    return {
        "room",  # too many missing values to remove
        "price",  # too many missing values to remove
        "price_currency",  # null when price is missing or invalid
        "price_amount",  # null when price is missing or invalid
    }


# -----------------------------------------------------------------------------

def get_noshow_column_expectations():
    """Return validation expectations organized by column."""
    months = list(calendar.month_name)[1:]
    countries = [country.name for country in pycountry.countries]
    currency_codes = [currency.alpha_3 for currency in pycountry.currencies]

    return {
        # row identifier
        "booking_id": {"types": ["IntegerType", "LongType"]},  # e.g. 116

        # target variable for prediction
        "no_show": {"types": ["IntegerType"], "allowed_values": [0, 1]},  # binary encoding

        # booking-related variables
        "booking_month": {"types": ["StringType"], "allowed_values": months},  # official month, e.g. "January"
        "platform": {"types": ["StringType"], "allowed_values": ["Agent", "Email", "Phone", "Website"]},  # from data exploration

        # date-related variables
        "arrival_month": {"types": ["StringType"], "allowed_values": months},  # official month, e.g. "January"
        "checkout_month": {"types": ["StringType"], "allowed_values": months},  # official month, e.g. "January"
        "arrival_day": {"types": ["IntegerType"], "allowed_values": list(range(1, 32))},  # valid day-of-month range
        "checkout_day": {"types": ["IntegerType"], "allowed_values": list(range(1, 32))},  # valid day-of-month range

        # guest-related variables
        "country": {"types": ["StringType"], "allowed_values": countries},  # official country names, e.g. "Singapore"
        "num_children": {"types": ["IntegerType"], "allowed_values": list(range(0, 11))},  # integer count; cannot have 1.5 children
        "num_adults": {"types": ["IntegerType"], "allowed_values": list(range(0, 11))},  # integer count; cannot have 1.5 adults
        "first_time": {"types": ["StringType"], "allowed_values": ["Yes", "No"]},  # from data exploration

        # accommodation-related variables
        "branch": {"types": ["StringType"], "allowed_values": ["Changi", "Orchard"]},  # from data exploration
        "room": {"types": ["StringType"], "allowed_values": ["King", "Queen", "Single", "President Suite"]},  # from data exploration

        # price-related variables
        "price": {"types": ["StringType"], "regex": r"^[A-Z]+\$\s\d+(\.\d{1,2})?$"},  # e.g. "SGD$ 100.01"
        "price_currency": {"types": ["StringType"], "allowed_values": currency_codes + [None]},  # official code, e.g. "SGD"; null if price missing/invalid
        "price_amount": {"types": ["DoubleType"], "min_value": 0},  # price amount should not be negative
        "is_valid_price": {"types": ["BooleanType"], "allowed_values": [True, False]},  # whether price could be parsed
    }


# -----------------------------------------------------------------------------

def add_no_duplication_expectation(suite):
    """Add no-duplicate-row expectations."""
    # booking_id is the row identifier, so duplicates would indicate duplicated bookings
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="booking_id"))
    return suite


# -----------------------------------------------------------------------------

def add_column_expectations(suite, df):
    """Add expectations one variable at a time."""
    nullable_columns = get_null_values_expectation()
    column_expectations = get_noshow_column_expectations()

    for column_name, expectations in column_expectations.items():
        if column_name not in df.columns:  # skip expectation definitions for absent columns
            continue

        if column_name not in nullable_columns:  # default: variables should be complete
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column_name))

        if "types" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToBeInTypeList(column=column_name, type_list=expectations["types"]))

        if "allowed_values" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column=column_name, value_set=expectations["allowed_values"]))

        if "regex" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(column=column_name, regex=expectations["regex"]))

        if "min_value" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column=column_name, min_value=expectations["min_value"]))

    return suite


# -----------------------------------------------------------------------------

def add_month_day_expectations(suite, df):
    """Add month-specific day-of-month expectations."""
    for month_column, day_column in (
        ("arrival_month", "arrival_day"),
        ("checkout_month", "checkout_day"),
    ):
        if month_column not in df.columns or day_column not in df.columns:
            continue

        for month_name, max_day in MONTH_DAY_LIMITS.items():
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(
                    column=day_column,
                    max_value=max_day,
                    row_condition=f'{month_column} == "{month_name}"',
                    condition_parser="spark",
                )
            )

    return suite


# =============================================================================
# Great Expectations helper functions
# =============================================================================

def get_gx_context():
    """Return the Great Expectations Data Context."""
    return gx.get_context()


# -----------------------------------------------------------------------------

def create_gx_batch(df, context=None):
    """Create a Great Expectations batch from a Spark DataFrame."""
    if context is None:
        context = get_gx_context()

    data_source = context.data_sources.add_or_update_spark(name="noshow_spark_datasource")
    data_asset = data_source.add_dataframe_asset(name="noshow_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("noshow_batch_definition")

    return batch_definition.get_batch(batch_parameters={"dataframe": df})


# -----------------------------------------------------------------------------

def build_noshow_expectation_suite(df):
    """Build the Great Expectations suite for the cleaned no-show DataFrame."""
    suite = gx.ExpectationSuite(name="noshow_expectation_suite")
    suite = add_no_duplication_expectation(suite)
    suite = add_column_expectations(suite, df)
    suite = add_month_day_expectations(suite, df)
    return suite


# -----------------------------------------------------------------------------

def validate_noshow_with_gx(df):
    """Validate the cleaned no-show DataFrame with Great Expectations."""
    context = get_gx_context()
    batch = create_gx_batch(df, context=context)
    suite = build_noshow_expectation_suite(df)
    return batch.validate(suite)


# -----------------------------------------------------------------------------

def validate_noshow_data(df):
    """Validate the cleaned no-show DataFrame."""
    validation_results = validate_noshow_with_gx(df)
    return df, validation_results
