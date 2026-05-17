"""Great Expectations validation functions for the hotel no-show ETL pipeline."""

import calendar

import great_expectations as gx
from great_expectations import expectations as gxe
import pycountry


# =============================================================================
# Functions to define expectations for noshow dataset
# =============================================================================

# -----------------------------------------------------------------------------

def add_no_duplicate_rows_expectation(suite):

    """Expect no duplicate rows"""
    # booking_id is the row identifier
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="booking_id"))

    return suite

# -----------------------------------------------------------------------------

def get_nullable_columns_expectation():
    
    """Expect nullable price_currency and price_amount"""

    nullable_columns = {
        "price_currency": True,
        "price_amount": True,
    }

    return nullable_columns

# -----------------------------------------------------------------------------

def get_per_column_expectations():
    
    """Per-column expectations based on prior knowledge and data exploration"""

    # official country names, e.g. "Singapore";; 
    countries = [country.name for country in pycountry.countries]
    # official currency codes, e.g. "SGD"
    currency_codes = [currency.alpha_3 for currency in pycountry.currencies]
    # official month names, e.g. "January"
    months = list(calendar.month_name)[1:]  

    per_column_expectations = {
        # row identifier
        "booking_id": {"InTypeList": ["LongType"]},  

        # target variable for prediction
        "no_show": {"InTypeList": ["IntegerType"], "InSet": [0, 1]}, # binary variable; 0 for show, 1 for no-show

        # booking-related variables
        "platform": {"InTypeList": ["StringType"], "InSet": ["Agent", "Email", "Phone", "Website"]},  # observed from exploration

        # date-related variables
        "booking_month": {"InTypeList": ["StringType"], "InSet": months},  # official month names, e.g. "January"
        "arrival_month": {"InTypeList": ["StringType"], "InSet": months},  # official month names, e.g. "January"
        "checkout_month": {"InTypeList": ["StringType"], "InSet": months},  # official month names, e.g. "January"
        "arrival_day": {"InTypeList": ["IntegerType"], "InSet": list(range(1, 32))},  # days from 1 to 31
        "checkout_day": {"InTypeList": ["IntegerType"], "InSet": list(range(1, 32))},  # days from 1 to 31

        # guest-related variables
        "country": {"InTypeList": ["StringType"], "InSet": countries},  # an official country name
        "num_children": {"InTypeList": ["IntegerType"], "InSet": list(range(0, 10))},  # non-negative integers not exceeding 9
        "num_adults": {"InTypeList": ["IntegerType"], "InSet": list(range(0, 10))},  # non-negative integers not exceeding 9
        "first_time": {"InTypeList": ["StringType"], "InSet": ["Yes", "No"]},  # observed from exploration

        # accommodation-related variables
        "branch": {"InTypeList": ["StringType"], "InSet": ["Changi", "Orchard"]},  # observed from exploration
        "room": {"InTypeList": ["StringType"], "InSet": ["King", "Queen", "Single", "President Suite", "Null"]},  # observed from exploration

        # price-related variables
        "price": {"InTypeList": ["StringType"], "MatchRegex": r"^([A-Z]+\$\s\d+(\.\d{1,2})?|Null)$"},  # e.g. "SGD$ 100.01" or "Null"
        "price_currency": {"InTypeList": ["StringType"], "InSet": currency_codes},  # official currency codes e.g. "SGD"
        "price_amount": {"InTypeList": ["DoubleType"], "BeBetween": {"min_value": 0, "max_value": 10000}},  # should be non-negative and within expected range
        "price_matches_expected_pattern": {"InTypeList": ["BooleanType"], "InSet": [True, False]},  # whether price conforms to "SGD$ 100.01"
        "price_is_missing": {"InTypeList": ["BooleanType"], "InSet": [True, False]},  # whether price was standardized as missing
    }

    return per_column_expectations

# -----------------------------------------------------------------------------

def add_month_specific_day_expectations(suite, df):

    """Expect valid days based on month"""
    # matched month and day columns
    month_day_columns  = (("arrival_month", "arrival_day"),
                         ("checkout_month", "checkout_day"))

    # month specific max days e.g. at most 29 days in February 2024 
    month_specific_max_days = {
        month_name: calendar.monthrange(2024, month_number)[1]
        for month_number, month_name in enumerate(calendar.month_name)
        if month_name } 
    
    # expect days between 1 and max_day for each month
    for month_column, day_column in month_day_columns:
        for month_name, max_day in month_specific_max_days.items():
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(
                    column=day_column,
                    max_value=max_day,
                    row_condition=f'{month_column} == "{month_name}"',
                    condition_parser="spark",))

    return suite


# =============================================================================
# Helper functions  to validate noshow dataset based on defined expectations
# =============================================================================

def create_gx_batch(df, context):
    """Create a Great Expectations batch from a Spark DataFrame."""
    data_source = context.data_sources.add_or_update_spark(name="noshow_spark_datasource")
    data_asset = data_source.add_dataframe_asset(name="noshow_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("noshow_batch_definition")

    return batch_definition.get_batch(batch_parameters={"dataframe": df})

# -----------------------------------------------------------------------------

def add_column_expectations(suite, df):
    """Add expectations one column at a time."""

    nullable_columns = get_nullable_columns_expectation()
    per_column_expectations = get_per_column_expectations()

    for column_name, expectations in per_column_expectations.items():

        # skip expectation definitions for absent columns
        if column_name not in df.columns:  
            continue

        #  add non-null expectation for columns not in nullable_columns
        if column_name not in nullable_columns:  
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column_name))

        # add type expectation if specified
        if "InTypeList" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToBeInTypeList(column=column_name, type_list=expectations["InTypeList"]))
        
        # add set of expected values expectation if specified
        if "InSet" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column=column_name, value_set=expectations["InSet"]))

        # add regex expectation if specified
        if "MatchRegex" in expectations:
            suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(column=column_name, regex=expectations["MatchRegex"]))
        
        # add between expectation if specified
        if "BeBetween" in expectations:
            min_value = expectations["BeBetween"]["min_value"]
            max_value = expectations["BeBetween"]["max_value"]
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column=column_name, min_value=min_value,max_value=max_value,)
        )

    return suite

# --------------------------------------------------------

def build_noshow_expectation_suite(df):
    """Build the Great Expectations suite for the cleaned no-show DataFrame."""
    suite = gx.ExpectationSuite(name="noshow_expectation_suite")
    suite = add_no_duplicate_rows_expectation(suite)
    suite = add_column_expectations(suite, df)
    suite = add_month_specific_day_expectations(suite, df)
    return suite


# -----------------------------------------------------------------------------

def validate_noshow_data(df):
    """Validate the cleaned no-show DataFrame."""
    context = gx.get_context()
    batch = create_gx_batch(df, context)
    suite = build_noshow_expectation_suite(df)
    validation_results = batch.validate(suite)
    return df, validation_results
