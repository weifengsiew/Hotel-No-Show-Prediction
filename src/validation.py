"""Great Expectations validation functions for the hotel no-show ETL pipeline."""

import calendar

import pycountry


def get_noshow_validation_rules():
    """Return expected types, allowed values, and regex rules."""
    expected_types_dict = {
        "booking_id": ["IntegerType", "LongType"],
        "no_show": ["IntegerType"],
        "booking_month": ["StringType"],
        "arrival_month": ["StringType"],
        "checkout_month": ["StringType"],
        "arrival_day": ["IntegerType"],
        "checkout_day": ["IntegerType"],
        "country": ["StringType"],
        "num_children": ["IntegerType"],
        "num_adults": ["IntegerType"],
        "first_time": ["StringType"],
        "platform": ["StringType"],
        "branch": ["StringType"],
        "room": ["StringType"],
        "price": ["StringType"],
        "price_currency": ["StringType"],
        "price_amount": ["DoubleType"],
        "is_valid_price": ["BooleanType"],
    }

    valid_values_dict = {
        "no_show": [0, 1],
        "booking_month": list(calendar.month_name)[1:],
        "arrival_month": list(calendar.month_name)[1:],
        "checkout_month": list(calendar.month_name)[1:],
        "arrival_day": list(range(1, 32)),
        "checkout_day": list(range(1, 32)),
        "country": [country.name for country in pycountry.countries],
        "num_children": list(range(0, 11)),
        "num_adults": list(range(0, 11)),
        "first_time": ["Yes", "No"],
        "platform": ["Agent", "Email", "Phone", "Website"],
        "branch": ["Changi", "Orchard"],
        "room": ["King", "Queen", "Single", "President Suite"],
        "is_valid_price": [True, False],
        "price_currency": [currency.alpha_3 for currency in pycountry.currencies] + [None],
    }
    
    regex_rules_dict = {
        "price": r"^[A-Z]+\$\s\d+(\.\d{1,2})?$",
    }

    return expected_types_dict, valid_values_dict, regex_rules_dict


def get_gx_context():
    """Return the Great Expectations Data Context."""
    import great_expectations as gx

    return gx.get_context()


def create_gx_batch(df, context=None):
    """Create a Great Expectations batch from a Spark DataFrame."""
    if context is None:
        context = get_gx_context()

    data_source = context.data_sources.add_or_update_spark(
        name="noshow_spark_datasource",
    )

    data_asset = data_source.add_dataframe_asset(
        name="noshow_data_asset",
    )

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "noshow_batch_definition",
    )

    return batch_definition.get_batch(batch_parameters={"dataframe": df})


def build_noshow_expectation_suite(df):
    """Build the Great Expectations suite for the cleaned no-show DataFrame."""
    import great_expectations as gx
    from great_expectations import expectations as gxe

    expected_types_dict, valid_values_dict, regex_rules_dict = (
        get_noshow_validation_rules()
    )

    suite = gx.ExpectationSuite(name="noshow_expectation_suite")

    nullable_columns = {
        "room",  # preserved as null for downstream tree-based models
        "price",  # preserved as null when missing in raw data
        "price_currency",  # null when price is missing or invalid
        "price_amount",  # null when price is missing or invalid
    }

    for column_name in df.columns:
        if column_name not in nullable_columns:
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column_name))

    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="booking_id"))

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="price_amount",
            min_value=0,
        )
    )

    for column_name, expected_types in expected_types_dict.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=column_name,
                type_list=expected_types,
            )
        )

    for column_name, valid_values in valid_values_dict.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column_name,
                value_set=valid_values,
            )
        )

    for column_name, regex_pattern in regex_rules_dict.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToMatchRegex(
                column=column_name,
                regex=regex_pattern,
            )
        )

    return suite


def validate_noshow_with_gx(df):
    """Validate the cleaned no-show DataFrame with Great Expectations."""
    context = get_gx_context()
    batch = create_gx_batch(df, context=context)
    suite = build_noshow_expectation_suite(df)

    return batch.validate(suite)


def validate_noshow_data(df):
    """Validate the cleaned no-show DataFrame."""
    validation_results = validate_noshow_with_gx(df)
    return df, validation_results
