"""Transformation functions for the hotel no-show ETL pipeline."""

import logging
from itertools import chain

from pyspark.sql.functions import abs, coalesce, col, create_map, initcap, lit, regexp_extract, trim, upper, when
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType


logger = logging.getLogger(__name__)
DEFAULT_MISSING_VALUES = ["", "Na", "N/a", "Null", "Nan"]


# -----------------------------------------------------------------------------

def drop_duplicate_rows(df, row_id: str):
    """Drop duplicate rows based on a unique row identifier."""
    df_deduped = df.dropDuplicates([row_id])
    duplicate_count = df.count() - df_deduped.count()

    if duplicate_count > 0:
        logger.info("Removed %s duplicate rows based on %s.", duplicate_count, row_id)

    return df_deduped


# -----------------------------------------------------------------------------

def drop_rows_with_missing_values(df, column: str, possible_missing_values: list[str]):
    """Drop rows with nulls or string-style missing values in one column."""
    condition = col(column).isNull()

    if isinstance(df.schema[column].dataType, StringType):
        condition = condition | trim(col(column)).isin(possible_missing_values)

    df_cleaned = df.filter(~condition)
    removed_count = df.count() - df_cleaned.count()

    if removed_count > 0:
        logger.info("Removed %s rows with missing values in column %s.", removed_count, column)

    return df_cleaned


# -----------------------------------------------------------------------------

def normalize_categorical_text_columns(
    df,
    columns: list[str],
    possible_missing_values: list[str] | None = None,
):
    """Normalize selected categorical text columns while preserving missing values as null."""
    possible_missing_values = possible_missing_values or DEFAULT_MISSING_VALUES

    for column_name in columns:
        missing_condition = col(column_name).isNull() | trim(col(column_name)).isin(possible_missing_values)
        df = df.withColumn(column_name, when(missing_condition, lit(None)).otherwise(initcap(trim(col(column_name)))))

    logger.info("Normalized %s categorical text columns: %s", len(columns), columns)
    return df


# -----------------------------------------------------------------------------

def normalize_price_currency_code(
    df,
    price_column: str = "price",
    possible_missing_values: list[str] | None = None,
):
    """Trim and uppercase non-missing price values while preserving missing values as null."""
    possible_missing_values = possible_missing_values or DEFAULT_MISSING_VALUES
    missing_condition = col(price_column).isNull() | trim(col(price_column)).isin(possible_missing_values)

    df = df.withColumn(price_column, when(missing_condition, lit(None)).otherwise(upper(trim(col(price_column)))))
    logger.info("Normalized non-missing %s values and preserved missing values as null.", price_column)

    return df


# -----------------------------------------------------------------------------

def parse_price_column(
    df,
    price_column: str = "price",
    currency_column: str = "price_currency",
    amount_column: str = "price_amount",
    is_valid_column: str = "is_valid_price",
):
    """Parse the price column into currency and numeric amount columns."""
    price_pattern = r"^([A-Z]+)\$\s(\d+(?:\.\d{1,2})?)$"
    price_text = trim(col(price_column))

    df = df.withColumn(is_valid_column, when(col(price_column).isNull(), lit(False)).otherwise(price_text.rlike(price_pattern)))
    df = df.withColumn(currency_column, when(col(is_valid_column), regexp_extract(price_text, price_pattern, 1)))
    df = df.withColumn(amount_column, when(col(is_valid_column), regexp_extract(price_text, price_pattern, 2).cast(DoubleType())))

    logger.info("Parsed %s into %s and %s; added %s validity flag.", price_column, currency_column, amount_column, is_valid_column)
    return df


# -----------------------------------------------------------------------------

def convert_negative_days_to_positive(df, day_column: str):
    """Convert negative day values to positive values using absolute values."""
    df = df.withColumn(day_column, abs(col(day_column)))
    logger.info("Cleaned negative values in column %s by taking absolute values.", day_column)
    return df


# -----------------------------------------------------------------------------

def convert_num_adults_words_to_digits(df, num_adults_column: str):
    """Convert word-number adult counts to digit strings."""
    number_mapping = {
        "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4",
        "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9", "Ten": "10",
    }

    mapping_expr = create_map([lit(x) for x in chain(*number_mapping.items())])
    df = df.withColumn(num_adults_column, coalesce(mapping_expr[trim(col(num_adults_column))], col(num_adults_column)))

    logger.info("Cleaned num_adults by mapping word numbers to digit strings.")
    return df


# -----------------------------------------------------------------------------

def cast_noshow_column_types(df):
    """Cast no-show columns to their expected Spark data types."""
    type_mapping = {
        "booking_id": IntegerType(), "no_show": IntegerType(),
        "booking_month": StringType(), "arrival_month": StringType(), "checkout_month": StringType(),
        "arrival_day": IntegerType(), "checkout_day": IntegerType(),
        "country": StringType(), "num_children": IntegerType(), "num_adults": IntegerType(),
        "first_time": StringType(), "platform": StringType(), "branch": StringType(), "room": StringType(),
        "price": StringType(), "price_currency": StringType(), "price_amount": DoubleType(), "is_valid_price": BooleanType(),
    }

    for column_name, target_type in type_mapping.items():
        df = df.withColumn(column_name, col(column_name).cast(target_type))

    logger.info("Cast no-show columns to expected Spark data types.")
    return df


# -----------------------------------------------------------------------------

def transform_noshow_data(
    df,
    row_id: str = "booking_id",
    possible_missing_values: list[str] | None = None,
):
    """Run the full transformation sequence before validation."""
    possible_missing_values = possible_missing_values or DEFAULT_MISSING_VALUES
    df = drop_duplicate_rows(df, row_id=row_id)

    columns_to_check = [field.name for field in df.schema.fields if field.name not in ["room", "price"]]
    for column_name in columns_to_check:
        df = drop_rows_with_missing_values(df, column=column_name, possible_missing_values=possible_missing_values)

    categorical_string_columns = [
        "booking_month", "arrival_month", "checkout_month", "country", "first_time",
        "platform", "branch", "room", "num_adults",
    ]

    df = normalize_categorical_text_columns(df, categorical_string_columns, possible_missing_values=possible_missing_values)
    df = normalize_price_currency_code(df, price_column="price", possible_missing_values=possible_missing_values)
    df = parse_price_column(df, "price")
    df = convert_negative_days_to_positive(df, "arrival_day")
    df = convert_negative_days_to_positive(df, "checkout_day")
    df = convert_num_adults_words_to_digits(df, "num_adults")
    df = cast_noshow_column_types(df)

    return df
