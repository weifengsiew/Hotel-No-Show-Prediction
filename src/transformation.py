"""Transformation functions for the hotel no-show ETL pipeline."""

import logging
from itertools import chain

from pyspark.sql.functions import abs, coalesce, col, create_map, initcap, lit, regexp_extract, trim, upper, when
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, LongType, StringType


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def drop_duplicate_rows(df, row_id: str):
    """Drop duplicate rows based on a unique row identifier."""
    df_deduped = df.dropDuplicates([row_id])
    duplicate_count = df.count() - df_deduped.count()

    if duplicate_count > 0:
        logger.info("Removed %s duplicate rows based on %s.", duplicate_count, row_id)

    return df_deduped

# -----------------------------------------------------------------------------

def standardize_string_format(
    df,
    except_columns: list[str],
):
    """Trim spaces and title case strings: e.g. '  john doe  ' -> 'John Doe'."""
    string_columns = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, StringType) and field.name not in except_columns
    ]

    for column in string_columns:
        df = df.withColumn(column, initcap(trim(col(column))))

    logger.info("Standardized %s string columns: %s", len(string_columns), string_columns)
    return df


# -----------------------------------------------------------------------------

def standardize_price_string_format(df):
    """Trim spaces and uppercase price strings: e.g. '  usd$ 100.00  ' -> 'USD$ 100.00'."""
    df = df.withColumn("price", upper(trim(col("price"))))
    logger.info("Standardized price string format.")
    return df

# -----------------------------------------------------------------------------

def get_missing_value_rows_for_column(field, string_type_missing_values: list[str]):
    """Identify rows where a column is null or contains string type missing values."""
    if isinstance(field.dataType, StringType):
        rows_missing_value_for_column = col(field.name).isNull() | col(field.name).isin(string_type_missing_values)
    else:
        rows_missing_value_for_column = col(field.name).isNull()

    return rows_missing_value_for_column


# -----------------------------------------------------------------------------

def drop_missing_value_rows(
    df,
    except_columns: list[str],
    string_type_missing_values: list[str],
):
    """Drop rows with missing values, except those in specified columns."""
    missing_value_rows = lit(False)

    for field in df.schema.fields:
        if field.name in except_columns:
            continue

        rows_missing_value_for_column = get_missing_value_rows_for_column(
            field=field,
            string_type_missing_values=string_type_missing_values,
        )
        missing_value_rows = missing_value_rows | rows_missing_value_for_column

    df_cleaned = df.filter(~missing_value_rows)
    removed_count = df.count() - df_cleaned.count()

    if removed_count > 0:
        logger.info(
            "Removed %s rows with missing values, excluding columns: %s.",
            removed_count,
            except_columns,
        )

    return df_cleaned


# -----------------------------------------------------------------------------

def standardize_missing_values(
    df,
    columns: list[str],
    string_type_missing_values: list[str],
):
    """Replace missing values in specified columns with "Null"."""
    for column in columns:
        rows_missing_value_for_column = get_missing_value_rows_for_column(
            field=df.schema[column],
            string_type_missing_values=string_type_missing_values,
        )
        df = df.withColumn(column, when(rows_missing_value_for_column, lit("Null")).otherwise(col(column)))

    logger.info("Standardized missing values with 'Null' for columns: %s.", columns)
    return df


# -----------------------------------------------------------------------------

def create_price_is_missing_column(df):
    """Create a column that marks standardized missing price values."""
    df = df.withColumn("price_is_missing", col("price") == lit("Null"))
    logger.info("Created price_is_missing flag.")
    return df


# -----------------------------------------------------------------------------

def get_expected_price_pattern():
    """Return the expected price pattern."""
    expected_price_pattern = r"^([A-Z]+)\$\s(\d+(?:\.\d{1,2})?)$"
    return expected_price_pattern

# -----------------------------------------------------------------------------

def create_price_matches_expected_pattern_column(df):
    """Create a column that marks whether price matches the expected pattern."""
    expected_price_pattern = get_expected_price_pattern()
    price_text = col("price")

    df = df.withColumn("price_matches_expected_pattern", when(col("price").isNull(), lit(False)).otherwise(price_text.rlike(expected_price_pattern)))
    logger.info("Created price_matches_expected_pattern flag for price pattern.")
    return df


# -----------------------------------------------------------------------------

def create_price_currency_and_amount_columns(df):
    """Create price currency and amount columns for valid price patterns."""
    expected_price_pattern = get_expected_price_pattern()
    price_text = col("price")

    df = df.withColumn("price_currency", when(col("price_matches_expected_pattern"), regexp_extract(price_text, expected_price_pattern, 1)))
    df = df.withColumn("price_amount", when(col("price_matches_expected_pattern"), regexp_extract(price_text, expected_price_pattern, 2).cast(DoubleType())))

    logger.info("Created price_currency and price_amount from valid price values.")
    return df


# -----------------------------------------------------------------------------

def convert_negative_days_to_positive(df, day_column: str):
    """Convert negative day numbers to positive numbers."""
    df = df.withColumn(day_column, abs(col(day_column)))
    logger.info("Cleaned negative values in column %s by taking absolute values.", day_column)
    return df


# -----------------------------------------------------------------------------

def convert_num_adults_words_to_digits(df):
    """Convert num_adults values written as words into digits."""
    words_to_digits = {
        "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4",
        "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9", "Ten": "10",
    }

    mapping_expr = create_map([lit(x) for x in chain(*words_to_digits.items())])
    df = df.withColumn("num_adults", coalesce(mapping_expr[col("num_adults")], col("num_adults")))

    logger.info("Cleaned num_adults by mapping word numbers to digit strings.")
    return df


# -----------------------------------------------------------------------------

def convert_column_types_to_expected(df):
    """Convert each no-show column to the expected data type."""
    type_mapping = {
        "booking_id": LongType(), "no_show": IntegerType(),
        "booking_month": StringType(), "arrival_month": StringType(), "checkout_month": StringType(),
        "arrival_day": IntegerType(), "checkout_day": IntegerType(),
        "country": StringType(), "num_children": IntegerType(), "num_adults": IntegerType(),
        "first_time": StringType(), "platform": StringType(), "branch": StringType(), "room": StringType(),
        "price": StringType(), "price_currency": StringType(), "price_amount": DoubleType(),
        "price_matches_expected_pattern": BooleanType(), "price_is_missing": BooleanType(),
    }

    for column, target_type in type_mapping.items():
        df = df.withColumn(column, col(column).cast(target_type))

    logger.info("Cast no-show columns to expected Spark data types.")
    return df


# -----------------------------------------------------------------------------

def transform_noshow_data(
    df,
):
    """Clean and reshape the no-show data before validation."""
    string_type_missing_values = ["", "Na", "N/a", "Null", "Nan", "NA", "N/A", "NULL", "NAN"]

    df = drop_duplicate_rows(df=df, row_id="booking_id")

    df = standardize_string_format(df=df, except_columns=["price"])
    df = standardize_price_string_format(df=df)
    
    df = drop_missing_value_rows(df=df, except_columns=["room", "price"], string_type_missing_values=string_type_missing_values)
    df = standardize_missing_values(df=df, columns=["room", "price"], string_type_missing_values=string_type_missing_values)
    df = create_price_is_missing_column(df=df)

    df = convert_negative_days_to_positive(df=df, day_column="arrival_day")
    df = convert_negative_days_to_positive(df=df, day_column="checkout_day")
    df = convert_num_adults_words_to_digits(df=df)

    df = create_price_matches_expected_pattern_column(df=df)
    df = create_price_currency_and_amount_columns(df=df)
    
    df = convert_column_types_to_expected(df=df)

    return df
