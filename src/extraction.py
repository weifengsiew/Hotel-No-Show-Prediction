"""Extraction functions for the hotel no-show ETL pipeline."""

import sqlite3
import re

import pandas as pd
from pyspark.sql import SparkSession


VALID_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# -----------------------------------------------------------------------------

def get_spark_session(app_name: str = "HotelNoShowPrediction") -> SparkSession:
    """Create or retrieve a Spark session."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# -----------------------------------------------------------------------------

def extract_noshow_data(
    db_path: str = "data/raw/noshow.db",
    table_name: str = "noshow",
    app_name: str = "HotelNoShowPrediction",
):
    """Extract the no-show SQLite table into a Spark DataFrame."""
    if not VALID_SQL_IDENTIFIER.fullmatch(table_name):
        raise ValueError(f"Invalid SQLite table name: {table_name!r}")

    with sqlite3.connect(db_path) as conn:
        df_pandas = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    return get_spark_session(app_name=app_name).createDataFrame(df_pandas)
