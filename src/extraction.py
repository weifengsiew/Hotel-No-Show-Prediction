"""Extraction functions for the hotel no-show ETL pipeline."""

import sqlite3

import pandas as pd
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "HotelNoShowPrediction") -> SparkSession:
    """Create or retrieve a Spark session."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def extract_noshow_data(
    db_path: str = "data/raw/noshow.db",
    table_name: str = "noshow",
    app_name: str = "HotelNoShowPrediction",
):
    """Extract the no-show SQLite table into a Spark DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df_pandas = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    spark = get_spark_session(app_name=app_name)
    return spark.createDataFrame(df_pandas)
