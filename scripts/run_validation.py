"""Run validation for the cleaned hotel no-show dataset."""

import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.extraction import get_spark_session
from src.loading import DEFAULT_INTERMEDIATE_PATH
from src.transformation import convert_column_types_to_expected
from src.validation import validate_noshow_data


def load_cleaned_data(input_path: str | Path = DEFAULT_INTERMEDIATE_PATH):
    """Load the cleaned CSV into a Spark DataFrame for validation."""
    spark = get_spark_session(app_name="HotelNoShowValidation")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(input_path))
    )
    df = convert_column_types_to_expected(df)

    return df


def run_validation(input_path: str | Path = DEFAULT_INTERMEDIATE_PATH):
    """Validate the cleaned no-show dataset."""
    df = load_cleaned_data(input_path=input_path)
    df, validation_results = validate_noshow_data(df)

    return df, validation_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    validated_df, results = run_validation()

    validated_df.show(5)
    print(f"Validation success: {results.success}")
