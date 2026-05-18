"""Run extraction, transformation, and loading for the hotel no-show dataset."""

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.extraction import extract_noshow_data
from src.transformation import transform_noshow_data
from src.loading import save_intermediate_data


DEFAULT_RAW_DB_PATH = PROJECT_ROOT / "data" / "raw" / "noshow.db"


def run_etl(
    db_path: str | Path = DEFAULT_RAW_DB_PATH,
    table_name: str = "noshow",
):
    """Extract, transform, and save the cleaned hotel no-show dataset."""
    df = extract_noshow_data(db_path=str(db_path), table_name=table_name)
    df = transform_noshow_data(df)

    intermediate_path = save_intermediate_data(df)

    return df, intermediate_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    transformed_df, intermediate_path = run_etl()

    transformed_df.show(5)
    print(f"Intermediate data saved to: {intermediate_path}")
