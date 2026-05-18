"""Clean the raw hotel no-show dataset."""

import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.run_etl import run_etl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    transformed_df, intermediate_path = run_etl()
    transformed_df.show(5)
    print(f"Cleaned data saved to: {intermediate_path}")
