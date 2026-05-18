"""Validate the cleaned hotel no-show dataset."""

import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.run_validation import run_validation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validated_df, results = run_validation()
    validated_df.show(5)
    print(f"Validation success: {results.success}")
