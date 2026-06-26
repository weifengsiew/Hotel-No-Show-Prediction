"""Data ingestion nodes."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


def load_raw_noshow(raw_data_config: dict[str, str]) -> pd.DataFrame:
    """Load the raw no-show SQLite table."""
    database_path = Path(raw_data_config["path"])
    table_name = raw_data_config["table_name"]

    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            connection,
        )
