"""Loading functions for the hotel no-show ETL pipeline."""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERMEDIATE_PATH = PROJECT_ROOT / "data" / "intermediate" / "noshow_cleaned.csv"


# -----------------------------------------------------------------------------

def save_intermediate_data(df, output_path: str | Path = DEFAULT_INTERMEDIATE_PATH):
    """Save the transformed Spark DataFrame as a single CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output_dir = output_path.with_suffix("")
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_output_dir))

    csv_parts = list(temp_output_dir.glob("part-*.csv"))
    if not csv_parts:
        raise FileNotFoundError(f"No Spark CSV part file found in {temp_output_dir}")

    if output_path.exists():
        output_path.unlink()

    csv_parts[0].rename(output_path)

    shutil.rmtree(temp_output_dir)

    return output_path
