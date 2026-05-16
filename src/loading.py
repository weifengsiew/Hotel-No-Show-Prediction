"""Loading functions for the hotel no-show ETL pipeline."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INTERMEDIATE_PATH = (
    PROJECT_ROOT / "data" / "intermediate" / "noshow_cleaned.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT / "reports" / "validation" / "noshow_validation_results.json"
)


def validation_results_to_dict(validation_results):
    """Convert Great Expectations validation results into a JSON-serializable dict."""
    if hasattr(validation_results, "to_json_dict"):
        return validation_results.to_json_dict()

    if hasattr(validation_results, "dict"):
        return validation_results.dict()

    return json.loads(str(validation_results))


def save_intermediate_data(df, output_path: str | Path = DEFAULT_INTERMEDIATE_PATH):
    """Save the transformed Spark DataFrame as a single CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output_dir = output_path.with_suffix("")
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(temp_output_dir)
    )

    csv_part = next(temp_output_dir.glob("part-*.csv"))

    if output_path.exists():
        output_path.unlink()

    csv_part.rename(output_path)

    for leftover in temp_output_dir.iterdir():
        leftover.unlink()

    temp_output_dir.rmdir()

    return output_path


def save_validation_results(
    validation_results,
    output_path: str | Path = DEFAULT_VALIDATION_PATH,
):
    """Save Great Expectations validation results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            validation_results_to_dict(validation_results),
            file,
            indent=2,
            default=str,
        )

    return output_path
