"""Inspect failed Great Expectations checks from noshow_validation_results.json."""

import json
from pathlib import Path


VALIDATION_RESULTS_PATH = (
    Path("reports") / "validation" / "noshow_validation_results.json"
)


def load_validation_results(path: Path = VALIDATION_RESULTS_PATH) -> dict:
    """Load Great Expectations validation results from JSON."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_expectation_type(result: dict) -> str:
    """Return the expectation type from a GX result object."""
    expectation_config = result.get("expectation_config", {})
    return (
        expectation_config.get("type")
        or expectation_config.get("expectation_type")
        or "UnknownExpectation"
    )


def get_expectation_kwargs(result: dict) -> dict:
    """Return expectation keyword arguments from a GX result object."""
    expectation_config = result.get("expectation_config", {})
    return expectation_config.get("kwargs", {})


def print_failed_expectations(validation_results: dict) -> None:
    """Print failed expectations with columns and example failed values."""
    failed_results = [
        result
        for result in validation_results.get("results", [])
        if result.get("success") is False
    ]

    if not failed_results:
        print("No failed expectations.")
        return

    print(f"Failed expectations: {len(failed_results)}")
    print()

    for result in failed_results:
        expectation_type = get_expectation_type(result)
        kwargs = get_expectation_kwargs(result)
        column = kwargs.get("column", "<table-level>")

        result_details = result.get("result", {})
        unexpected_count = result_details.get("unexpected_count")
        unexpected_percent = result_details.get("unexpected_percent")
        partial_unexpected_list = result_details.get("partial_unexpected_list", [])
        partial_unexpected_counts = result_details.get("partial_unexpected_counts", [])

        print(f"Column: {column}")
        print(f"Expectation: {expectation_type}")

        if unexpected_count is not None:
            print(f"Unexpected count: {unexpected_count}")

        if unexpected_percent is not None:
            print(f"Unexpected percent: {unexpected_percent:.2f}%")

        if partial_unexpected_counts:
            print("Example failed values:")
            for item in partial_unexpected_counts[:5]:
                value = item.get("value")
                count = item.get("count")
                print(f"  - {value!r}: {count}")
        elif partial_unexpected_list:
            print("Example failed values:")
            for value in partial_unexpected_list[:5]:
                print(f"  - {value!r}")
        else:
            print("Example failed values: not available in saved JSON")

        print("-" * 60)


def main() -> None:
    validation_results = load_validation_results()
    print_failed_expectations(validation_results)


if __name__ == "__main__":
    main()
