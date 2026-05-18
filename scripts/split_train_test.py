"""Split cleaned data into train and test files."""

from pathlib import Path
import sys

import click
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path


DEFAULT_SHARED_CONFIG_PATH = PROJECT_ROOT / "configs" / "shared.yaml"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "noshow_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "noshow_test.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_config_and_data(config_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load the shared config and cleaned dataset used for splitting."""
    # Load the shared pipeline settings.
    config = load_config(config_path)
    data_config = config["data"]

    # Read the cleaned dataset configured as the split input.
    cleaned_path = resolve_project_path(data_config["cleaned_path"])
    df = pd.read_csv(cleaned_path)

    return config, df


# ---------------------------------------------------------------------------
# Train-test split
# ---------------------------------------------------------------------------


def split_data(config: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train-test split using the configured settings."""
    # Read split settings from config.
    data_config = config["data"]
    training_config = config["training"]

    # Split while preserving the target-class distribution.
    train_df, test_df = train_test_split(
        df,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=df[data_config["target_column"]],
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_split_files(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save the train and test datasets to the processed data folder."""
    # Resolve output paths and ensure the output folder exists.
    train_path = resolve_project_path(DEFAULT_TRAIN_PATH)
    test_path = resolve_project_path(DEFAULT_TEST_PATH)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    # Write train and test splits.
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


@click.command()
@click.option(
    "--config",
    default=str(DEFAULT_SHARED_CONFIG_PATH),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the shared YAML config.",
)
def main(config: Path) -> None:
    """Run the train-test split workflow from cleaned data to saved CSV files."""
    # Load inputs.
    config, df = load_config_and_data(config)

    # Create split.
    train_df, test_df = split_data(config, df)

    # Save outputs.
    train_path, test_path = save_split_files(train_df, test_df)
    print(f"Training split saved to: {train_path}")
    print(f"Test split saved to: {test_path}")


if __name__ == "__main__":
    main()
