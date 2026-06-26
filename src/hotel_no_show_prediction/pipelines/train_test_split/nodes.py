"""Train/test split nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_and_target(
    modeling_data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Split modeling data into feature matrix and target vector."""
    target_column = config["target"]["column"]
    features = modeling_data.drop(columns=[target_column])
    target = modeling_data[target_column]
    return features, target


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
):
    """Create a stratified train/test split before model selection."""
    return train_test_split(
        features,
        target,
        test_size=config["data"]["test_size"],
        random_state=config["experiment"]["random_state"],
        stratify=target,
    )
