"""Build scikit-learn objects from project configuration."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier


def split_features_and_target(
    df: pd.DataFrame,
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the configured target column."""
    target_column = data_config["target_column"]
    columns_to_drop = [*data_config["drop_columns"], target_column]
    features = df.drop(columns=columns_to_drop)
    target = df[target_column]

    return features, target


def create_preprocessor(
    processing_config: dict[str, Any],
) -> ColumnTransformer:
    """Create the configured categorical and numeric preprocessing steps."""
    categorical_pipeline = create_categorical_pipeline(processing_config)
    numeric_imputer = SimpleImputer(strategy=processing_config["numeric_imputation"])
    preprocessor = make_column_transformer(
        (
            categorical_pipeline,
            make_column_selector(dtype_include=["object", "bool"]),
        ),
        (
            numeric_imputer,
            make_column_selector(dtype_exclude=["object", "bool"]),
        ),
    )

    return preprocessor


def create_categorical_pipeline(processing_config: dict[str, Any]) -> Pipeline:
    """Create the configured categorical preprocessing pipeline."""
    if processing_config["categorical_encoding"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif processing_config["categorical_encoding"] == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(
            f"Unsupported categorical encoding: {processing_config['categorical_encoding']}"
        )

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy=processing_config["categorical_imputation"]),
        encoder,
    )

    return categorical_pipeline


def create_model(model_config: dict[str, Any]):
    """Create the configured model object."""
    model_classes = {
        "xgboost": XGBClassifier,
        "random_forest": RandomForestClassifier,
    }
    model = model_classes[model_config["name"]](**model_config["parameters"])

    return model


def create_pipeline(preprocessor: Any, model: Any) -> Pipeline:
    """Combine preprocessing and model steps into one scikit-learn pipeline."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
