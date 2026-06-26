from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

NUMERIC_FEATURES = [
    "arrival_day", "checkout_day", "num_adults", "num_children",
    "price_amount_sgd", "stay_length_nights", "booking_to_arrival_month_gap",
    "total_guests", "price_per_guest", "price_per_night",
    "price_per_guest_per_night",
]
CATEGORICAL_FEATURES = [
    "branch", "booking_month", "arrival_month", "checkout_month", "country",
    "first_time", "room", "platform", "price_currency",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def convert_numeric_features_to_float(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric feature columns to float64 dtype."""
    return feature_data.astype("float64")


def fill_missing_categorical_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in categorical feature columns with the string "missing"."""
    return feature_data.astype("object").where(pd.notna(feature_data), "missing")


def get_passthrough_features(transformed_features: list[str]) -> list[str]:
    """Return list of features that are not transformed but passed to the model."""
    transformed = set(transformed_features)
    return [feature for feature in ALL_FEATURES if feature not in transformed]


def build_preprocessor_v1(force_dense_output: bool) -> ColumnTransformer:
    """Build preprocessor_v1."""
    numeric_features = get_passthrough_features(CATEGORICAL_FEATURES)
    fill_missing_categories = FunctionTransformer(fill_missing_categorical_features,
                                                  feature_names_out="one-to-one",)
    convert_numeric_to_float = FunctionTransformer(convert_numeric_features_to_float,
                                                   feature_names_out="one-to-one",)
    categorical_pipeline = Pipeline(
        steps=[("fill_missing", fill_missing_categories),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),])
    numeric_pipeline = Pipeline(
        steps=[("to_float", convert_numeric_to_float)])
    
    preprocessor_v1= ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, numeric_features),],
        sparse_threshold=0.0 if force_dense_output else 0.3,)
    
    return preprocessor_v1


def build_preprocessor(preprocessing_config: dict, force_dense_output: bool) -> ColumnTransformer:
    """Build preprocessor named in preprocessing_config."""
    preprocessor_type = preprocessing_config["type"]
    if preprocessor_type == "preprocessor_v1":
        return build_preprocessor_v1(force_dense_output)

    raise ValueError(f"Unsupported preprocessing type: {preprocessor_type}")
