from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}
DENSE_INPUT_MODELS = {"RandomForestClassifier"}


def build_model(model_config: dict) -> ClassifierMixin:
    """Return model with hyperparameters given a model config."""
    model_type = model_config["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Expected one of: {sorted(MODEL_REGISTRY)}"
        )

    model_class = MODEL_REGISTRY[model_type]
    return model_class(**model_config.get("params", {}))


def model_requires_dense_input(model_type: str) -> bool:
    """Return True if the model expects dense input arrays."""
    return model_type in DENSE_INPUT_MODELS
