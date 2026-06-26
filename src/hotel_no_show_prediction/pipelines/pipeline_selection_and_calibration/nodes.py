"""Pipeline selection and calibration nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def select_best_pipeline(grid_search) -> tuple[Any, float, dict]:
    """Select the best pipeline from GridSearchCV and return its key details."""
    return (
        grid_search.best_estimator_,
        float(grid_search.best_score_),
        grid_search.best_params_,
    )


def calibrate_pipeline(
    selected_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> CalibratedClassifierCV:
    """Calibrate the selected pipeline on the training data."""
    calibration_config = config["calibration"]
    calibration_cv = StratifiedKFold(
        n_splits=calibration_config["cv_folds"],
        shuffle=True,
        random_state=config["experiment"]["random_state"],
    )
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=selected_pipeline,
        method=config["calibration"]["method"],
        cv=calibration_cv,
    )
    calibrated_pipeline.fit(X_train, y_train)
    return calibrated_pipeline
