"""Gold grade regression models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from aureo_ml.data.preprocessor import build_preprocessor


def build_regressor(random_state: int = 42) -> Pipeline:
    """Build baseline Au grade regressor."""

    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
