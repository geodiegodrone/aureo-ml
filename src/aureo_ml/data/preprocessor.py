"""Preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from aureo_ml.config import ELEMENTS


def build_preprocessor(categorical: list[str] | None = None) -> ColumnTransformer:
    """Build a robust preprocessing transformer."""

    categorical = categorical or ["lithology"]
    numeric = ELEMENTS + ["depth_m", "alteration_intensity", "structural_score"]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("log1p", Log1pTransformer()), ("scale", RobustScaler())]),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible log1p transformer for positive geochemistry."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> Log1pTransformer:
        """Fit no-op transformer."""

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply log1p transform."""

        return np.log1p(np.asarray(X, dtype=float))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return transformed feature names."""

        return np.asarray(input_features if input_features is not None else [])
