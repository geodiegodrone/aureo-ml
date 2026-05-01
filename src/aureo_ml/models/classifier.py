"""Supervised prospectivity classification."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from aureo_ml.data.preprocessor import build_preprocessor
from aureo_ml.features.geochemical_indices import add_geochemical_indices
from aureo_ml.features.spatial_features import add_spatial_features


def build_classifier(random_state: int = 42) -> Pipeline:
    """Build baseline prospectivity classifier."""

    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature engineering required by supervised models."""

    return add_spatial_features(add_geochemical_indices(df))
