"""Supervised prospectivity classification."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from aureo_ml.data.preprocessor import build_preprocessor
from aureo_ml.features.geochemical_indices import add_geochemical_indices
from aureo_ml.features.spatial_features import add_spatial_features
from aureo_ml.tracking import log_model_run


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


def train_classifier(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.25,
    run_name: str = "prospectivity-random-forest",
) -> tuple[Pipeline, dict[str, float]]:
    """Fit, evaluate, and log a prospectivity classifier."""

    features = [
        "Au", "Ag", "As", "Sb", "Cu", "Pb", "Zn", "Fe", "Mn", "Hg", "Bi", "Te", "Mo", "W", "S",
        "depth_m", "alteration_intensity", "structural_score", "lithology",
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df["prospective"], test_size=test_size, random_state=random_state,
        stratify=df["prospective"],
    )
    model = build_classifier(random_state)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "f1_at_0_5": float(f1_score(y_test, probabilities >= 0.5)),
    }
    log_model_run(
        run_name=run_name,
        params={"model": "random_forest", "random_state": random_state, "test_size": test_size,
                "train_rows": len(x_train), "test_rows": len(x_test)},
        metrics=metrics,
        models={"classifier": model},
        input_example=x_train.head(3),
    )
    return model, metrics
