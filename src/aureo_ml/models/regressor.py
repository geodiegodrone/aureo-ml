"""Gold grade regression models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aureo_ml.data.preprocessor import build_preprocessor
from aureo_ml.tracking import log_model_run


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


def train_regressor(
    df,
    features: list[str],
    target: str = "Au",
    random_state: int = 42,
    test_size: float = 0.25,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit, evaluate, and log an Au-grade regressor."""

    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=test_size, random_state=random_state
    )
    model = build_regressor(random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    log_model_run(
        run_name="gold-grade-random-forest",
        params={"model": "random_forest_regressor", "random_state": random_state,
                "test_size": test_size, "target": target, "train_rows": len(x_train),
                "test_rows": len(x_test)},
        metrics=metrics,
        models={"regressor": model},
        input_example=x_train.head(3),
    )
    return model, metrics
