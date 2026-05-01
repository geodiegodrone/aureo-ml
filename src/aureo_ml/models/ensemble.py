"""Stacking ensemble training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aureo_ml.config import ELEMENTS, ProjectConfig
from aureo_ml.data.loader import load_geochemistry_csv
from aureo_ml.data.preprocessor import build_preprocessor


def build_stacking_classifier(random_state: int = 42) -> Pipeline:
    """Build a tree-based stacking classifier with reproducible defaults."""

    estimators: list[tuple[str, Any]] = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=220, random_state=random_state, class_weight="balanced"
            ),
        ),
        ("gb", GradientBoostingClassifier(random_state=random_state)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", stack)])


def train_stacking(input_path: str | Path, output_path: str | Path) -> dict[str, float]:
    """Train and persist stacking classifier."""

    cfg = ProjectConfig()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    df = load_geochemistry_csv(input_path)
    features = ELEMENTS + ["depth_m", "alteration_intensity", "structural_score", "lithology"]
    target = "prospective"
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.25, random_state=cfg.random_seed, stratify=df[target]
    )
    model = build_stacking_classifier(cfg.random_seed)
    with mlflow.start_run(run_name="stacking_prospectivity"):
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        mlflow.log_metric("auc", auc)
        mlflow.log_param("model", "stacking_rf_gradient_boosting")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        mlflow.log_artifact(str(out))
    return {"auc": auc}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(train_stacking(args.input, args.output))


if __name__ == "__main__":
    main()
