"""Shared MLflow logging for repeatable model training."""

from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from aureo_ml.config import ProjectConfig


def log_model_run(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    models: dict[str, Any] | None = None,
    input_example: pd.DataFrame | None = None,
) -> str:
    """Log reproducible parameters, metrics, and fitted sklearn models."""

    config = ProjectConfig()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(os.getenv("AUREO_MLFLOW_EXPERIMENT", "aureo-mineral-prospectivity"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for name, model in (models or {}).items():
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"models/{name}",
                input_example=input_example,
            )
        return run.info.run_id
