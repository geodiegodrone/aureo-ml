"""Mining-oriented model metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error, roc_auc_score


def classification_report_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute probability-aware classification metrics."""

    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def regression_report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for Au grade prediction."""

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": float(rmse), "mae": float(mae)}


def discovery_efficiency(
    df: pd.DataFrame,
    score_col: str = "prospectivity_score",
    target_col: str = "prospective",
    top_fraction: float = 0.1,
) -> float:
    """Measure share of known positives captured in top-ranked exploration area."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    top_n = max(1, int(len(df) * top_fraction))
    ranked = df.sort_values(score_col, ascending=False).head(top_n)
    total_positive = max(1, int(df[target_col].sum()))
    return float(ranked[target_col].sum() / total_positive)
