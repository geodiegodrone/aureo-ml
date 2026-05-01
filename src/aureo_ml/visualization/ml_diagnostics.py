"""Machine-learning diagnostic plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve


def roc_figure(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    """Create ROC curve figure."""

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"})
    )
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def precision_recall_figure(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    """Create precision-recall figure."""

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return fig
