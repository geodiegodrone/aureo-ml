"""Plotly geochemistry plots."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def ternary_au_as_sb(df: pd.DataFrame) -> go.Figure:
    """Create Au-As-Sb ternary plot."""

    return px.scatter_ternary(
        df,
        a="Au",
        b="As",
        c="Sb",
        color="potential_class",
        hover_name="sample_id",
        title="Au-As-Sb Pathfinder Ternary Diagram",
    )


def element_boxplot(df: pd.DataFrame, element: str = "Au") -> go.Figure:
    """Create lithology-grouped element boxplot."""

    return px.box(
        df, x="lithology", y=element, color="potential_class", points="outliers", log_y=True
    )
