"""Compositional and distributional transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from aureo_ml.config import ELEMENTS


def multiplicative_replacement(values: pd.DataFrame, delta: float = 1e-6) -> pd.DataFrame:
    """Replace zero or negative values before compositional log-ratio transforms."""

    clean = values.copy().astype(float)
    clean[clean <= 0] = delta
    return clean


def clr_transform(values: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Apply centered log-ratio transform.

    Args:
        values: DataFrame with strictly positive compositional parts.
        columns: Columns to transform. Defaults to AUREO element list.

    Returns:
        CLR-transformed DataFrame with ``clr_`` prefixes.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"Au": [1, 2], "Ag": [2, 4], "As": [4, 8]})
        >>> out = clr_transform(data, ["Au", "Ag", "As"])
        >>> round(float(out.iloc[0].sum()), 8)
        0.0
    """

    columns = columns or ELEMENTS
    x = multiplicative_replacement(values[columns])
    logs = np.log(x)
    centered = logs.sub(logs.mean(axis=1), axis=0)
    return centered.add_prefix("clr_")


def boxcox_transform(series: pd.Series) -> tuple[pd.Series, float]:
    """Apply Box-Cox transform to a positive series."""

    clean = series.astype(float).clip(lower=1e-9)
    transformed, lambda_value = stats.boxcox(clean)
    return pd.Series(transformed, index=series.index, name=f"boxcox_{series.name}"), float(
        lambda_value
    )


def robust_log_transform(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add log1p features for skewed geochemical concentrations."""

    columns = columns or ELEMENTS
    out = df.copy()
    for column in columns:
        out[f"log_{column}"] = np.log1p(out[column].astype(float))
    return out
