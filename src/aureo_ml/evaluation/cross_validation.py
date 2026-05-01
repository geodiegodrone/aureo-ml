"""Spatial cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from aureo_ml.features.spatial_features import assign_spatial_blocks


def spatial_group_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold splits using spatial blocks."""

    groups = assign_spatial_blocks(df)
    splitter = GroupKFold(n_splits=n_splits)
    x_dummy = np.zeros((len(df), 1))
    return list(splitter.split(x_dummy, df.get("prospective", np.zeros(len(df))), groups))
