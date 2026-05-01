"""Domain-specific geochemical indices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_geochemical_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add pathfinder and alteration indices used in gold exploration."""

    out = df.copy()
    eps = 1e-9
    out["Au_Ag_ratio"] = out["Au"] / (out["Ag"] + eps)
    out["As_Sb_ratio"] = out["As"] / (out["Sb"] + eps)
    out["Cu_Pb_Zn_index"] = np.log1p(out["Cu"] + out["Pb"] + out["Zn"])
    out["epithermal_pathfinder_index"] = np.log1p(out["As"] * out["Sb"] * out["Hg"] * out["Te"])
    out["intrusion_related_index"] = np.log1p(out["Bi"] * out["Te"] * out["W"] * out["Mo"])
    out["sulfide_proxy"] = np.log1p(out["S"] * (out["Fe"] + out["Cu"] + out["Pb"] + out["Zn"]))
    return out


def summarize_pathfinders(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for gold pathfinder elements."""

    cols = ["Au", "Ag", "As", "Sb", "Cu", "Pb", "Zn", "Hg", "Bi", "Te"]
    return df[cols].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T
