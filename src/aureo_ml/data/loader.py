"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_geochemistry_csv(path: str | Path) -> pd.DataFrame:
    """Load a geochemical CSV file."""

    return pd.read_csv(path)


def to_geodataframe(df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Convert tabular longitude/latitude samples to a GeoDataFrame."""

    required = {"longitude", "latitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing coordinate columns: {sorted(missing)}")
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )
