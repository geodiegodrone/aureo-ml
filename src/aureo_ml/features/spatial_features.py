"""Spatial feature engineering for geochemical samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def add_spatial_features(df: pd.DataFrame, k_neighbors: int = 8) -> pd.DataFrame:
    """Add neighbor-density and coordinate interaction features."""

    out = df.copy()
    coords_rad = np.deg2rad(out[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    distances, _ = tree.query(coords_rad, k=min(k_neighbors + 1, len(out)))
    km = distances[:, 1:] * 6371.0
    out["mean_neighbor_distance_km"] = km.mean(axis=1)
    out["local_sample_density"] = 1.0 / (out["mean_neighbor_distance_km"] + 1e-6)
    out["lon_lat_interaction"] = out["longitude"] * out["latitude"]
    return out


def assign_spatial_blocks(df: pd.DataFrame, n_lon_bins: int = 5, n_lat_bins: int = 5) -> pd.Series:
    """Assign deterministic spatial blocks for spatial cross-validation."""

    lon_bin = pd.cut(df["longitude"], bins=n_lon_bins, labels=False, include_lowest=True)
    lat_bin = pd.cut(df["latitude"], bins=n_lat_bins, labels=False, include_lowest=True)
    return (lon_bin.astype(str) + "_" + lat_bin.astype(str)).rename("spatial_block")
