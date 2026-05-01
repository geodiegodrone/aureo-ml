"""Feature engineering tests."""

from __future__ import annotations

import numpy as np

from aureo_ml.features.geochemical_indices import add_geochemical_indices
from aureo_ml.features.spatial_features import add_spatial_features, assign_spatial_blocks
from aureo_ml.features.transformations import clr_transform, robust_log_transform


def test_clr_rows_sum_to_zero(sample_df):
    """CLR transform preserves compositional center."""

    clr = clr_transform(sample_df)
    assert np.allclose(clr.sum(axis=1).to_numpy(), 0.0, atol=1e-8)


def test_geochemical_indices_added(sample_df):
    """Domain indices are added and finite."""

    out = add_geochemical_indices(sample_df)
    assert "epithermal_pathfinder_index" in out.columns
    assert np.isfinite(out["epithermal_pathfinder_index"]).all()


def test_spatial_features_and_blocks(sample_df):
    """Spatial density features and blocks are produced."""

    out = add_spatial_features(sample_df, k_neighbors=4)
    blocks = assign_spatial_blocks(out)
    assert (out["mean_neighbor_distance_km"] > 0).all()
    assert blocks.nunique() > 1


def test_robust_log_transform(sample_df):
    """Log features are created."""

    out = robust_log_transform(sample_df, ["Au", "As"])
    assert {"log_Au", "log_As"}.issubset(out.columns)
