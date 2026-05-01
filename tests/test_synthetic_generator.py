"""Tests for synthetic generator."""

from __future__ import annotations

from aureo_ml.config import ANTIOQUIA_BOUNDS, ELEMENTS
from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator


def test_generator_shape_and_columns(sample_df):
    """Generated frame has expected size and scientific columns."""

    assert len(sample_df) == 300
    for element in ELEMENTS:
        assert element in sample_df.columns
        assert (sample_df[element] > 0).all()
    assert {"prospective", "potential_class", "prospectivity_score"}.issubset(sample_df.columns)


def test_coordinates_inside_antioquia(sample_df):
    """Coordinates remain inside Antioquia bounding box."""

    assert (
        sample_df["longitude"]
        .between(ANTIOQUIA_BOUNDS["min_lon"], ANTIOQUIA_BOUNDS["max_lon"])
        .all()
    )
    assert (
        sample_df["latitude"]
        .between(ANTIOQUIA_BOUNDS["min_lat"], ANTIOQUIA_BOUNDS["max_lat"])
        .all()
    )


def test_reproducibility():
    """Fixed seed produces deterministic samples."""

    a = AntioquiaGeochemicalGenerator(random_seed=9).generate(120)
    b = AntioquiaGeochemicalGenerator(random_seed=9).generate(120)
    assert a[["longitude", "latitude", "Au", "As"]].equals(b[["longitude", "latitude", "Au", "As"]])
