"""Validator tests."""

from __future__ import annotations

import pytest

from aureo_ml.data.validator import validate_samples


def test_validate_samples(sample_df):
    """Valid samples pass Pydantic schema."""

    validated = validate_samples(sample_df, max_rows=5)
    assert len(validated) == 5


def test_validate_samples_rejects_missing_column(sample_df):
    """Missing required columns raise ValueError."""

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_samples(sample_df.drop(columns=["Au"]), max_rows=1)
