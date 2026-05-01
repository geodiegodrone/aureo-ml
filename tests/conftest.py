"""Pytest fixtures."""

from __future__ import annotations

import pytest

from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator


@pytest.fixture()
def sample_df():
    """Small deterministic geochemical dataset."""

    return AntioquiaGeochemicalGenerator(random_seed=123).generate(300)
