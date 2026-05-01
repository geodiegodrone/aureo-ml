"""Tests for loaders, metrics, utilities, and visualization builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aureo_ml.data.loader import load_geochemistry_csv, to_geodataframe
from aureo_ml.evaluation.metrics import classification_report_metrics, regression_report_metrics
from aureo_ml.features.geochemical_indices import summarize_pathfinders
from aureo_ml.features.transformations import boxcox_transform
from aureo_ml.utils.helpers import set_global_seed
from aureo_ml.utils.logger import get_logger
from aureo_ml.visualization.geochemistry_plots import element_boxplot, ternary_au_as_sb
from aureo_ml.visualization.geospatial_maps import prospectivity_map
from aureo_ml.visualization.ml_diagnostics import precision_recall_figure, roc_figure


def test_loader_and_geodataframe(sample_df, tmp_path):
    """CSV loader and GeoDataFrame conversion work."""

    path = tmp_path / "samples.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_geochemistry_csv(path)
    gdf = to_geodataframe(loaded)
    assert len(loaded) == len(sample_df)
    assert gdf.crs.to_string() == "EPSG:4326"


def test_metrics_functions(sample_df):
    """Classification, regression, and error branches are covered."""

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert classification_report_metrics(y_true, y_score)["auc"] == 1.0
    assert regression_report_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.5]))["rmse"] > 0
    assert summarize_pathfinders(sample_df).shape[0] == 10


def test_transform_utils_and_logger(sample_df):
    """Box-Cox, seed setter, and logger helpers work."""

    transformed, lambda_value = boxcox_transform(sample_df["As"])
    set_global_seed(7)
    logger = get_logger("aureo_ml_test")
    assert len(transformed) == len(sample_df)
    assert isinstance(lambda_value, float)
    assert logger.name == "aureo_ml_test"


def test_visualization_builders(sample_df):
    """Visualization functions return figure/map objects."""

    small = sample_df.head(50).copy()
    assert ternary_au_as_sb(small).data
    assert element_boxplot(small, "Au").data
    assert prospectivity_map(small)._children
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.25, 0.7, 0.95])
    assert roc_figure(y_true.to_numpy(), y_score.to_numpy()).data
    assert precision_recall_figure(y_true.to_numpy(), y_score.to_numpy()).data
