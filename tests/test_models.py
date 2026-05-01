"""Model tests."""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from aureo_ml.config import ELEMENTS
from aureo_ml.evaluation.cross_validation import spatial_group_splits
from aureo_ml.evaluation.metrics import discovery_efficiency
from aureo_ml.models.anomaly import fit_anomaly_detectors
from aureo_ml.models.classifier import build_classifier
from aureo_ml.models.clustering import cluster_geochemical_populations
from aureo_ml.models.ensemble import build_stacking_classifier
from aureo_ml.models.regressor import build_regressor


def test_anomaly_detection_outputs(sample_df):
    """Anomaly methods append consensus scores."""

    out = fit_anomaly_detectors(sample_df, contamination=0.1)
    assert {"iforest_anomaly_score", "mahalanobis_distance", "anomaly_consensus"}.issubset(
        out.columns
    )
    assert out["anomaly_consensus"].between(0, 4).all()


def test_clustering_outputs(sample_df):
    """Clustering methods append labels."""

    out = cluster_geochemical_populations(sample_df, n_clusters=3)
    assert {"kmeans_cluster", "gmm_cluster", "dbscan_cluster"}.issubset(out.columns)


def test_classifier_learns_signal(sample_df):
    """Baseline classifier captures synthetic pathfinder signal."""

    features = ELEMENTS + ["depth_m", "alteration_intensity", "structural_score", "lithology"]
    model = build_classifier(random_state=123)
    model.fit(sample_df[features], sample_df["prospective"])
    proba = model.predict_proba(sample_df[features])[:, 1]
    assert roc_auc_score(sample_df["prospective"], proba) > 0.95


def test_spatial_cv_and_discovery_efficiency(sample_df):
    """Spatial CV and mining efficiency metrics are usable."""

    splits = spatial_group_splits(sample_df, n_splits=3)
    assert len(splits) == 3
    assert discovery_efficiency(sample_df, top_fraction=0.2) > 0


def test_regressor_and_stacking_builders(sample_df):
    """Regressor and stacking factory functions return usable pipelines."""

    features = ELEMENTS + ["depth_m", "alteration_intensity", "structural_score", "lithology"]
    regressor = build_regressor(random_state=123)
    regressor.fit(sample_df[features], sample_df["Au"])
    predictions = regressor.predict(sample_df[features].head(5))
    assert len(predictions) == 5
    stack = build_stacking_classifier(random_state=123)
    assert "model" in stack.named_steps
