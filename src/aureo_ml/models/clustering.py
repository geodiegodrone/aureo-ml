"""Geochemical clustering models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from aureo_ml.config import ELEMENTS
from aureo_ml.tracking import log_model_run


def cluster_geochemical_populations(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster geochemical samples using K-Means, DBSCAN, and GMM."""

    x = df[ELEMENTS].astype(float)
    models = {
        "kmeans": Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", RobustScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=20)),
        ]),
        "gmm": Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", RobustScaler()),
            ("cluster", GaussianMixture(n_components=n_clusters, random_state=42)),
        ]),
        "dbscan": Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", RobustScaler()),
            ("cluster", DBSCAN(eps=1.75, min_samples=20)),
        ]),
    }
    out = df.copy()
    labels = {name: model.fit_predict(x) for name, model in models.items()}
    for name, values in labels.items():
        out[f"{name}_cluster"] = values
    metrics = {}
    for name, values in labels.items():
        valid = values != -1
        unique = np.unique(values[valid])
        if 1 < len(unique) < int(valid.sum()):
            metrics[f"{name}_silhouette"] = float(silhouette_score(x.loc[valid], values[valid]))
        metrics[f"{name}_clusters"] = float(len(unique))
    log_model_run(
        run_name="geochemical-population-clustering",
        params={"n_clusters": n_clusters, "dbscan_eps": 1.75, "dbscan_min_samples": 20},
        metrics=metrics,
        models={"kmeans": models["kmeans"], "gmm": models["gmm"]},
        input_example=x.head(3),
    )
    return out
