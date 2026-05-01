"""Geochemical clustering models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from aureo_ml.config import ELEMENTS


def cluster_geochemical_populations(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster geochemical samples using K-Means, DBSCAN, and GMM."""

    x = RobustScaler().fit_transform(np.log1p(df[ELEMENTS].astype(float)))
    out = df.copy()
    out["kmeans_cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x)
    out["gmm_cluster"] = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(x)
    out["dbscan_cluster"] = DBSCAN(eps=1.75, min_samples=20).fit_predict(x)
    return out
