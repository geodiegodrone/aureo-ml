"""Multivariate anomaly detection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from aureo_ml.config import ELEMENTS


def fit_anomaly_detectors(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """Score samples with Isolation Forest, LOF, One-Class SVM, and Mahalanobis distance."""

    x = np.log1p(df[ELEMENTS].astype(float))
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    iso = IsolationForest(contamination=contamination, random_state=42)
    lof = LocalOutlierFactor(contamination=contamination, novelty=False)
    svm = Pipeline(
        [("scale", RobustScaler()), ("model", OneClassSVM(gamma="scale", nu=contamination))]
    )
    cov = EmpiricalCovariance().fit(x_scaled)
    inv_cov = np.linalg.pinv(cov.covariance_)
    center = cov.location_

    out = df.copy()
    out["iforest_anomaly_score"] = -iso.fit(x_scaled).score_samples(x_scaled)
    out["lof_anomaly_label"] = (lof.fit_predict(x_scaled) == -1).astype(int)
    out["ocsvm_anomaly_label"] = (svm.fit_predict(x) == -1).astype(int)
    out["mahalanobis_distance"] = [mahalanobis(row, center, inv_cov) for row in x_scaled]
    out["anomaly_consensus"] = (
        (
            out["iforest_anomaly_score"] >= out["iforest_anomaly_score"].quantile(1 - contamination)
        ).astype(int)
        + out["lof_anomaly_label"]
        + out["ocsvm_anomaly_label"]
        + (
            out["mahalanobis_distance"] >= out["mahalanobis_distance"].quantile(1 - contamination)
        ).astype(int)
    )
    return out
