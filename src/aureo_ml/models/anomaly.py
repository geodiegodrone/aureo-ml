"""Multivariate anomaly detection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import OneClassSVM

from aureo_ml.config import ELEMENTS
from aureo_ml.tracking import log_model_run


def fit_anomaly_detectors(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """Score samples with Isolation Forest, LOF, One-Class SVM, and Mahalanobis distance."""

    x = df[ELEMENTS].astype(float)
    def pipeline(estimator):
        return Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", RobustScaler()),
            ("model", estimator),
        ])

    iso = pipeline(IsolationForest(contamination=contamination, random_state=42))
    lof = pipeline(LocalOutlierFactor(contamination=contamination, novelty=False))
    svm = pipeline(OneClassSVM(gamma="scale", nu=contamination))
    covariance = pipeline(EmpiricalCovariance()).fit(x)
    x_scaled = covariance.named_steps["scale"].transform(
        covariance.named_steps["log1p"].transform(x)
    )
    cov = covariance.named_steps["model"]
    inv_cov = np.linalg.pinv(cov.covariance_)
    center = cov.location_

    out = df.copy()
    out["iforest_anomaly_score"] = -iso.fit(x).score_samples(x)
    out["lof_anomaly_label"] = (lof.fit_predict(x) == -1).astype(int)
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
    log_model_run(
        run_name="geochemical-anomaly-detection",
        params={"contamination": contamination, "random_state": 42},
        metrics={
            "consensus_outlier_rate": float((out["anomaly_consensus"] >= 2).mean()),
            "iforest_outlier_rate": float((out["iforest_anomaly_score"] >= out["iforest_anomaly_score"].quantile(1 - contamination)).mean()),
        },
        models={"isolation_forest": iso, "one_class_svm": svm},
        input_example=x.head(3),
    )
    return out
