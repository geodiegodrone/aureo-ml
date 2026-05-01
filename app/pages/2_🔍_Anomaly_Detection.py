"""Anomaly detection page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator
from aureo_ml.models.anomaly import fit_anomaly_detectors

st.set_page_config(page_title="AUREO-ML Anomalies", layout="wide")
df = fit_anomaly_detectors(AntioquiaGeochemicalGenerator().generate(1200))
st.title("Multivariate Anomaly Detection")
st.plotly_chart(
    px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="anomaly_consensus",
        size="Au",
        zoom=7,
        height=600,
        mapbox_style="carto-positron",
    ),
    use_container_width=True,
)
st.dataframe(df.sort_values("anomaly_consensus", ascending=False).head(25))
