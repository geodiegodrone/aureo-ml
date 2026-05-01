"""SHAP explanations page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aureo_ml.config import ELEMENTS
from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator

st.set_page_config(page_title="AUREO-ML SHAP", layout="wide")
df = AntioquiaGeochemicalGenerator().generate(1500)
st.title("SHAP Explanations")
st.caption(
    "Global SHAP workflow is implemented in notebook 06. This dashboard view shows a deterministic pathfinder contribution proxy for fast interactive use."
)
contrib = df[ELEMENTS].apply(lambda s: s.corr(df["prospective"]))
st.plotly_chart(
    px.bar(contrib.sort_values(), orientation="h", title="Feature contribution proxy"),
    use_container_width=True,
)
