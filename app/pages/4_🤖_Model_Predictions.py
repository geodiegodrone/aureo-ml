"""Model predictions page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aureo_ml.config import ELEMENTS
from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator
from aureo_ml.models.classifier import build_classifier

st.set_page_config(page_title="AUREO-ML Predictions", layout="wide")
df = AntioquiaGeochemicalGenerator().generate(1800)
features = ELEMENTS + ["depth_m", "alteration_intensity", "structural_score", "lithology"]
model = build_classifier()
model.fit(df[features], df["prospective"])
st.title("Model Predictions")
row = df[features].sample(1, random_state=4).copy()
edited = st.data_editor(row, num_rows="fixed", use_container_width=True)
probability = float(model.predict_proba(pd.DataFrame(edited))[0, 1])
st.metric("Prospective probability", f"{probability:.1%}")
