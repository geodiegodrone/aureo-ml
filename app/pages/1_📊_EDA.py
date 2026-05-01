"""EDA page."""

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
from aureo_ml.visualization.geochemistry_plots import element_boxplot, ternary_au_as_sb

st.set_page_config(page_title="AUREO-ML EDA", layout="wide")
df = AntioquiaGeochemicalGenerator().generate(1800)
st.title("Geochemical EDA")
element = st.selectbox("Element", ["Au", "Ag", "As", "Sb", "Cu", "Pb", "Zn", "Hg", "Bi", "Te"])
st.plotly_chart(element_boxplot(df, element), use_container_width=True)
st.plotly_chart(ternary_au_as_sb(df.sample(900, random_state=42)), use_container_width=True)
st.plotly_chart(
    px.imshow(
        df[["Au", "Ag", "As", "Sb", "Cu", "Pb", "Zn", "Hg", "Bi", "Te"]].corr(), text_auto=".2f"
    ),
    use_container_width=True,
)
