"""Prospectivity map page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator
from aureo_ml.visualization.geospatial_maps import prospectivity_map

st.set_page_config(page_title="AUREO-ML Map", layout="wide")
df = AntioquiaGeochemicalGenerator().generate(2500)
st.title("Prospectivity Map")
st_folium(prospectivity_map(df.head(1200)), height=650, use_container_width=True)
