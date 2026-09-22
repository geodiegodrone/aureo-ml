"""AUREO-ML Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aureo_ml.config import ELEMENTS
from aureo_ml.data.synthetic_generator import AntioquiaGeochemicalGenerator
from aureo_ml.features.geochemical_indices import add_geochemical_indices
from aureo_ml.models.classifier import train_classifier
from aureo_ml.visualization.geospatial_maps import prospectivity_map


st.set_page_config(page_title="AUREO-ML", layout="wide", page_icon="⛏️")


@st.cache_data(show_spinner=False)
def load_demo_data(n_samples: int = 2500) -> pd.DataFrame:
    """Load deterministic demo data."""

    return AntioquiaGeochemicalGenerator(random_seed=42).generate(n_samples)


@st.cache_resource(show_spinner=False)
def train_demo_model(df: pd.DataFrame):
    """Train and track the lightweight demo classifier."""

    return train_classifier(df, random_state=42, run_name="streamlit-demo-prospectivity")


df = load_demo_data()
model, model_metrics = train_demo_model(df)

st.title("AUREO-ML")
st.caption("Auriferous Exploration & Mineral Characterization with Machine Learning")

left, middle, right = st.columns(3)
left.metric("Samples", f"{len(df):,}")
middle.metric("High potential", f"{(df['potential_class'] == 'high').mean():.1%}")
right.metric("Median Au", f"{df['Au'].median():.3f} g/t")
st.caption(f"MLflow holdout ROC AUC: {model_metrics['roc_auc']:.3f}")

with st.sidebar:
    st.header("Filters")
    min_grade = st.slider("Minimum Au grade (g/t)", 0.0, float(df["Au"].quantile(0.99)), 0.0, 0.01)
    max_depth = st.slider("Maximum depth (m)", 1, 180, 180)
    classes = st.multiselect(
        "Potential class",
        ["low", "medium", "high"],
        default=["low", "medium", "high"],
    )

filtered = df[
    (df["Au"] >= min_grade) & (df["depth_m"] <= max_depth) & (df["potential_class"].isin(classes))
]

tab_map, tab_eda, tab_predict = st.tabs(["Map", "Geochemistry", "Live prediction"])

with tab_map:
    st.subheader("Prospectivity Map")
    st_folium(prospectivity_map(filtered.head(900)), height=560, use_container_width=True)

with tab_eda:
    st.subheader("Pathfinder Elements")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="As",
                y="Sb",
                color="potential_class",
                size="Au",
                hover_name="sample_id",
                log_x=True,
                log_y=True,
            ),
            use_container_width=True,
        )
    with c2:
        indices = add_geochemical_indices(filtered)
        st.dataframe(
            indices[
                ["sample_id", "Au", "As", "Sb", "Au_Ag_ratio", "epithermal_pathfinder_index"]
            ].head(20)
        )

with tab_predict:
    st.subheader("Prospectivity Prediction")
    cols = st.columns(5)
    values: dict[str, float | str] = {}
    defaults = df[ELEMENTS].median()
    for idx, element in enumerate(ELEMENTS):
        values[element] = cols[idx % 5].number_input(
            element, min_value=0.000001, value=float(defaults[element]), format="%.4f"
        )
    values["depth_m"] = st.slider("Depth (m)", 1.0, 180.0, 35.0)
    values["alteration_intensity"] = st.slider("Alteration intensity", 0.0, 1.0, 0.55)
    values["structural_score"] = st.slider("Structural score", 0.0, 1.0, 0.55)
    values["lithology"] = st.selectbox(
        "Lithology", ["metamorphic", "intrusive", "volcanosedimentary", "alluvial"]
    )
    sample = pd.DataFrame([values])
    probability = float(model.predict_proba(sample)[0, 1])
    st.metric("Prospective probability", f"{probability:.1%}")
    contribution = np.log1p(sample[ELEMENTS].iloc[0]).sort_values(ascending=False).head(8)
    st.plotly_chart(
        px.bar(contribution, orientation="h", title="Local geochemical contribution proxy"),
        use_container_width=True,
    )
