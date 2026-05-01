"""Interactive geospatial maps."""

from __future__ import annotations

import folium
import pandas as pd


def prospectivity_map(df: pd.DataFrame, score_col: str = "prospectivity_score") -> folium.Map:
    """Build a folium prospectivity map centered on Antioquia."""

    fmap = folium.Map(location=[6.85, -75.55], zoom_start=8, tiles="CartoDB positron")
    for _, row in df.iterrows():
        score = float(row[score_col])
        color = "red" if score >= 0.62 else "orange" if score >= 0.4 else "blue"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3 + 5 * score,
            color=color,
            fill=True,
            fill_opacity=0.65,
            popup=f"{row['sample_id']} | Au={row['Au']:.3f} g/t | score={score:.2f}",
        ).add_to(fmap)
    return fmap
