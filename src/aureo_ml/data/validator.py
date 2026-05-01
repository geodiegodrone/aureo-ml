"""Pydantic validation for geochemical samples."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from aureo_ml.config import ANTIOQUIA_BOUNDS, ELEMENTS


class GeochemicalSample(BaseModel):
    """Validated geochemical sample schema."""

    sample_id: str
    longitude: float = Field(ge=ANTIOQUIA_BOUNDS["min_lon"], le=ANTIOQUIA_BOUNDS["max_lon"])
    latitude: float = Field(ge=ANTIOQUIA_BOUNDS["min_lat"], le=ANTIOQUIA_BOUNDS["max_lat"])
    department: str
    lithology: str
    depth_m: float = Field(ge=0.0, le=500.0)
    prospective: int = Field(ge=0, le=1)
    potential_class: Literal["low", "medium", "high"]
    Au: float = Field(gt=0.0)
    Ag: float = Field(gt=0.0)
    As: float = Field(gt=0.0)
    Sb: float = Field(gt=0.0)
    Cu: float = Field(gt=0.0)
    Pb: float = Field(gt=0.0)
    Zn: float = Field(gt=0.0)
    Fe: float = Field(gt=0.0)
    Mn: float = Field(gt=0.0)
    Hg: float = Field(gt=0.0)
    Bi: float = Field(gt=0.0)
    Te: float = Field(gt=0.0)
    Mo: float = Field(gt=0.0)
    W: float = Field(gt=0.0)
    S: float = Field(gt=0.0)


def validate_samples(df: pd.DataFrame, max_rows: int | None = None) -> list[GeochemicalSample]:
    """Validate a DataFrame of samples.

    Args:
        df: Input DataFrame.
        max_rows: Optional row limit for fast validation.

    Returns:
        Validated Pydantic model instances.

    Raises:
        ValueError: If required columns are absent or rows fail validation.
    """

    required = set(GeochemicalSample.model_fields).union(ELEMENTS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    records = (
        df.head(max_rows).to_dict(orient="records") if max_rows else df.to_dict(orient="records")
    )
    try:
        return [GeochemicalSample.model_validate(record) for record in records]
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
