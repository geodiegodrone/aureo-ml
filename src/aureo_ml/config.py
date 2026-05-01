"""Project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime configuration for AUREO-ML.

    Attributes:
        random_seed: Global random seed.
        data_dir: Base data directory.
        mlflow_tracking_uri: Local or remote MLflow tracking URI.
    """

    random_seed: int = int(os.getenv("AUREO_RANDOM_SEED", "42"))
    data_dir: Path = Path(os.getenv("AUREO_DATA_DIR", "data"))
    mlflow_tracking_uri: str = os.getenv("AUREO_MLFLOW_TRACKING_URI", "mlruns")


ANTIOQUIA_BOUNDS = {
    "min_lon": -77.15,
    "max_lon": -73.85,
    "min_lat": 5.35,
    "max_lat": 8.95,
}

ELEMENTS = [
    "Au",
    "Ag",
    "As",
    "Sb",
    "Cu",
    "Pb",
    "Zn",
    "Fe",
    "Mn",
    "Hg",
    "Bi",
    "Te",
    "Mo",
    "W",
    "S",
]
