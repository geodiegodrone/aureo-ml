"""Synthetic geochemical data generator for Antioquia, Colombia."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aureo_ml.config import ANTIOQUIA_BOUNDS, ELEMENTS, ProjectConfig


class AntioquiaGeochemicalGenerator:
    """Generate realistic multi-element gold exploration samples.

    The generator combines background lithogeochemical variation, district-scale
    hydrothermal centers, structural corridor effects, and log-normal analytical
    noise. Concentrations are reported in ppm except Au, which is converted to g/t.

    Example:
        >>> gen = AntioquiaGeochemicalGenerator(random_seed=7)
        >>> df = gen.generate(50)
        >>> set(["Au", "As", "prospective"]).issubset(df.columns)
        True
    """

    def __init__(self, random_seed: int = ProjectConfig.random_seed) -> None:
        """Initialize generator.

        Args:
            random_seed: Seed for deterministic sampling.
        """

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.centers = np.array(
            [
                [-75.56, 6.92],  # Segovia-Remedios
                [-75.21, 6.54],  # Amalfi-Anori trend
                [-75.83, 6.49],  # Buritica belt
                [-74.92, 7.12],  # Nordeste Antioqueno
            ]
        )
        self.center_weights = np.array([0.36, 0.25, 0.22, 0.17])

    def generate(self, n_samples: int = 10_000) -> pd.DataFrame:
        """Generate synthetic geochemical samples.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            DataFrame with coordinates, elemental concentrations, targets, and metadata.
        """

        if n_samples < 100:
            raise ValueError("n_samples must be at least 100 for stable mixture structure.")

        coordinates = self._sample_coordinates(n_samples)
        longitude = coordinates[:, 0]
        latitude = coordinates[:, 1]
        alteration = self._alteration_intensity(coordinates)
        structural = self._structural_score(longitude, latitude)
        lithology = self.rng.choice(
            ["metamorphic", "intrusive", "volcanosedimentary", "alluvial"],
            size=n_samples,
            p=[0.34, 0.28, 0.25, 0.13],
        )
        depth_m = self.rng.gamma(shape=2.2, scale=18.0, size=n_samples).clip(0.5, 180.0)

        df = pd.DataFrame(
            {
                "sample_id": [f"ANT-{i:05d}" for i in range(n_samples)],
                "longitude": longitude,
                "latitude": latitude,
                "department": "Antioquia",
                "lithology": lithology,
                "depth_m": depth_m,
                "alteration_intensity": alteration,
                "structural_score": structural,
            }
        )
        element_frame = self._simulate_elements(alteration, structural, lithology, depth_m)
        df = pd.concat([df, element_frame], axis=1)
        df["prospectivity_score"] = self._prospectivity_score(df)
        df["prospective"] = (df["prospectivity_score"] >= 0.62).astype(int)
        df["potential_class"] = pd.cut(
            df["prospectivity_score"],
            bins=[-np.inf, 0.4, 0.62, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)
        return df

    def _sample_coordinates(self, n_samples: int) -> np.ndarray:
        background_n = int(n_samples * 0.55)
        clustered_n = n_samples - background_n
        bg_lon = self.rng.uniform(
            ANTIOQUIA_BOUNDS["min_lon"], ANTIOQUIA_BOUNDS["max_lon"], background_n
        )
        bg_lat = self.rng.uniform(
            ANTIOQUIA_BOUNDS["min_lat"], ANTIOQUIA_BOUNDS["max_lat"], background_n
        )
        center_idx = self.rng.choice(len(self.centers), size=clustered_n, p=self.center_weights)
        centers = self.centers[center_idx]
        clustered = centers + self.rng.normal(0.0, [0.22, 0.18], size=(clustered_n, 2))
        coords = np.vstack([np.column_stack([bg_lon, bg_lat]), clustered])
        coords[:, 0] = coords[:, 0].clip(ANTIOQUIA_BOUNDS["min_lon"], ANTIOQUIA_BOUNDS["max_lon"])
        coords[:, 1] = coords[:, 1].clip(ANTIOQUIA_BOUNDS["min_lat"], ANTIOQUIA_BOUNDS["max_lat"])
        self.rng.shuffle(coords)
        return coords

    def _alteration_intensity(self, coordinates: np.ndarray) -> np.ndarray:
        distances = np.stack(
            [np.hypot(coordinates[:, 0] - c[0], coordinates[:, 1] - c[1]) for c in self.centers],
            axis=1,
        )
        gaussian = np.exp(-((distances / 0.34) ** 2))
        intensity = gaussian.max(axis=1) + self.rng.normal(0.0, 0.06, len(coordinates))
        return intensity.clip(0.0, 1.0)

    @staticmethod
    def _structural_score(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        corridor_1 = np.exp(-((latitude - (0.62 * longitude + 53.25)) ** 2) / 0.065)
        corridor_2 = np.exp(-((latitude - (-0.48 * longitude - 29.05)) ** 2) / 0.09)
        return np.maximum(corridor_1, corridor_2).clip(0.0, 1.0)

    def _simulate_elements(
        self,
        alteration: np.ndarray,
        structural: np.ndarray,
        lithology: np.ndarray,
        depth_m: np.ndarray,
    ) -> pd.DataFrame:
        n_samples = len(alteration)
        lithology_factor = (
            pd.Series(lithology)
            .map(
                {
                    "metamorphic": 0.12,
                    "intrusive": 0.20,
                    "volcanosedimentary": 0.08,
                    "alluvial": -0.08,
                }
            )
            .to_numpy()
        )
        mineralization = (
            1.35 * alteration
            + 0.75 * structural
            + lithology_factor
            - 0.0015 * depth_m
            + self.rng.normal(0.0, 0.18, n_samples)
        )
        mineralization = np.clip(mineralization, -0.5, 2.5)

        params = {
            "Au": (-3.85, 1.05, 2.6),
            "Ag": (0.10, 0.72, 1.1),
            "As": (2.55, 0.86, 1.6),
            "Sb": (0.95, 0.78, 1.35),
            "Cu": (3.35, 0.62, 0.50),
            "Pb": (2.80, 0.58, 0.45),
            "Zn": (4.10, 0.54, 0.35),
            "Fe": (10.0, 0.20, 0.10),
            "Mn": (6.10, 0.42, -0.10),
            "Hg": (-1.20, 0.90, 1.45),
            "Bi": (-0.25, 0.80, 1.10),
            "Te": (-1.55, 0.83, 1.45),
            "Mo": (0.90, 0.62, 0.45),
            "W": (0.20, 0.70, 0.70),
            "S": (7.25, 0.35, 0.55),
        }
        data: dict[str, np.ndarray] = {}
        for element in ELEMENTS:
            mean, sigma, loading = params[element]
            values = self.rng.lognormal(mean + loading * mineralization, sigma, n_samples)
            data[element] = values
        data["Au"] = (data["Au"] / 1000.0).clip(0.0001, None)
        return pd.DataFrame(data)

    @staticmethod
    def _prospectivity_score(df: pd.DataFrame) -> pd.Series:
        log = np.log1p
        raw = (
            0.33 * (log(df["Au"]) / log(df["Au"]).quantile(0.99))
            + 0.18 * (log(df["As"]) / log(df["As"]).quantile(0.99))
            + 0.14 * (log(df["Sb"]) / log(df["Sb"]).quantile(0.99))
            + 0.10 * (log(df["Ag"]) / log(df["Ag"]).quantile(0.99))
            + 0.08 * (log(df["Hg"]) / log(df["Hg"]).quantile(0.99))
            + 0.08 * (log(df["Te"]) / log(df["Te"]).quantile(0.99))
            + 0.05 * df["alteration_intensity"]
            + 0.04 * df["structural_score"]
        )
        return raw.clip(0.0, 1.0)

    def to_csv(self, path: str | Path, n_samples: int = 10_000) -> Path:
        """Generate and save samples as CSV."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.generate(n_samples).to_csv(out, index=False)
        return out


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic/antioquia_geochemistry.csv")
    )
    args = parser.parse_args()
    path = AntioquiaGeochemicalGenerator().to_csv(args.output, args.samples)
    print(f"Generated {args.samples} samples at {path}")


if __name__ == "__main__":
    main()
