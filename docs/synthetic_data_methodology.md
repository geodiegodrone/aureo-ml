# Synthetic Data Methodology

The synthetic dataset simulates 10,000 stream-sediment or soil-like geochemical samples in Antioquia, Colombia. It is designed for portfolio-grade reproducibility while preserving geological realism.

## Spatial Domain

Coordinates are sampled inside an Antioquia bounding box:

- Longitude: -77.15 to -73.85
- Latitude: 5.35 to 8.95

Four mineralized centers approximate known regional gold districts: Segovia-Remedios, Amalfi-Anori, Buritica, and Nordeste Antioqueno. Samples combine uniform regional background and clustered sampling around mineralized centers.

## Geochemical Model

All 15 elements are positive and simulated as log-normal variables:

`X_e = lognormal(mu_e + loading_e * M, sigma_e)`

where `M` is latent mineralization intensity controlled by:

- hydrothermal alteration intensity,
- structural corridor proximity,
- lithological fertility,
- depth attenuation,
- analytical/geological noise.

Au is generated in ppb-equivalent space and converted to g/t. Pathfinder loadings are strongest for Au, As, Sb, Hg, Te, Bi, W, and Ag.

## Targets

`prospectivity_score` is a bounded weighted score using Au and pathfinder ranks plus alteration and structure. The binary target `prospective` marks scores above 0.62. The multiclass label uses:

- low: score < 0.40
- medium: 0.40 <= score < 0.62
- high: score >= 0.62

## Scientific Justification

Gold exploration geochemistry commonly shows skewed populations, multiple background/anomalous mixtures, pathfinder associations, spatial autocorrelation, and sampling bias near known districts. The generator intentionally encodes those properties so that EDA, anomaly detection, spatial validation, and explainability methods face realistic ML failure modes.
