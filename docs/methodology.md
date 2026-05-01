# Methodology

AUREO-ML follows CRISP-DM adapted to mineral exploration:

1. Business understanding: reduce drilling risk by ranking geochemical targets.
2. Data understanding: inspect multi-element assays, coordinates, lithology, depth, and sampling density.
3. Data preparation: validate coordinates, apply log transforms, centered log-ratio transforms, domain ratios, and spatial blocking.
4. Modeling: anomaly detection, clustering, supervised classification, Au regression, and stacking.
5. Evaluation: spatial cross-validation, AUC, F1, average precision, RMSE, and discovery efficiency.
6. Deployment: Streamlit dashboard and MLflow experiment tracking.

Normal K-Fold is intentionally avoided for headline validation because nearby samples share geology, alteration, weathering, and sampling campaigns. Spatial GroupKFold gives a more honest estimate of target transfer to untested blocks.

## Technical Decisions

- Log-normal concentrations represent the positive, right-skewed behavior of trace elements.
- CLR transformation addresses closure effects in compositional geochemistry.
- Au, As, Sb, Hg, Te, Bi, W, and Mo are emphasized as pathfinder and intrusion-related indicators.
- Tree ensembles are selected because they handle non-linear thresholds, interactions, and mixed feature scales.
- SHAP and permutation importance are used because exploration teams need interpretable element controls, not only scores.

## Limitations

Synthetic data demonstrates workflow maturity but does not replace certified assays, QA/QC, field mapping, geophysics, structural geology, or metallurgical tests.
