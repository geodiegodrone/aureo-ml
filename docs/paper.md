# AUREO-ML: Machine Learning for Auriferous Prospectivity Mapping in Synthetic Antioquia Geochemistry

Diego F. Pulido Sastoque  
Ingeniero Geólogo, MSc Big Data, PhD candidate in Data Science and Advanced AI  
Email: dfpulidos@unal.edu.co

## Abstract

Gold exploration remains a high-risk, high-cost decision problem. Multi-element geochemical surveys provide evidence of hydrothermal systems, structural controls, lithological fertility, and pathfinder element associations, but classical single-element thresholding often fails to exploit the multivariate and spatial nature of these datasets. AUREO-ML presents a reproducible research-grade machine-learning workflow for auriferous exploration using synthetic but geologically informed Antioquia geochemistry. The workflow integrates compositional data analysis, anomaly detection, geochemical population clustering, spatial cross-validation, supervised prospectivity modeling, explainability, and interactive deployment through Streamlit. The project is intended as a transparent benchmark and portfolio artifact, not as a mineral resource estimate or claim recommendation.

## Keywords

gold exploration; geochemistry; mineral prospectivity mapping; machine learning; compositional data analysis; spatial cross-validation; SHAP; Antioquia

## 1. Introduction

Exploration teams must prioritize targets under uncertainty. Drill decisions typically depend on sparse observations, incomplete geological interpretation, biased sampling, and economic constraints. Geochemical datasets are particularly valuable because pathfinder elements can reveal mineralizing processes before a deposit is directly observed. However, geochemical signals are commonly right-skewed, compositional, spatially autocorrelated, and mixture-distributed. These properties make naive thresholding and random validation fragile.

AUREO-ML addresses this gap by implementing a complete workflow where geology-informed synthetic data generation, robust preprocessing, spatial validation, and explainable machine learning are treated as one reproducible system.

## 2. Geological Context

Antioquia, Colombia, is a historically significant gold province with orogenic, structurally controlled, epithermal, and intrusion-related exploration signatures. Gold mineralization may be associated with Au-Ag enrichment, arsenic-antimony pathfinder behavior, volatile pathfinders such as Hg and Te, and intrusion-related associations involving Bi, W, and Mo. The synthetic generator encodes these relationships at a regional scale through district centers, structural corridor proximity, alteration intensity, lithological fertility, and depth attenuation.

## 3. Data and Methods

The project generates 10,000 synthetic geochemical samples inside an Antioquia bounding box. Fifteen elements are simulated: Au, Ag, As, Sb, Cu, Pb, Zn, Fe, Mn, Hg, Bi, Te, Mo, W, and S. Elemental concentrations follow log-normal distributions because trace-element geochemistry is positive and often strongly right-skewed. A latent mineralization score controls pathfinder enrichment and is computed from hydrothermal alteration, structural proximity, lithology, and depth.

The pipeline includes:

- Pydantic validation for data quality.
- Log transformation for skew control.
- Centered log-ratio transformation for compositional interpretation.
- Geochemical indices such as Au/Ag, As/Sb, epithermal pathfinder index, intrusion-related index, and sulfide proxy.
- Neighbor-density and spatial block features.
- Isolation Forest, Local Outlier Factor, One-Class SVM, and Mahalanobis distance.
- K-Means, DBSCAN, and Gaussian Mixture Models.
- Random Forest and stacking classifiers.
- Spatial GroupKFold validation.
- MLflow experiment tracking.
- SHAP-ready explainability.

## 4. Validation Strategy

Random K-Fold validation is inappropriate for mineral exploration because adjacent samples share geology, alteration, sampling method, and campaign bias. AUREO-ML uses deterministic spatial blocks and GroupKFold to estimate model transfer to unseen regions. This design makes performance estimates more conservative and more relevant to exploration targeting.

## 5. Expected Results

On the synthetic benchmark, tree-based models generally achieve high ranking performance because the generator intentionally encodes realistic pathfinder controls. Typical local validation results are:

- AUC above 0.90 under spatial cross-validation.
- F1 above 0.80 for prospectivity classification.
- Coverage above 75%, currently near 90%.
- Anomaly consensus ranking that captures a high share of prospective samples in the top-ranked area.

These results should be interpreted as workflow verification, not discovery evidence.

## 6. Explainability

Explainability is critical because exploration geologists need to know why a target is ranked highly. AUREO-ML emphasizes Au, As, Sb, Hg, Te, Bi, W, Mo, alteration intensity, and structural score as interpretable controls. SHAP analysis, permutation importance, and partial dependence plots are included to connect model behavior back to geological reasoning.

## 7. Deployment

The Streamlit dashboard provides:

- Interactive folium prospectivity maps.
- EDA and geochemical scatterplots.
- Live prospectivity prediction from user-entered element values.
- Fast local contribution diagnostics.
- Multi-page navigation for exploration review.

For Streamlit Cloud, use:

- Repository: `geodiegodrone/aureo-ml`
- Branch: `main`
- Main file: `app/streamlit_app.py`

## 8. Limitations

The dataset is synthetic and should not be interpreted as real mineral potential, legal tenure, a reserve estimate, or a drilling recommendation. Real deployment requires certified assays, QA/QC, sample media metadata, geologic mapping, structural interpretation, laboratory method harmonization, censored-data treatment, and field validation.

## 9. Future Work

Future versions should integrate public Colombian geological data, USGS MRDS occurrences, fault and lithology layers, geophysical rasters, terrain features, Bayesian uncertainty, active-learning sampling design, and GeoPackage export for GIS teams.

## 10. Reproducibility

The repository includes fixed seeds, pinned dependencies, tests, coverage thresholds, linting, type checking, Docker, GitHub Actions, notebooks, and a Makefile. A minimal reproduction is:

```bash
make setup
make data
make train
make test
make app
```

## References

Full BibTeX references are available in `docs/references.bib`.
