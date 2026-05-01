# API Reference

## Data

- `AntioquiaGeochemicalGenerator.generate(n_samples)`: deterministic synthetic multi-element dataset.
- `load_geochemistry_csv(path)`: CSV loader.
- `to_geodataframe(df)`: converts longitude/latitude to GeoDataFrame.
- `validate_samples(df)`: Pydantic validation.

## Features

- `clr_transform(df)`: centered log-ratio transform.
- `add_geochemical_indices(df)`: Au/Ag, As/Sb, epithermal, intrusion-related, and sulfide indices.
- `add_spatial_features(df)`: neighbor density and coordinate interactions.
- `assign_spatial_blocks(df)`: deterministic spatial CV groups.

## Models

- `fit_anomaly_detectors(df)`: Isolation Forest, LOF, One-Class SVM, Mahalanobis.
- `cluster_geochemical_populations(df)`: K-Means, DBSCAN, GMM.
- `build_classifier()`: random-forest prospectivity classifier.
- `build_regressor()`: random-forest Au regressor.
- `train_stacking(input_path, output_path)`: MLflow-tracked stacking training.
