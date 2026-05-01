# Streamlit Cloud Deployment

Use this checklist to publish AUREO-ML as a public interactive app.

## Settings

- Repository: `geodiegodrone/aureo-ml`
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Python version: 3.11

## Required Files

Streamlit Cloud reads:

- `requirements.txt`
- `.streamlit/config.toml`
- `app/streamlit_app.py`
- `src/aureo_ml/**`

The app generates deterministic synthetic data on first load, so no external data upload is required.

## Local Smoke Test

```bash
python -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Troubleshooting

If Streamlit Cloud cannot import `aureo_ml`, ensure the repository root is present and `src/` is included. The app explicitly inserts `src/` into `sys.path` for cloud compatibility.
