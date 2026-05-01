.PHONY: setup data train test lint app clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install

data:
	python -m aureo_ml.data.synthetic_generator --samples 10000 --output data/synthetic/antioquia_geochemistry.csv

train: data
	python -m aureo_ml.models.ensemble --input data/synthetic/antioquia_geochemistry.csv --output models/aureo_stacking.joblib

test:
	pytest

lint:
	ruff check src tests
	black --check src tests app
	mypy src

app:
	streamlit run app/streamlit_app.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','mlruns']]"
