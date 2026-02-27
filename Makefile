.PHONY: all data train test lint clean app help

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON   := python3
PIP      := pip3
DATA_DIR := data/raw
MODEL    := models/best_model.joblib

# ── Default target ────────────────────────────────────────────────────────────
all: data train test

# ── Setup ─────────────────────────────────────────────────────────────────────
install:          ## Install all dependencies
	$(PIP) install -r requirements.txt

# ── Data pipeline ─────────────────────────────────────────────────────────────
data:             ## Generate synthetic dataset (10,000 rows)
	$(PYTHON) scripts/generate_data.py

data-large:       ## Generate larger dataset (50,000 rows)
	$(PYTHON) scripts/generate_data.py --samples 50000 --seed 7

# ── Model training ────────────────────────────────────────────────────────────
train: $(DATA_DIR)/telco_churn.csv  ## Train all models and save best
	$(PYTHON) scripts/train_pipeline.py

# ── Testing ───────────────────────────────────────────────────────────────────
test:             ## Run all unit tests with pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:         ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────────────────────
lint:             ## Lint with flake8
	$(PYTHON) -m flake8 src/ scripts/ app/ --max-line-length=100 --ignore=E501,W503

format:           ## Format with black
	$(PYTHON) -m black src/ scripts/ app/ notebooks/ --line-length=100

# ── Notebooks ─────────────────────────────────────────────────────────────────
eda:              ## Execute EDA notebook and save outputs
	jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb \
	  --output 01_EDA.ipynb --output-dir notebooks/ \
	  --ExecutePreprocessor.timeout=300

model-nb:         ## Execute Modeling notebook and save outputs
	jupyter nbconvert --to notebook --execute notebooks/02_Modeling.ipynb \
	  --output 02_Modeling.ipynb --output-dir notebooks/ \
	  --ExecutePreprocessor.timeout=300

notebooks: eda model-nb  ## Execute both notebooks

# ── App ───────────────────────────────────────────────────────────────────────
app:              ## Launch the Streamlit prediction app
	streamlit run app/streamlit_app.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:            ## Remove generated data, models, and figures
	rm -f data/raw/telco_churn.csv
	rm -f data/processed/*.csv data/processed/*.pkl
	rm -f models/best_model.joblib
	rm -f reports/figures/*.png reports/figures/*.jpg
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-nb:         ## Clear notebook outputs
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True \
	  --to notebook notebooks/*.ipynb --inplace

# ── Help ──────────────────────────────────────────────────────────────────────
help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
