cat > README.md << 'EOF'
# Predictive Maintenance System Using Machine Learning

An end-to-end machine learning pipeline for predicting industrial machine
failures, built on the AI4I 2020 synthetic predictive-maintenance
benchmark. This is an educational portfolio project demonstrating
reproducible ML engineering practice — data preparation, feature
engineering, model selection, frozen evaluation, modular reusable code,
and automated testing.

**Important limitation:** This project uses a synthetic, publicly
available benchmark dataset. It demonstrates methodology, reproducibility,
and engineering practice — it does not prove guaranteed performance in a
real factory environment.

---

## Dataset

- Source: AI4I 2020 synthetic predictive-maintenance benchmark
- 10,000 rows × 14 columns
- No missing values, no duplicate rows
- Target: `Machine failure` (binary)
- Class distribution: 9,661 normal rows, 339 failure rows (3.39% failure rate)

---

## Project Structure
predictive_maintenance/
├── app/ # (planned) FastAPI / Streamlit application
├── data/
│ ├── raw/ # Original AI4I 2020 CSV
│ └── processed/ # Processed data artifacts
├── models/
│ ├── model_metadata.json # Frozen model parameters, threshold, feature names, metrics
│ └── README.md # Model reproducibility notes
├── notebooks/
│ └── 01_data_exploration.ipynb # EDA, feature engineering, model comparison
├── reports/
│ └── figures/ # Saved EDA charts
├── scripts/
│ └── save_final_model.py # Deterministic training + save with safety check
├── src/
│ ├── data_processing.py # Load, clean, encode, split
│ ├── features.py # Engineered feature logic
│ ├── train_model.py # Frozen model training
│ ├── evaluate.py # Metric computation
│ └── predict.py # Inference on new data
├── tests/
│ ├── test_data_processing.py
│ ├── test_features.py
│ ├── test_train_model.py
│ ├── test_evaluate.py
│ └── test_predict.py
├── Dockerfile # (planned)
└── requirements.txt

---

## Pipeline Overview

### 1. Exploratory Data Analysis
Verified dataset shape, checked for missing values and duplicates,
analyzed class imbalance, reviewed correlations and failure-type counts,
and documented the synthetic-data limitation.

### 2. Data Preparation
- Dropped identifier columns: `UDI`, `Product ID`
- Excluded leakage columns (features that describe *how* a failure
  happened, not usable as predictors): `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- One-hot encoded the `Type` column
- Stratified train/validation/test split preserving the 3.39% failure rate:
  - Train: 6,999 rows
  - Validation: 1,501 rows
  - Test: 1,500 rows
- `random_state=42` throughout for reproducibility

### 3. Feature Engineering
Three engineered features were added on top of the raw sensor columns:

| Feature | Formula |
|---|---|
| `Power` | `Torque [Nm] × Rotational speed [rpm] × (2π / 60)` |
| `Temp_diff` | `Process temperature [K] − Air temperature [K]` |
| `Wear_Torque` | `Tool wear [min] × Torque [Nm]` |

### 4. Model Selection
Multiple approaches were compared on the validation set (never the test
set) before a final model was chosen:

- Majority-class baseline
- Logistic Regression
- Class-weighted Random Forest
- Random Forest with SMOTE oversampling
- XGBoost
- Threshold sweep on the class-weighted Random Forest

The class-weighted Random Forest at a tuned decision threshold gave the
best validation trade-off for failure detection (recall and F1) without
ever touching the test set for tuning.

### 5. Final Test Evaluation (frozen)
The selected model was evaluated exactly once on the untouched 1,500-row
test set. These results are frozen and are not re-tuned:

**Model configuration:**
- Class-weighted Random Forest
- `n_estimators = 200`
- `threshold = 0.30`
- `random_state = 42`

**Test-set metrics:**

| Metric | Value |
|---|---|
| Precision | 0.9020 |
| Recall | 0.9020 |
| F1 | 0.9020 |
| ROC-AUC | 0.9910 |
| PR-AUC | 0.9371 |

**Confusion matrix:**

|  | Predicted: No failure | Predicted: Failure |
|---|---|---|
| **Actual: No failure** | 1,444 (TN) | 5 (FP) |
| **Actual: Failure** | 5 (FN) | 46 (TP) |

Out of 51 real failures in the test set, 46 were detected, 5 were missed,
and 5 false alarms were raised.

---

## Reusable Source Modules

The full pipeline was refactored out of the exploratory notebook into
tested, reusable modules:

| Module | Purpose |
|---|---|
| `src/data_processing.py` | Loads the raw CSV, drops identifiers, excludes leakage columns, one-hot encodes `Type`, and produces the stratified train/val/test split |
| `src/features.py` | Adds the `Power`, `Temp_diff`, and `Wear_Torque` engineered features, single-dataframe and multi-split versions |
| `src/train_model.py` | Trains the frozen class-weighted Random Forest with fixed hyperparameters |
| `src/evaluate.py` | Computes precision, recall, F1, ROC-AUC, PR-AUC, and confusion-matrix breakdown at the frozen threshold |
| `src/predict.py` | Loads the saved model and metadata (or accepts injected ones) and produces predictions on new, already-cleaned data |

`scripts/save_final_model.py` retrains the model deterministically and
includes a built-in safety check: it compares the freshly trained model's
metrics against the frozen Phase 7 results and only saves the model if
they match exactly.

---

## Automated Tests

27 tests across 5 files, all passing:

| Test file | Tests | Covers |
|---|---|---|
| `test_data_processing.py` | 7 | Identifier/leakage column exclusion, `Type` encoding, split sizes, stratification, end-to-end pipeline |
| `test_features.py` | 6 | Engineered feature formulas, non-mutation of input, multi-split application |
| `test_train_model.py` | 4 | Fitted model returned, frozen hyperparameters, training determinism, valid prediction labels |
| `test_evaluate.py` | 5 | Metric dict shape, default threshold, confusion-matrix correctness, alias-field consistency, row count |
| `test_predict.py` | 5 | Result dict shape, metadata-driven threshold, engineered features applied before scoring, threshold-based prediction, feature-name order alignment |

Run the full suite with:

```bash
python -m pytest
```

> Note: use `python -m pytest`, not a bare `pytest` command — this
> guarantees the test run uses the active virtual environment's installed
> pytest rather than any other `pytest` that might be earlier on `PATH`.

---

## Setup & Reproduction

```bash
# Clone the repository
git clone https://github.com/merinjo90/predictive_maintenance.git
cd predictive_maintenance

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest

# Retrain and save the final model (with a built-in safety check
# against the frozen Phase 7 metrics)
python scripts/save_final_model.py
```

---

## Status & Remaining Work

**Completed:**
- Exploratory data analysis
- Data preparation and stratified splitting
- Feature engineering
- Model comparison and selection
- Frozen final test evaluation
- Reusable, tested source modules
- Full automated test suite (27 tests)

**Not yet implemented:**
- FastAPI serving layer
- Streamlit dashboard
- Docker integration
- Deployment

---

## License

This project is for educational and portfolio purposes. The AI4I 2020
dataset is a publicly available synthetic benchmark.
EOF