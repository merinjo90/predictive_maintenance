# Models

Trained model artifacts (`.joblib` files) are intentionally **not committed**
to this repository — see `.gitignore`.

## Why

The training pipeline is fully deterministic: every step (data split, feature
engineering, model training) uses `random_state=42`. Running the training
script on any machine with the same dataset reproduces the exact same model
and the exact same final test metrics every time. Committing large trained
binaries adds unnecessary bloat to Git history when the artifact can be
regenerated reliably from code.

## How to regenerate the model

From the project root, with the virtual environment active:

```bash
PYTHONPATH=. python scripts/save_final_model.py
```

This will:
1. Load and prepare the data (`src/data_processing.py`)
2. Add engineered features (`src/features.py`)
3. Train the final Random Forest (`src/train_model.py`)
4. Evaluate on the test set and **verify it matches the frozen Phase 7 results**
   before saving anything
5. Save the model to `models/random_forest_final.joblib`
6. Save metadata (threshold, feature names, params, frozen metrics) to
   `models/model_metadata.json`

If the safety check fails (results don't match the frozen Phase 7 numbers),
the script will raise an error and refuse to save — this protects against
accidentally saving a different model than the one that was validated.

## Final model summary

- Type: Random Forest (class-weighted)
- Trees: 200
- Threshold: 0.30
- See `model_metadata.json` (generated after running the script) for full
  feature list and frozen test metrics.