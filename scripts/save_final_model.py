"""
One-time script to train the final selected model and save it to disk.

Run this script once to produce:
- models/random_forest_final.joblib
- models/model_metadata.json

Do not change the model, threshold, or features here without
re-running the full validation/test workflow first.
"""

import json
from datetime import datetime

import joblib

from src.data_processing import load_and_prepare_data
from src.features import add_engineered_features_to_splits
from src.train_model import train_random_forest, RF_PARAMS
from src.evaluate import evaluate_model, print_evaluation, FINAL_THRESHOLD

MODEL_PATH = "models/random_forest_final.joblib"
METADATA_PATH = "models/model_metadata.json"

# The frozen Phase 7 test results, for safety comparison after retraining
FROZEN_TEST_RESULTS = {
    "precision": 0.9020,
    "recall": 0.9020,
    "f1": 0.9020,
    "roc_auc": 0.9910,
    "pr_auc": 0.9371,
    "failures_detected": 46,
    "failures_missed": 5,
    "false_alarms": 5,
    "total_rows": 1500,
}


def main():
    print("Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    X_train, X_val, X_test = add_engineered_features_to_splits(
        X_train, X_val, X_test
    )

    print("Training final Random Forest model...")
    model = train_random_forest(X_train, y_train)

    print("\nEvaluating on test set (safety check against frozen results)...")
    results = evaluate_model(model, X_test, y_test, threshold=FINAL_THRESHOLD)
    print_evaluation(results)

    # Safety check: confirm this matches the frozen Phase 7 results
    mismatches = []
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc",
                "failures_detected", "failures_missed", "false_alarms",
                "total_rows"]:
        if results[key] != FROZEN_TEST_RESULTS[key]:
            mismatches.append(
                f"{key}: expected {FROZEN_TEST_RESULTS[key]}, got {results[key]}"
            )

    if mismatches:
        print("\nWARNING: Results do not match frozen Phase 7 results:")
        for m in mismatches:
            print(" -", m)
        raise SystemExit(
            "Aborting save: retrained model does not match frozen results. "
            "Do not save a model that differs from the frozen evaluation."
        )

    print("\nResults match frozen Phase 7 test results exactly. Proceeding to save.")

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "model_params": RF_PARAMS,
        "threshold": FINAL_THRESHOLD,
        "feature_names": list(X_train.columns),
        "n_features": len(X_train.columns),
        "trained_on_rows": len(X_train),
        "frozen_test_results": FROZEN_TEST_RESULTS,
        "saved_at": datetime.now().isoformat(),
        "notes": (
            "Class-weighted Random Forest, final selected model. "
            "Dataset: AI4I 2020 (synthetic). "
            "Do not retune after this point."
        ),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()