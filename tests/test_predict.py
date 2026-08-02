"""
Tests for src/predict.py

Verifies predict() returns the expected result dict, correctly uses the
threshold and feature order from injected metadata, applies engineered
features before scoring, and computes the threshold decision correctly.
Model and metadata are injected directly so these tests don't depend on
the real saved .joblib/.json artifacts on disk.
"""
import numpy as np
import pandas as pd

from src.predict import predict


RAW_COLUMNS = {
    'Torque [Nm]': [10.0, 20.0],
    'Rotational speed [rpm]': [1000.0, 1500.0],
    'Process temperature [K]': [310.0, 312.5],
    'Air temperature [K]': [300.0, 301.5],
    'Tool wear [min]': [5.0, 8.0],
    'Type_H': [1, 0],
    'Type_L': [0, 1],
    'Type_M': [0, 0],
}

FEATURE_NAMES = list(RAW_COLUMNS.keys()) + ['Power', 'Temp_diff', 'Wear_Torque']


class FakeModel:
    """Stub model with a configurable predict_proba, for deterministic testing."""

    def __init__(self, probabilities, expected_columns=None):
        self._probabilities = np.array(probabilities)
        self._expected_columns = expected_columns

    def predict_proba(self, X):
        if self._expected_columns is not None:
            assert list(X.columns) == self._expected_columns
        neg = 1 - self._probabilities
        return np.column_stack([neg, self._probabilities])


def make_raw_df():
    return pd.DataFrame(RAW_COLUMNS)


def test_predict_returns_expected_keys():
    df = make_raw_df()
    model = FakeModel([0.1, 0.9])
    metadata = {"threshold": 0.30, "feature_names": FEATURE_NAMES}

    result = predict(df, model=model, metadata=metadata)

    assert set(result.keys()) == {"probability", "prediction", "threshold"}


def test_predict_uses_metadata_threshold():
    df = make_raw_df()
    model = FakeModel([0.1, 0.9])
    metadata = {"threshold": 0.65, "feature_names": FEATURE_NAMES}

    result = predict(df, model=model, metadata=metadata)

    assert result["threshold"] == 0.65


def test_predict_applies_engineered_features():
    df = make_raw_df()
    model = FakeModel([0.1, 0.9], expected_columns=FEATURE_NAMES)
    metadata = {"threshold": 0.30, "feature_names": FEATURE_NAMES}

    predict(df, model=model, metadata=metadata)


def test_predict_prediction_matches_threshold():
    df = make_raw_df()
    model = FakeModel([0.2, 0.8])
    metadata = {"threshold": 0.30, "feature_names": FEATURE_NAMES}

    result = predict(df, model=model, metadata=metadata)

    expected = (result["probability"] >= 0.30).astype(int)
    assert list(result["prediction"]) == list(expected)
    assert list(result["prediction"]) == [0, 1]


def test_predict_respects_feature_name_order():
    df = make_raw_df()
    reordered_names = list(reversed(FEATURE_NAMES))
    model = FakeModel([0.1, 0.9], expected_columns=reordered_names)
    metadata = {"threshold": 0.30, "feature_names": reordered_names}

    predict(df, model=model, metadata=metadata)
