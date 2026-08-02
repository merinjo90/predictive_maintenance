"""
Tests for src/train_model.py

Verifies the Phase 6/7 frozen training logic stays exactly as specified:
class-weighted Random Forest with n_estimators=200, class_weight='balanced',
random_state=42, deterministic output, and valid predictions.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train_model import train_random_forest, RF_PARAMS


def make_synthetic_data(n_rows=30, random_state=0):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        'feature_a': rng.normal(size=n_rows),
        'feature_b': rng.normal(size=n_rows),
    })
    y = pd.Series([0, 1] * (n_rows // 2))
    return X, y


def test_train_random_forest_returns_fitted_model():
    X_train, y_train = make_synthetic_data()
    model = train_random_forest(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'estimators_')


def test_train_random_forest_uses_frozen_params():
    X_train, y_train = make_synthetic_data()
    model = train_random_forest(X_train, y_train)
    assert model.n_estimators == RF_PARAMS['n_estimators'] == 200
    assert model.class_weight == RF_PARAMS['class_weight'] == 'balanced'
    assert model.random_state == RF_PARAMS['random_state'] == 42


def test_train_random_forest_is_deterministic():
    X_train, y_train = make_synthetic_data()
    model_1 = train_random_forest(X_train, y_train)
    model_2 = train_random_forest(X_train, y_train)
    preds_1 = model_1.predict(X_train)
    preds_2 = model_2.predict(X_train)
    assert list(preds_1) == list(preds_2)


def test_train_random_forest_predicts_valid_labels():
    X_train, y_train = make_synthetic_data()
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_train)
    assert set(preds).issubset({0, 1})