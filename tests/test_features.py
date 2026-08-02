"""
Tests for src/features.py

Verifies the Phase 5 feature-engineering logic stays exactly as frozen:
Power, Temp_diff, and Wear_Torque formulas, non-mutation of the input
dataframe, and correct application across train/val/test splits.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import add_engineered_features, add_engineered_features_to_splits


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Torque [Nm]': [10.0, 20.0],
        'Rotational speed [rpm]': [1000.0, 1500.0],
        'Process temperature [K]': [310.0, 312.5],
        'Air temperature [K]': [300.0, 301.5],
        'Tool wear [min]': [5.0, 8.0],
    })


def test_add_engineered_features_creates_columns(sample_df):
    result = add_engineered_features(sample_df)
    for col in ('Power', 'Temp_diff', 'Wear_Torque'):
        assert col in result.columns


def test_power_formula(sample_df):
    result = add_engineered_features(sample_df)
    expected = sample_df['Torque [Nm]'] * sample_df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    assert result['Power'].tolist() == pytest.approx(expected.tolist())


def test_temp_diff_formula(sample_df):
    result = add_engineered_features(sample_df)
    expected = sample_df['Process temperature [K]'] - sample_df['Air temperature [K]']
    assert result['Temp_diff'].tolist() == pytest.approx(expected.tolist())


def test_wear_torque_formula(sample_df):
    result = add_engineered_features(sample_df)
    expected = sample_df['Tool wear [min]'] * sample_df['Torque [Nm]']
    assert result['Wear_Torque'].tolist() == pytest.approx(expected.tolist())


def test_add_engineered_features_does_not_mutate_input(sample_df):
    original_columns = list(sample_df.columns)
    add_engineered_features(sample_df)
    assert list(sample_df.columns) == original_columns


def test_add_engineered_features_to_splits_applies_to_all_three(sample_df):
    X_train = sample_df.copy()
    X_val = sample_df.copy()
    X_test = sample_df.copy()

    result_train, result_val, result_test = add_engineered_features_to_splits(
        X_train, X_val, X_test
    )

    for result, original in (
        (result_train, X_train),
        (result_val, X_val),
        (result_test, X_test),
    ):
        for col in ('Power', 'Temp_diff', 'Wear_Torque'):
            assert col in result.columns
        assert len(result) == len(original)