"""
Tests for src/data_processing.py

Verifies the Phase 4 data-preparation pipeline stays exactly as frozen:
identifier columns dropped, Type one-hot encoded, leakage columns excluded,
stratified split sizes, and stratification quality.
"""
import pytest

from src.data_processing import (
    load_raw_data,
    clean_and_encode,
    split_features_target,
    train_val_test_split,
    load_and_prepare_data,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)

CSV_PATH = "data/raw/ai4i2020.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(CSV_PATH)


@pytest.fixture(scope="module")
def encoded_df(raw_df):
    return clean_and_encode(raw_df)


@pytest.fixture(scope="module")
def X_y(encoded_df):
    return split_features_target(encoded_df)


def test_clean_and_encode_drops_identifiers(encoded_df):
    for col in ID_COLUMNS:
        assert col not in encoded_df.columns


def test_clean_and_encode_encodes_type(encoded_df):
    assert "Type" not in encoded_df.columns
    type_dummy_cols = [c for c in encoded_df.columns if c.startswith("Type_")]
    assert len(type_dummy_cols) > 0


def test_split_features_target_excludes_leakage(X_y):
    X, _ = X_y
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns


def test_split_features_target_excludes_target(X_y):
    X, y = X_y
    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN


def test_train_val_test_split_sizes(X_y):
    X, y = X_y
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=42
    )
    assert len(X_train) == 6999
    assert len(X_val) == 1501
    assert len(X_test) == 1500


def test_split_is_stratified(X_y):
    X, y = X_y
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=42
    )
    overall_rate = y.mean()
    for split_y in (y_train, y_val, y_test):
        assert abs(split_y.mean() - overall_rate) < 0.005  # within 0.5 pp


def test_load_and_prepare_data_end_to_end():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        CSV_PATH, random_state=42
    )
    assert len(X_train) == 6999
    assert len(X_val) == 1501
    assert len(X_test) == 1500