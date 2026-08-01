"""
Data processing module for the Predictive Maintenance project.

Loads the raw AI4I 2020 dataset, drops identifier columns, one-hot encodes
Type, excludes leakage columns, and produces the stratified
train/validation/test split.

This logic is moved as-is from notebooks/01_data_exploration.ipynb
(Phase 4 — Data Preparation). Do not change the split logic, random_state,
or leakage columns without re-running the full validation/test workflow.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Columns with no predictive value
ID_COLUMNS = ['UDI', 'Product ID']

# Columns that describe HOW a failure happened - not usable as predictors
LEAKAGE_COLUMNS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

TARGET_COLUMN = 'Machine failure'


def load_raw_data(csv_path="data/raw/ai4i2020.csv"):
    """Load the raw AI4I 2020 dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def clean_and_encode(df):
    """
    Drop identifier columns and one-hot encode the Type column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from ai4i2020.csv

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with identifiers dropped and Type one-hot encoded.
    """
    df_model = df.drop(columns=ID_COLUMNS)
    df_model = pd.get_dummies(df_model, columns=['Type'], prefix='Type')
    return df_model


def split_features_target(df_model):
    """
    Separate features (X) and target (y), excluding leakage columns.

    Parameters
    ----------
    df_model : pd.DataFrame
        Cleaned, encoded dataframe (output of clean_and_encode).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    y = df_model[TARGET_COLUMN]
    X = df_model.drop(columns=[TARGET_COLUMN] + LEAKAGE_COLUMNS)
    return X, y


def train_val_test_split(X, y, random_state=42):
    """
    Create the stratified train/validation/test split.

    Matches Phase 4 exactly:
    - First split: 15% held out as test set
    - Second split: 0.1765 of the remaining 85% held out as validation
      (≈15% of the original total)

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare_data(csv_path="data/raw/ai4i2020.csv", random_state=42):
    """
    Full Phase 4 pipeline: load, clean, encode, split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = load_raw_data(csv_path)
    df_model = clean_and_encode(df)
    X, y = split_features_target(df_model)
    return train_val_test_split(X, y, random_state=random_state)