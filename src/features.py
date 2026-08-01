"""
Feature engineering module for the Predictive Maintenance project.

Adds three engineered features to a features dataframe:
- Power = Torque x Rotational speed x (2*pi / 60)
- Temp_diff = Process temperature - Air temperature
- Wear_Torque = Tool wear x Torque

This logic is moved as-is from notebooks/01_data_exploration.ipynb
(Phase 5 - Feature Engineering). Do not change the formulas without
re-running the full validation/test workflow.
"""

import numpy as np


def add_engineered_features(df):
    """
    Add Power, Temp_diff, and Wear_Torque columns to a copy of df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: 'Torque [Nm]', 'Rotational speed [rpm]',
        'Process temperature [K]', 'Air temperature [K]',
        'Tool wear [min]'

    Returns
    -------
    pd.DataFrame
        A copy of df with the three engineered columns added.
    """
    dataset = df.copy()

    dataset['Power'] = (
        dataset['Torque [Nm]']
        * dataset['Rotational speed [rpm]']
        * (2 * np.pi / 60)
    )

    dataset['Temp_diff'] = (
        dataset['Process temperature [K]'] - dataset['Air temperature [K]']
    )

    dataset['Wear_Torque'] = (
        dataset['Tool wear [min]'] * dataset['Torque [Nm]']
    )

    return dataset


def add_engineered_features_to_splits(X_train, X_val, X_test):
    """
    Apply add_engineered_features to train, validation, and test sets.

    Matches the original notebook loop over [X_train, X_val, X_test],
    but returns new dataframes rather than mutating in place.

    Returns
    -------
    X_train, X_val, X_test (with engineered features added)
    """
    X_train = add_engineered_features(X_train)
    X_val = add_engineered_features(X_val)
    X_test = add_engineered_features(X_test)
    return X_train, X_val, X_test