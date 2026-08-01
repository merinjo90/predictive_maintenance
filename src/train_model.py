"""
Model training module for the Predictive Maintenance project.

Trains the final selected model: class-weighted Random Forest,
200 trees, random_state=42. This matches the frozen Phase 6/7
model exactly - do not change these hyperparameters without
re-running the full validation/test workflow.
"""

from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": 42,
}


def train_random_forest(X_train, y_train):
    """
    Train the final selected Random Forest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (output of data_processing + features pipeline).
    y_train : pd.Series
        Training target ('Machine failure').

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Fitted model.
    """
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf