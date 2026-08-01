"""
Prediction module for the Predictive Maintenance project.

Loads the saved final model and metadata, and produces predictions
on new raw data using the exact same preprocessing pipeline used
during training (src/features.py's engineered features, applied to
already-cleaned/encoded feature columns).
"""

import json

import joblib

from src.features import add_engineered_features

MODEL_PATH = "models/random_forest_final.joblib"
METADATA_PATH = "models/model_metadata.json"


def load_model(model_path=MODEL_PATH):
    """Load the trained Random Forest model from disk."""
    return joblib.load(model_path)


def load_metadata(metadata_path=METADATA_PATH):
    """Load model metadata (threshold, feature names, etc.) from disk."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def predict(df_raw_features, model=None, metadata=None):
    """
    Predict machine failure probability and class for new data.

    Parameters
    ----------
    df_raw_features : pd.DataFrame
        Must already be cleaned and encoded the same way as training data
        (see src.data_processing.clean_and_encode) - i.e. contains
        'Torque [Nm]', 'Rotational speed [rpm]', 'Process temperature [K]',
        'Air temperature [K]', 'Tool wear [min]', 'Type_H', 'Type_L', 'Type_M'.
        Engineered features (Power, Temp_diff, Wear_Torque) are added here.
    model : fitted classifier, optional
        If not provided, loads the saved final model from disk.
    metadata : dict, optional
        If not provided, loads the saved metadata from disk.

    Returns
    -------
    dict with keys:
        'probability': array of failure probabilities
        'prediction': array of 0/1 predictions at the frozen threshold
        'threshold': the threshold used
    """
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()

    threshold = metadata["threshold"]

    # Add engineered features (Power, Temp_diff, Wear_Torque)
    df_with_features = add_engineered_features(df_raw_features)

    # Align to the exact feature order the model was trained on
    feature_names = metadata["feature_names"]
    X = df_with_features[feature_names]

    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)

    return {
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,
    }