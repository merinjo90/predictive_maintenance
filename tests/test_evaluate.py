"""
Tests for src/evaluate.py

Verifies evaluate_model returns the expected metric dict, uses the frozen
threshold (0.30) by default, and computes confusion-matrix-derived values
correctly against a known, hand-verifiable prediction set.
"""
import numpy as np
import pandas as pd

from src.evaluate import evaluate_model, FINAL_THRESHOLD


class FakeModel:
    """Stub model with a fixed predict_proba, for deterministic testing."""

    def __init__(self, probabilities):
        self._probabilities = np.array(probabilities)

    def predict_proba(self, X):
        neg = 1 - self._probabilities
        return np.column_stack([neg, self._probabilities])


def test_evaluate_model_returns_expected_keys():
    model = FakeModel([0.1, 0.9, 0.2, 0.8])
    X = pd.DataFrame({'a': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    results = evaluate_model(model, X, y)

    expected_keys = {
        "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
        "failures_detected", "failures_missed", "false_alarms", "total_rows",
    }
    assert expected_keys.issubset(results.keys())


def test_evaluate_model_uses_default_threshold():
    model = FakeModel([0.1, 0.9, 0.2, 0.8])
    X = pd.DataFrame({'a': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    results = evaluate_model(model, X, y)

    assert results['threshold'] == FINAL_THRESHOLD == 0.30


def test_evaluate_model_confusion_matrix_correctness():
    # y_true:  0    1    0    1    0
    # y_proba: 0.1  0.9  0.4  0.2  0.5
    # threshold 0.30 -> y_pred: 0    1    1    0    1
    # TN: row0 (0,0) -> 1
    # FP: row2 (0,1) -> 1
    # FN: row3 (1,0) -> 1
    # TP: row1 (1,1) -> 1
    # row4: true=0, pred=1 (0.5>=0.3) -> FP
    model = FakeModel([0.1, 0.9, 0.4, 0.2, 0.5])
    X = pd.DataFrame({'a': range(5)})
    y = pd.Series([0, 1, 0, 1, 0])

    results = evaluate_model(model, X, y)

    assert results['true_negatives'] == 1
    assert results['false_positives'] == 2
    assert results['false_negatives'] == 1
    assert results['true_positives'] == 1


def test_evaluate_model_failure_aliases_match_confusion_matrix():
    model = FakeModel([0.1, 0.9, 0.4, 0.2, 0.5])
    X = pd.DataFrame({'a': range(5)})
    y = pd.Series([0, 1, 0, 1, 0])

    results = evaluate_model(model, X, y)

    assert results['failures_detected'] == results['true_positives']
    assert results['failures_missed'] == results['false_negatives']
    assert results['false_alarms'] == results['false_positives']


def test_evaluate_model_total_rows_matches_input_length():
    model = FakeModel([0.1, 0.9, 0.4, 0.2, 0.5])
    X = pd.DataFrame({'a': range(5)})
    y = pd.Series([0, 1, 0, 1, 0])

    results = evaluate_model(model, X, y)

    assert results['total_rows'] == len(y)