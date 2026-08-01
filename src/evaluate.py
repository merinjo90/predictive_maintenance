"""
Evaluation module for the Predictive Maintenance project.

Computes the standard metric set used throughout this project:
precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix breakdown,
and failure-detection counts.
"""

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

FINAL_THRESHOLD = 0.30


def evaluate_model(model, X, y, threshold=FINAL_THRESHOLD):
    """
    Evaluate a fitted model on a given feature/target set.

    Returns
    -------
    dict
        Dictionary of metrics and confusion-matrix breakdown.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "threshold": threshold,
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "pr_auc": round(average_precision_score(y, y_proba), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "failures_detected": int(tp),
        "failures_missed": int(fn),
        "false_alarms": int(fp),
        "total_rows": int(len(y)),
    }
    return results


def print_evaluation(results):
    """Pretty-print an evaluation results dict."""
    print(f"Threshold: {results['threshold']}")
    print()
    print("Confusion matrix:")
    print(f"  TN={results['true_negatives']}  FP={results['false_positives']}")
    print(f"  FN={results['false_negatives']}  TP={results['true_positives']}")
    print()
    print(f"Precision: {results['precision']}")
    print(f"Recall:    {results['recall']}")
    print(f"F1:        {results['f1']}")
    print(f"ROC-AUC:   {results['roc_auc']}")
    print(f"PR-AUC:    {results['pr_auc']}")
    print()
    print(f"Failures detected: {results['failures_detected']}")
    print(f"Failures missed:   {results['failures_missed']}")
    print(f"False alarms:      {results['false_alarms']}")
    print(f"Total test rows:   {results['total_rows']}")