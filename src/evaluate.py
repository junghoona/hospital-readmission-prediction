"""evaluate.py — shared classification metrics (imbalance-aware)."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, accuracy_score,
                             confusion_matrix)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
    }


def confusion(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
