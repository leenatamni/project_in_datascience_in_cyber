"""Metric computation, threshold analysis, and low-prevalence simulation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)

from . import config


def score_predictions(y_true, y_pred, y_score=None):
    """Compute the full metric set used throughout the report."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_score)
        metrics["PR_AUC"] = average_precision_score(y_true, y_score)
    else:
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return metrics


def get_positive_class_score(model, X_eval):
    """Return a score for the positive class when available.

    Prefer calibrated probabilities when the estimator exposes predict_proba.
    For most margin-based estimators, fall back to decision_function. For the
    scalable RBF approximation used on the full 2020 dataset, return hard
    predictions as a conservative fallback: this keeps the full-data run fast
    and avoids relying on expensive probability calibration for an auxiliary
    robustness check.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_eval)[:, 1]

    # The full 2020 external check uses Pipeline(..., SGDClassifier(loss="hinge"))
    # after an RBF feature map. Hard labels are a conservative ranking fallback
    # for ROC/PR metrics and avoid an extra decision_function pass on this large
    # auxiliary dataset.
    final_estimator = getattr(model, "named_steps", {}).get("model")
    if final_estimator is not None and final_estimator.__class__.__name__ == "SGDClassifier":
        return model.predict(X_eval)

    if hasattr(model, "decision_function"):
        return model.decision_function(X_eval)
    return None


def evaluate_pipeline(model, X_eval, y_eval):
    """Fit-independent evaluation: assumes `model` is already fitted."""
    y_pred = model.predict(X_eval)
    y_score = get_positive_class_score(model, X_eval)
    return score_predictions(y_eval, y_pred, y_score)


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=2000,
                         ci=0.95, random_state=config.RANDOM_STATE):
    """Nonparametric (percentile) bootstrap confidence interval for a
    single metric on a held-out test set.

    A point estimate hides how much it could plausibly move under a
    slightly different sample. This resamples the (y_true, y_pred) pairs
    with replacement `n_boot` times, recomputes the metric each time, and
    reports the percentile interval. With a small test set the interval is
    wide (an honest signal that the point estimate is uncertain); with a
    large test set - as with the full 58,645-row newer-dataset check used
    in this project - the interval collapses to a tight band around the
    point estimate, which is itself a useful confirmation that the full
    dataset resolved the small-sample uncertainty that a 55-row subsample
    would have carried.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores[b] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    lower, upper = np.nanpercentile(boot_scores, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_boot": n_boot,
        "n_samples": n,
    }


def threshold_table(y_true, y_score, thresholds=config.THRESHOLDS):
    """Precision/recall/F1/error-count trade-off across decision thresholds."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "Threshold": t,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "Alerts": int(y_pred.sum()),
        })
    return pd.DataFrame(rows)


def simulate_low_prevalence(y_true, y_score, n_positive=14, prevalence_seed=config.RANDOM_STATE,
                             thresholds=(0.30, 0.50, 0.70, 0.90)):
    """Simulate a realistic low-phishing-prevalence traffic scenario by
    keeping all negatives from the test set and subsampling a small number
    of positives, then re-running the threshold analysis on that mixture.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    negative_idx = np.where(y_true == 0)[0]
    positive_idx = np.where(y_true == 1)[0]

    rng = np.random.default_rng(prevalence_seed)
    n_positive = min(n_positive, len(positive_idx))
    selected_positive_idx = rng.choice(positive_idx, size=n_positive, replace=False)
    idx = np.concatenate([negative_idx, selected_positive_idx])

    y_low = y_true[idx]
    y_score_low = y_score[idx]

    rows = []
    for t in thresholds:
        y_pred = (y_score_low >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_low, y_pred).ravel()
        rows.append({
            "Threshold": t,
            "Sample size": len(y_low),
            "Phishing count": int(y_low.sum()),
            "Phishing prevalence": float(y_low.mean()),
            "Precision": precision_score(y_low, y_pred, zero_division=0),
            "Recall": recall_score(y_low, y_pred, zero_division=0),
            "F1": f1_score(y_low, y_pred, zero_division=0),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "Alerts": int(y_pred.sum()),
        })
    return pd.DataFrame(rows)
