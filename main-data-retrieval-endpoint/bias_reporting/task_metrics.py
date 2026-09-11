"""Task-specific performance metrics for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .statistics import expected_calibration_error, rank_auc


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def binary_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    positive_label: str,
    scores: pd.Series | None = None,
) -> dict[str, Any]:
    actual_positive = actual.astype("string").str.strip() == positive_label.strip()
    predicted_positive = predicted.astype("string").str.strip() == positive_label.strip()
    tp = int((actual_positive & predicted_positive).sum())
    tn = int((~actual_positive & ~predicted_positive).sum())
    fp = int((~actual_positive & predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    metrics: dict[str, Any] = {
        "accuracy": safe_div(tp + tn, len(actual)),
        "balanced_accuracy": (
            None if recall is None or specificity is None else (recall + specificity) / 2
        ),
        "precision": precision,
        "negative_predictive_value": safe_div(tn, tn + fn),
        "true_positive_rate": recall,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "specificity": specificity,
        "f1": (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        ),
        "matthews_correlation_coefficient": safe_div(
            tp * tn - fp * fn,
            float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
        ),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    if scores is not None:
        numeric = pd.to_numeric(scores, errors="coerce")
        valid = numeric.notna()
        y = actual_positive[valid].to_numpy()
        probability = numeric[valid].clip(0, 1).to_numpy()
        metrics["score_metrics"] = {
            "roc_auc": rank_auc(y, probability),
            "brier_score": float(np.mean((probability - y.astype(float)) ** 2))
            if len(probability)
            else None,
            "expected_calibration_error": expected_calibration_error(y, probability),
            "mean_score": float(probability.mean()) if len(probability) else None,
        }
    return metrics


def multiclass_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    labels = sorted(set(actual.astype(str)) | set(predicted.astype(str)))
    actual_string, predicted_string = actual.astype(str), predicted.astype(str)
    per_class: dict[str, Any] = {}
    f1_values = []
    recalls = []
    for label in labels:
        result = binary_metrics(actual_string, predicted_string, label)
        per_class[label] = {
            key: result[key]
            for key in (
                "precision",
                "true_positive_rate",
                "false_positive_rate",
                "f1",
            )
        }
        if result["f1"] is not None:
            f1_values.append(result["f1"])
        if result["true_positive_rate"] is not None:
            recalls.append(result["true_positive_rate"])
    return {
        "accuracy": float((actual_string == predicted_string).mean()),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "per_class": per_class,
    }


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    truth = pd.to_numeric(actual, errors="coerce")
    estimate = pd.to_numeric(predicted, errors="coerce")
    valid = truth.notna() & estimate.notna()
    truth, estimate = truth[valid].to_numpy(), estimate[valid].to_numpy()
    if not len(truth):
        return {}
    residual = estimate - truth
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "mean_signed_error": float(np.mean(residual)),
        "underprediction_rate": float(np.mean(residual < 0)),
        "overprediction_rate": float(np.mean(residual > 0)),
        "residual_q10": float(np.quantile(residual, 0.10)),
        "residual_q50": float(np.quantile(residual, 0.50)),
        "residual_q90": float(np.quantile(residual, 0.90)),
    }
