"""Small statistical helpers used by all bias-report adapters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    iterations: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Return a deterministic percentile bootstrap interval."""
    clean = np.asarray(values)
    clean = clean[~np.asarray([value is None for value in clean])]
    if len(clean) == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "samples": 0}
    estimate = float(statistic(clean))
    if len(clean) < 2 or iterations < 2:
        return {
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
            "samples": int(len(clean)),
        }
    rng = np.random.default_rng(seed)
    estimates = np.array(
        [statistic(clean[rng.integers(0, len(clean), len(clean))]) for _ in range(iterations)],
        dtype=float,
    )
    alpha = (1 - confidence) / 2
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1 - alpha)),
        "samples": int(len(clean)),
    }


def disparity_bootstrap_interval(
    group_values: list[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    iterations: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap a max-minus-min group disparity."""
    usable = [np.asarray(values) for values in group_values if len(values)]
    if len(usable) < 2:
        return {"estimate": None, "ci_low": None, "ci_high": None, "iterations": 0}

    def disparity(groups: list[np.ndarray]) -> float:
        estimates = [float(statistic(group)) for group in groups]
        return max(estimates) - min(estimates)

    estimate = disparity(usable)
    if iterations < 2:
        return {"estimate": estimate, "ci_low": None, "ci_high": None, "iterations": 0}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        resampled = [
            group[rng.integers(0, len(group), len(group))] for group in usable
        ]
        samples.append(disparity(resampled))
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1 - alpha)),
        "iterations": iterations,
    }


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute binary ROC-AUC using average ranks, without extra dependencies."""
    y = np.asarray(y_true, dtype=bool)
    score = np.asarray(scores, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float | None:
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(scores, dtype=float)
    if not len(y):
        return None
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (score >= edges[index]) & (
            score <= edges[index + 1] if upper_inclusive else score < edges[index + 1]
        )
        if mask.any():
            total += float(mask.mean()) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return float(total)
