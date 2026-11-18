"""
Numbers behind the report's charts, for the JSON output.

Each builder reuses the calculation the matching chart uses, so the JSON and the
HTML report always agree. Images (Grad-CAM, embeddings, galleries) are not covered.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix

from xai_core.visualizations.performance_plots import coerce_class_predictions, roc_curve_points
from xai_core.visualizations.embedding_plots import fit_pca

RESIDUAL_BINS = 30          # same bin count as the residual histogram
MAX_SCATTER_POINTS = 1000   # predicted-vs-actual points kept in the JSON


def _raw_feature_charts(metrics: Dict) -> bool:
    # Text and image models flag that SHAP/PCA on raw columns is not a valid explanation
    return not metrics.get('skip_shap_section')


def _confusion_matrix(explainer, metrics: Dict) -> Optional[Dict]:
    if not explainer.is_classification or 'confusion_matrix' in metrics:
        return None  # already in metrics for models that compute their own
    classes = explainer.classes
    y_pred = coerce_class_predictions(explainer.get_predictions(), classes)
    return {
        'labels': classes,
        'matrix': confusion_matrix(explainer.y, y_pred, labels=classes),
        'rows': 'actual',
        'columns': 'predicted',
    }


def _roc_curve(explainer, metrics: Dict) -> Optional[Dict]:
    if not explainer.is_classification:
        return None
    y_proba = explainer.get_prediction_probabilities()
    if y_proba is None:
        return None
    return roc_curve_points(explainer.y, np.asarray(y_proba), explainer.classes)


def _residuals(explainer, metrics: Dict) -> Optional[Dict]:
    if not explainer.is_regression:
        return None
    actual = np.asarray(explainer.y, dtype=float)
    predicted = np.asarray(explainer.get_predictions(), dtype=float)
    residuals = actual - predicted
    counts, edges = np.histogram(residuals, bins=RESIDUAL_BINS)
    keep = np.arange(len(actual))
    if len(keep) > MAX_SCATTER_POINTS:
        keep = np.sort(np.random.default_rng(0).choice(keep, MAX_SCATTER_POINTS, replace=False))
    return {
        'summary': {
            'mean': residuals.mean(), 'std': residuals.std(),
            'min': residuals.min(), 'max': residuals.max(),
            'mean_absolute': np.abs(residuals).mean(),
        },
        'histogram': {'counts': counts, 'bin_edges': edges},
        'predicted_vs_actual': {
            'predicted': predicted[keep], 'actual': actual[keep],
            'points': len(keep), 'total_rows': len(actual),
        },
    }


def _shap_summary(explainer, metrics: Dict) -> Optional[Dict]:
    if not _raw_feature_charts(metrics) or not hasattr(explainer, 'get_shap_sample'):
        return None
    X_sample, values = explainer.get_shap_sample()
    if values is None:
        return None
    if isinstance(values, list):  # one array per class
        values = np.stack([np.asarray(v) for v in values], axis=-1)
    values = np.abs(np.asarray(values, dtype=float))
    names = list(X_sample.columns)
    if values.shape[1] != len(names):
        names = [f'feature_{i}' for i in range(values.shape[1])]

    per_feature = values.mean(axis=0)  # (features,) or (features, classes)
    classes = explainer.classes if values.ndim == 3 else None
    if classes is not None and len(classes) != per_feature.shape[1]:
        classes = [f'output_{i}' for i in range(per_feature.shape[1])]
    rows = []
    for i, name in enumerate(names):
        row = {'feature': name}
        if values.ndim == 3:
            # Same total as SHAP's stacked multiclass bar chart: sum over classes
            row['mean_abs_shap'] = per_feature[i].sum()
            row['per_class'] = dict(zip(map(str, classes), per_feature[i]))
        else:
            row['mean_abs_shap'] = per_feature[i]
        rows.append(row)
    rows.sort(key=lambda r: r['mean_abs_shap'], reverse=True)
    # Some explainers explain only part of the sample (AutoGluon: first 50 rows)
    return {'samples_explained': values.shape[0], 'features': rows}


def _pca(explainer, metrics: Dict) -> Optional[Dict]:
    if not _raw_feature_charts(metrics):
        return None
    fitted = fit_pca(explainer.X)
    if fitted is None:
        return None
    pca, _, X_clean = fitted
    ratio = pca.explained_variance_ratio_
    return {
        'explained_variance_ratio': ratio,
        'cumulative_variance_ratio': np.cumsum(ratio),
        'rows_used': len(X_clean),
    }


_BUILDERS: Dict[str, Callable] = {
    'confusion_matrix': _confusion_matrix,
    'roc_curve': _roc_curve,
    'residuals': _residuals,
    'shap_summary': _shap_summary,
    'pca': _pca,
}


def build_chart_data(explainer: Any, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Chart data for every chart that applies to this model. A chart whose data
    cannot be computed is left out, as the report leaves out its image."""
    if not hasattr(explainer, 'get_predictions') or metrics.get('model_type') == 'pytorch_vision':
        return {}  # legacy service / vision galleries: no reusable chart data
    charts = {}
    for name, builder in _BUILDERS.items():
        try:
            value = builder(explainer, metrics)
        except Exception as exc:
            print(f"Chart data '{name}' failed: {exc}")
            continue
        if value is not None:
            charts[name] = value
    return charts
