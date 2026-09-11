"""AutoGluon raw-input contract shared by API and explainers."""

import pandas as pd

from xai_core.utils import InputValidationError


def original_features(predictor) -> list:
    """Never use feature_metadata: it describes generated features, e.g. ngrams."""
    metadata = getattr(predictor, 'feature_metadata_in', None)
    if metadata is not None:
        expected = list(metadata.type_map_raw)
    elif callable(getattr(predictor, 'features', None)):
        expected = list(predictor.features(feature_stage='original'))
    else:
        raise RuntimeError('AutoGluon original input feature metadata is unavailable')
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError('AutoGluon original feature names are empty or duplicated')
    return expected


def align_autogluon_features(predictor, X: pd.DataFrame) -> pd.DataFrame:
    """Select exact raw columns in model order; never synthesize or encode inputs."""
    if X.columns.has_duplicates:
        raise InputValidationError('Duplicate feature column names are not allowed')
    expected = original_features(predictor)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise InputValidationError(f'Missing required AutoGluon raw input features: {missing}')
    return X.loc[:, expected].copy()


def text_columns(predictor, X: pd.DataFrame) -> list:
    """Original text special types take precedence over hints and a modest heuristic."""
    metadata = getattr(predictor, 'feature_metadata_in', None)
    specials = getattr(metadata, 'type_group_map_special', {}) or {}
    declared = specials.get('text', [])
    found = [c for c in X.columns if c in declared]
    for col in X.columns:
        if col in found:
            continue
        values = X[col].dropna().head(200)
        if values.empty or not values.map(lambda v: isinstance(v, str)).all():
            continue
        words = values.str.split().str.len().mean()
        hint = str(col).lower() in {'text', 'review', 'description', 'document', 'sentence', 'content'}
        if hint or (words >= 5 and values.str.len().mean() >= 30 and values.nunique() / len(values) >= 0.5):
            found.append(col)
    return found
