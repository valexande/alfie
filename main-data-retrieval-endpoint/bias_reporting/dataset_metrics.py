"""Dataset-level representation, missingness, and proxy diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalized_mutual_information(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype(str), right.astype(str), normalize=True)
    if table.empty:
        return 0.0
    joint = table.to_numpy()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    mutual_information = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    denominator = max(hx, hy)
    return 0.0 if denominator == 0 else float(mutual_information / denominator)


def _association_fallback(
    data: pd.DataFrame, target: pd.Series, feature_columns: list[str]
) -> dict[str, Any]:
    associations = {}
    excluded_text = []
    for column in feature_columns:
        feature = data[column]
        if feature.dtype == object and feature.astype(str).str.len().median() > 80:
            excluded_text.append(column)
            continue
        if pd.api.types.is_numeric_dtype(feature) and feature.nunique(dropna=True) > 10:
            try:
                prepared = pd.qcut(feature, q=10, duplicates="drop").astype(str)
            except ValueError:
                prepared = feature.astype(str)
        else:
            prepared = feature.fillna("<missing>").astype(str)
        associations[column] = _normalized_mutual_information(prepared, target)
    ordered = sorted(associations.items(), key=lambda item: item[1], reverse=True)
    maximum = ordered[0][1] if ordered else None
    return {
        "status": "complete",
        "method": "univariate_normalized_mutual_information",
        "maximum_proxy_association": maximum,
        "top_proxy_features": [
            {"feature": feature, "normalized_mutual_information": value}
            for feature, value in ordered[:10]
        ],
        "excluded_free_text_columns": excluded_text,
        "review_recommended": maximum is not None and maximum >= 0.20,
        "interpretation": (
            "This dependency-free fallback detects univariate associations. "
            "It can miss proxies encoded through combinations of features."
        ),
    }


def group_data_quality(
    frame: pd.DataFrame,
    sensitive_attributes: list[str],
    target_column: str | None,
    positive_label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in sensitive_attributes:
        rows = []
        for group_name, group in frame.groupby(attribute, dropna=False):
            row: dict[str, Any] = {
                "group": str(group_name),
                "sample_count": len(group),
                "sample_share": float(len(group) / len(frame)),
                "mean_missing_value_rate": float(group.isna().mean().mean()),
            }
            if target_column:
                valid = group[target_column].dropna().astype("string").str.strip()
                row["observed_positive_label_rate"] = (
                    float((valid == positive_label.strip()).mean()) if len(valid) else None
                )
            rows.append(row)
        result[attribute] = rows
    return result


def proxy_predictability(
    frame: pd.DataFrame,
    sensitive_attribute: str,
    excluded_columns: list[str],
) -> dict[str, Any]:
    """Estimate whether remaining tabular features predict a sensitive attribute."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        data = frame.dropna(subset=[sensitive_attribute]).copy()
        feature_columns = [
            column
            for column in data.columns
            if column not in set(excluded_columns + [sensitive_attribute])
        ]
        if not feature_columns:
            return {"status": "not_run", "reason": "No eligible feature columns remain"}
        return _association_fallback(
            data,
            data[sensitive_attribute].astype(str),
            feature_columns,
        )

    data = frame.dropna(subset=[sensitive_attribute]).copy()
    feature_columns = [
        column
        for column in data.columns
        if column not in set(excluded_columns + [sensitive_attribute])
    ]
    if not feature_columns:
        return {"status": "not_run", "reason": "No eligible feature columns remain"}
    target = data[sensitive_attribute].astype(str)
    counts = target.value_counts()
    if len(target) < 40 or len(counts) < 2 or counts.min() < 5:
        return {
            "status": "not_run",
            "reason": "At least 40 rows and 5 examples per sensitive group are required",
        }
    # Bound runtime and avoid accidentally feeding large raw media/text blobs.
    data = data.sample(min(len(data), 5000), random_state=42)
    target = data[sensitive_attribute].astype(str)
    features = data[feature_columns].copy()
    text_like = [
        column
        for column in features.select_dtypes(include=["object", "string"]).columns
        if features[column].astype(str).str.len().median() > 80
    ]
    features = features.drop(columns=text_like)
    if features.shape[1] == 0:
        return {"status": "not_run", "reason": "Only free-text/media columns remain"}
    numeric = list(features.select_dtypes(include=[np.number]).columns)
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
                    ]
                ),
                categorical,
            )
        )
    folds = min(5, int(target.value_counts().min()))
    model = Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    scores = cross_val_score(
        model,
        features,
        target,
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
    )
    majority_baseline = float(target.value_counts(normalize=True).max())
    return {
        "status": "complete",
        "balanced_accuracy": float(scores.mean()),
        "cross_validation_standard_deviation": float(scores.std()),
        "majority_class_baseline": majority_baseline,
        "features_tested": list(features.columns),
        "excluded_free_text_columns": text_like,
        "review_recommended": float(scores.mean()) >= 0.70,
        "interpretation": "High predictability indicates proxy information, not necessarily improper model use.",
    }
