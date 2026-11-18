"""Chart data in the JSON report output."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xai_core.explainer_factory import ExplainerFactory
from xai_core.report_builder import ReportBuilder


def _charts(model, X, y):
    explainer = ExplainerFactory.create(model, X, y, max_samples=100)
    data = ReportBuilder(explainer).build_json()
    json.dumps(data, allow_nan=False)  # must be strict JSON
    return data["charts"]


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return X, y


def test_multiclass_chart_data(iris):
    X, y = iris
    charts = _charts(RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y), X, y)

    assert set(charts) == {"confusion_matrix", "roc_curve", "shap_summary", "pca"}
    cm = charts["confusion_matrix"]
    assert cm["labels"] == [0, 1, 2]
    assert np.array(cm["matrix"]).sum() == len(y)

    roc = charts["roc_curve"]
    assert roc["kind"] == "micro_average"
    assert len(roc["fpr"]) == len(roc["tpr"]) and 0.5 < roc["auc"] <= 1.0

    shap_rows = charts["shap_summary"]["features"]
    assert {row["feature"] for row in shap_rows} == set(X.columns)
    assert set(shap_rows[0]["per_class"]) == {"0", "1", "2"}
    assert shap_rows[0]["mean_abs_shap"] == pytest.approx(sum(shap_rows[0]["per_class"].values()))
    assert [r["mean_abs_shap"] for r in shap_rows] == sorted((r["mean_abs_shap"] for r in shap_rows), reverse=True)

    pca = charts["pca"]
    assert pca["cumulative_variance_ratio"][-1] == pytest.approx(1.0)
    assert pca["rows_used"] == len(X)


def test_binary_roc_uses_positive_class(iris):
    X, y = iris
    X, y = X[y > 0], y[y > 0].map({1: "versicolor", 2: "virginica"})
    model = LogisticRegression(max_iter=500).fit(X, y)
    roc = _charts(model, X, y)["roc_curve"]
    assert roc["kind"] == "binary"
    assert roc["positive_class"] == "virginica"


def test_regression_chart_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    charts = _charts(RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y), X, y)

    assert "confusion_matrix" not in charts and "roc_curve" not in charts
    residuals = charts["residuals"]
    assert sum(residuals["histogram"]["counts"]) == len(y)
    assert len(residuals["histogram"]["bin_edges"]) == 31
    assert residuals["predicted_vs_actual"]["points"] == len(y)
    assert len(charts["shap_summary"]["features"]) == X.shape[1]
    assert charts["shap_summary"]["samples_explained"] == 100  # max_samples
    assert "per_class" not in charts["shap_summary"]["features"][0]


def test_text_model_skips_raw_feature_charts():
    X = pd.DataFrame({"text": ["urgent risk found", "urgent risk detected",
                               "routine payment ok", "routine payment approved"]})
    y = pd.Series([1, 1, 0, 0])
    model = Pipeline([
        ("prep", ColumnTransformer([("tfidf", TfidfVectorizer(), "text")])),
        ("clf", LogisticRegression()),
    ]).fit(X, y)
    charts = _charts(model, X, y)
    assert "shap_summary" not in charts and "pca" not in charts
    assert "confusion_matrix" in charts


def test_shap_computed_once_for_html_and_json(iris, monkeypatch):
    X, y = iris
    explainer = ExplainerFactory.create(RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y), X, y)
    calls = []
    original = explainer.get_shap_values
    monkeypatch.setattr(explainer, "get_shap_values", lambda *a, **k: calls.append(1) or original(*a, **k))
    ReportBuilder(explainer).build_with_json("expert")
    assert len(calls) == 1
