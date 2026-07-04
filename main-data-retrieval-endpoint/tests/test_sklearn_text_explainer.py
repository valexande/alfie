import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xai_core.explainer_factory import ExplainerFactory
from xai_core.data_interpretability_service import DataInterpretabilityService
from xai_core.report_builder import ReportBuilder


def _text_pipeline():
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("tfidf", TfidfVectorizer(), "text"),
        ])),
        ("classifier", LogisticRegression(max_iter=200)),
    ])


def test_detects_and_explains_raw_text_pipeline():
    X = pd.DataFrame({"text": [
        "critical account risk was detected during the international transfer review",
        "critical payment risk was detected during the customer transaction review",
        "routine payment was approved after completing the standard customer review",
        "routine account was approved after completing the standard compliance review",
    ]})
    y = pd.Series([1, 1, 2, 2], name="labels")
    model = _text_pipeline().fit(X, y)

    assert ExplainerFactory.is_sklearn_text_pipeline(model)
    assert ExplainerFactory.detect_model_type(model) == "sklearn_text"

    explainer = ExplainerFactory.create(model, X, y)
    assert explainer.get_metrics()["accuracy"] == 1.0
    assert "SHAP Text Explanations" in explainer.generate_report("expert")
    assert "token-positive" in explainer.generate_report("expert")

    report = ReportBuilder(
        explainer,
        data_service=DataInterpretabilityService(pd.concat([X, y], axis=1)),
    ).build("expert")
    assert "Dataset Overview" in report
    assert "Text Data Profile" in report
    assert "MODEL EXPLAINABILITY" in report
    assert "SHAP Text Explanations" in report
    assert "LIME Text Explanations" in report
    assert "Token-Removal Sensitivity" in report
