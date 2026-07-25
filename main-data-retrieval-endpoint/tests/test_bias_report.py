"""Tests for the isolated AutoML bias reporting endpoint."""

import io
import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.app import app


def _csv() -> io.BytesIO:
    frame = pd.DataFrame(
        {
            "gender": ["A"] * 4 + ["B"] * 4,
            "region": ["north", "south"] * 4,
            "label": ["yes", "yes", "no", "no", "yes", "yes", "no", "no"],
            "prediction": ["yes", "yes", "no", "no", "no", "no", "no", "no"],
        }
    )
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def test_json_report_calculates_group_disparities():
    response = TestClient(app).post(
        "/bias-report",
        files={"data_file": ("results.csv", _csv(), "text/csv")},
        data={
            "sensitive_attributes": "gender",
            "target_column": "label",
            "prediction_column": "prediction",
            "positive_label": "yes",
            "minimum_group_size": "1",
        },
    )
    assert response.status_code == 200
    report = response.json()
    analysis = report["analyses"][0]
    assert report["report_type"] == "automl_bias_detection"
    assert analysis["disparities"]["demographic_parity_difference"] == 0.5
    assert analysis["disparities"]["disparate_impact_ratio"] == 0.0
    assert analysis["disparities"]["equal_opportunity_difference"] == 1.0


def test_multiple_attributes_add_intersectional_analysis():
    response = TestClient(app).post(
        "/bias-report",
        files={"data_file": ("results.csv", _csv(), "text/csv")},
        data={
            "sensitive_attributes": "gender,region",
            "prediction_column": "prediction",
            "positive_label": "yes",
            "minimum_group_size": "1",
        },
    )
    assert response.status_code == 200
    assert len(response.json()["analyses"]) == 3


def test_html_report():
    response = TestClient(app).post(
        "/bias-report",
        files={"data_file": ("results.csv", _csv(), "text/csv")},
        data={
            "sensitive_attributes": "gender",
            "prediction_column": "prediction",
            "positive_label": "yes",
            "minimum_group_size": "1",
            "response_format": "html",
        },
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AutoML Bias Detection Report" in response.text


def test_missing_sensitive_column_returns_400():
    response = TestClient(app).post(
        "/bias-report",
        files={"data_file": ("results.csv", _csv(), "text/csv")},
        data={"sensitive_attributes": "missing", "prediction_column": "prediction"},
    )
    assert response.status_code == 400
    assert "Columns not found" in response.json()["detail"]


def test_rows_with_missing_predictions_are_excluded():
    frame = pd.DataFrame(
        {"group": ["A", "A", "B"], "prediction": ["yes", None, "no"]}
    )
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    response = TestClient(app).post(
        "/bias-report",
        files={"data_file": ("results.csv", buffer, "text/csv")},
        data={
            "sensitive_attributes": "group",
            "prediction_column": "prediction",
            "positive_label": "yes",
            "minimum_group_size": "1",
        },
    )
    analysis = response.json()["analyses"][0]
    assert analysis["rows_analyzed"] == 2
    assert analysis["rows_excluded_missing_outcomes"] == 1
