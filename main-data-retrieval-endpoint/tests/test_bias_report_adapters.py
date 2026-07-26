"""Coverage for task and modality adapters."""

import pandas as pd

from bias_reporting.service import BiasReportService


def test_binary_scores_add_calibration_thresholds_and_confidence_interval():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "label": [1, 0] * 20,
            "prediction": [1, 0] * 10 + [0] * 20,
            "score": [0.9, 0.1] * 10 + [0.4, 0.2] * 10,
        }
    )
    report = BiasReportService(
        frame,
        ["group"],
        "label",
        "prediction",
        score_column="score",
        minimum_group_size=5,
        bootstrap_iterations=50,
    ).build()
    assert report["score_threshold_analysis"]["status"] == "complete"
    analysis = report["analyses"][0]
    assert analysis["uncertainty"]["demographic_parity_difference"]["ci_low"] is not None
    assert "brier_score" in analysis["groups"][0]["performance"]["score_metrics"]


def test_multiclass_adapter_reports_per_class_metrics():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "label": ["cat", "dog", "bird"] * 4,
            "prediction": ["cat", "dog", "bird", "cat", "cat", "bird"] * 2,
        }
    )
    report = BiasReportService(
        frame,
        ["group"],
        "label",
        "prediction",
        task_type="multiclass_classification",
        positive_label="cat",
        minimum_group_size=1,
        bootstrap_iterations=10,
    ).build()
    assert set(report["analyses"][0]["groups"][0]["performance"]["per_class"]) == {
        "bird",
        "cat",
        "dog",
    }


def test_regression_adapter_uses_error_disparities():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [10, 20, 30, 40] * 2,
            "prediction": [11, 19, 31, 39, 15, 25, 35, 45],
        }
    )
    report = BiasReportService(
        frame,
        ["group"],
        "target",
        "prediction",
        task_type="regression",
        minimum_group_size=1,
    ).build()
    disparities = report["analyses"][0]["disparities"]
    assert disparities["mae_difference"] == 4.0
    assert "demographic_parity_difference" not in disparities


def test_text_counterfactual_adapter():
    frame = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "pair": [1, 1, 2, 2],
            "prediction": [1, 0, 1, 1],
            "score": [0.8, 0.3, 0.7, 0.65],
        }
    )
    report = BiasReportService(
        frame,
        ["group"],
        prediction_column="prediction",
        modality="tabular_text",
        score_column="score",
        counterfactual_pair_column="pair",
        minimum_group_size=1,
        bootstrap_iterations=10,
    ).build()
    counterfactual = report["modality_analysis"]["text_counterfactual"]
    assert counterfactual["counterfactual_flip_rate"] == 0.5
    assert counterfactual["maximum_score_difference"] == 0.5


def test_image_detection_and_quality_context_adapter():
    frame = pd.DataFrame(
        {
            "skin_tone": ["light", "light", "dark", "dark"],
            "lighting": ["good", "low", "good", "low"],
            "detected": [1, 1, 1, 0],
            "iou": [0.9, 0.7, 0.6, 0.2],
        }
    )
    report = BiasReportService(
        frame,
        ["skin_tone"],
        prediction_column="detected",
        modality="image",
        context_attributes=["lighting"],
        vision_task="object_detection",
        detected_column="detected",
        iou_column="iou",
        minimum_group_size=1,
        bootstrap_iterations=10,
    ).build()
    assert report["modality_analysis"]["vision"]["detection_rate"] == 0.75
    assert any(item["attributes"] == ["lighting"] for item in report["analyses"])


def test_video_adapter_aggregates_subjects_and_temporal_flips():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "video": ["v1"] * 4 + ["v2"] * 4,
            "subject": ["s1"] * 4 + ["s2"] * 4,
            "timestamp": [1, 2, 3, 4] * 2,
            "label": [1] * 8,
            "prediction": [1, 0, 1, 1, 1, 1, 1, 1],
            "duration": [1] * 8,
            "delay": [0.2] * 4 + [0.6] * 4,
        }
    )
    report = BiasReportService(
        frame,
        ["group"],
        "label",
        "prediction",
        modality="video",
        video_id_column="video",
        subject_id_column="subject",
        timestamp_column="timestamp",
        duration_column="duration",
        detection_delay_column="delay",
        minimum_group_size=1,
        bootstrap_iterations=10,
    ).build()
    video = report["modality_analysis"]["video"]
    assert video["videos"] == 2
    assert video["subjects"] == 2
    assert video["temporal_prediction_flip_rate"] > 0
