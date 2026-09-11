"""Group-level bias indicators for binary classification AutoML outputs.

This module intentionally does not claim to determine whether a model is fair.
It reports observable group disparities that require human review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .modality_metrics import text_counterfactual_metrics, video_metrics, vision_metrics
from .dataset_metrics import group_data_quality, proxy_predictability
from .statistics import disparity_bootstrap_interval
from .task_metrics import binary_metrics, multiclass_metrics, regression_metrics


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _difference(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if len(valid) < 2 else float(max(valid) - min(valid))


def _ratio(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if len(valid) < 2 or max(valid) == 0:
        return None
    return float(min(valid) / max(valid))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    return value


class BiasReportService:
    """Create a descriptive, group-based bias report from an AutoML dataframe."""

    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_attributes: list[str],
        target_column: str | None = None,
        prediction_column: str | None = None,
        positive_label: str = "1",
        minimum_group_size: int = 20,
        include_intersections: bool = True,
        task_type: str = "binary_classification",
        modality: str = "tabular",
        score_column: str | None = None,
        split_column: str | None = None,
        context_attributes: list[str] | None = None,
        bootstrap_iterations: int = 500,
        counterfactual_pair_column: str | None = None,
        vision_task: str | None = None,
        iou_column: str | None = None,
        detected_column: str | None = None,
        dice_column: str | None = None,
        video_id_column: str | None = None,
        subject_id_column: str | None = None,
        timestamp_column: str | None = None,
        duration_column: str | None = None,
        detection_delay_column: str | None = None,
        average_precision_column: str | None = None,
        boundary_error_column: str | None = None,
        track_lost_column: str | None = None,
    ) -> None:
        self.data = data.copy()
        self.sensitive_attributes = list(dict.fromkeys(sensitive_attributes))
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.positive_label = str(positive_label)
        self.minimum_group_size = minimum_group_size
        self.include_intersections = include_intersections
        self.task_type = task_type
        self.modality = modality
        self.score_column = score_column
        self.split_column = split_column
        self.context_attributes = context_attributes or []
        self.bootstrap_iterations = bootstrap_iterations
        self.counterfactual_pair_column = counterfactual_pair_column
        self.vision_task = vision_task
        self.iou_column = iou_column
        self.detected_column = detected_column
        self.dice_column = dice_column
        self.video_id_column = video_id_column
        self.subject_id_column = subject_id_column
        self.timestamp_column = timestamp_column
        self.duration_column = duration_column
        self.detection_delay_column = detection_delay_column
        self.average_precision_column = average_precision_column
        self.boundary_error_column = boundary_error_column
        self.track_lost_column = track_lost_column
        self._validate()

    def _validate(self) -> None:
        if self.data.empty:
            raise ValueError("The uploaded dataset is empty.")
        if not self.sensitive_attributes:
            raise ValueError("At least one sensitive attribute is required.")
        requested = self.sensitive_attributes + self.context_attributes + [
            column
            for column in (
                self.target_column,
                self.prediction_column,
                self.score_column,
                self.split_column,
            )
            if column
        ]
        missing = [column for column in requested if column not in self.data.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. Available columns: {list(self.data.columns)}"
            )
        if not self.target_column and not self.prediction_column:
            raise ValueError(
                "Provide target_column, prediction_column, or both. A model-outcome "
                "column is required to calculate group disparities."
            )
        if self.minimum_group_size < 1:
            raise ValueError("minimum_group_size must be at least 1.")
        if self.task_type not in {"binary_classification", "multiclass_classification", "regression"}:
            raise ValueError(
                "task_type must be binary_classification, multiclass_classification, or regression."
            )
        if self.modality not in {"tabular", "tabular_text", "image", "video"}:
            raise ValueError("modality must be tabular, tabular_text, image, or video.")
        if self.target_column and not self.prediction_column and self.task_type == "regression":
            raise ValueError("Regression bias analysis requires prediction_column.")
        if self.bootstrap_iterations < 0 or self.bootstrap_iterations > 10000:
            raise ValueError("bootstrap_iterations must be between 0 and 10000.")

    def _is_positive(self, series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip() == self.positive_label.strip()

    def _attribute_sets(self) -> list[tuple[str, ...]]:
        auditable = list(dict.fromkeys(self.sensitive_attributes + self.context_attributes))
        sets = [(attribute,) for attribute in auditable]
        if self.include_intersections and len(self.sensitive_attributes) > 1:
            for size in range(2, len(self.sensitive_attributes) + 1):
                sets.extend(combinations(self.sensitive_attributes, size))
        return sets

    def _analyze(self, attributes: tuple[str, ...]) -> dict[str, Any]:
        sensitive_complete = self.data.dropna(subset=list(attributes)).copy()
        outcome_column = self.prediction_column or self.target_column
        assert outcome_column is not None
        required_outcomes = list(
            dict.fromkeys(
                column
                for column in (self.target_column, self.prediction_column)
                if column
            )
        )
        frame = sensitive_complete.dropna(subset=required_outcomes).copy()
        frame["_positive_outcome"] = self._is_positive(frame[outcome_column])
        if self.target_column:
            frame["_actual_positive"] = self._is_positive(frame[self.target_column])
        if self.prediction_column:
            frame["_predicted_positive"] = self._is_positive(frame[self.prediction_column])

        grouper: Any = attributes[0] if len(attributes) == 1 else list(attributes)
        groups: list[dict[str, Any]] = []
        eligible_selection_arrays: list[np.ndarray] = []
        for key, group in frame.groupby(grouper, dropna=False, sort=True):
            keys = (key,) if len(attributes) == 1 else tuple(key)
            count = len(group)
            item: dict[str, Any] = {
                "group": {attribute: str(value) for attribute, value in zip(attributes, keys)},
                "sample_count": count,
                "sample_share": _safe_div(count, len(frame)),
                "below_minimum_group_size": count < self.minimum_group_size,
            }
            if self.task_type != "regression":
                item["selection_rate"] = float(group["_positive_outcome"].mean())
            if self.target_column and self.prediction_column:
                scores = group[self.score_column] if self.score_column else None
                if self.task_type == "binary_classification":
                    item["performance"] = binary_metrics(
                        group[self.target_column],
                        group[self.prediction_column],
                        self.positive_label,
                        scores,
                    )
                elif self.task_type == "multiclass_classification":
                    item["performance"] = multiclass_metrics(
                        group[self.target_column], group[self.prediction_column]
                    )
                else:
                    item["performance"] = regression_metrics(
                        group[self.target_column], group[self.prediction_column]
                    )
            groups.append(item)
            if count >= self.minimum_group_size and self.task_type != "regression":
                eligible_selection_arrays.append(group["_positive_outcome"].astype(float).to_numpy())

        eligible = [group for group in groups if not group["below_minimum_group_size"]]
        disparities: dict[str, Any]
        uncertainty: dict[str, Any] = {}
        if self.task_type == "regression":
            mae_values = [group.get("performance", {}).get("mae") for group in eligible]
            signed_values = [
                group.get("performance", {}).get("mean_signed_error") for group in eligible
            ]
            disparities = {
                "mae_difference": _difference(mae_values),
                "mean_signed_error_range": _difference(signed_values),
            }
        else:
            selection_rates = [group["selection_rate"] for group in eligible]
            disparities = {
                "demographic_parity_difference": _difference(selection_rates),
                "disparate_impact_ratio": _ratio(selection_rates),
            }
            uncertainty["demographic_parity_difference"] = disparity_bootstrap_interval(
                eligible_selection_arrays,
                np.mean,
                iterations=self.bootstrap_iterations,
            )
        if (
            self.target_column
            and self.prediction_column
            and self.task_type == "binary_classification"
        ):
            tprs = [group["performance"]["true_positive_rate"] for group in eligible]
            fprs = [group["performance"]["false_positive_rate"] for group in eligible]
            disparities.update(
                {
                    "equal_opportunity_difference": _difference(tprs),
                    "false_positive_rate_difference": _difference(fprs),
                    "equalized_odds_difference": max(
                        [
                            value
                            for value in (_difference(tprs), _difference(fprs))
                            if value is not None
                        ],
                        default=None,
                    ),
                }
            )

        flags = []
        if self.task_type != "regression":
            ratio = disparities["disparate_impact_ratio"]
            parity_difference = disparities["demographic_parity_difference"]
            if ratio is not None and ratio < 0.8:
                flags.append("Disparate impact ratio is below the commonly used 0.80 screening threshold.")
            if parity_difference is not None and parity_difference > 0.10:
                flags.append("Selection-rate difference exceeds 0.10.")
        if any(group["below_minimum_group_size"] for group in groups):
            flags.append("One or more groups are too small for stable comparison.")
        if len(eligible) < 2:
            flags.append("Fewer than two groups meet the minimum size; disparity metrics are unavailable.")

        return {
            "attributes": list(attributes),
            "analysis_type": "intersectional" if len(attributes) > 1 else "single_attribute",
            "rows_analyzed": len(frame),
            "rows_excluded_missing_sensitive_values": len(self.data) - len(sensitive_complete),
            "rows_excluded_missing_outcomes": len(sensitive_complete) - len(frame),
            "groups": groups,
            "disparities": disparities,
            "uncertainty": uncertainty,
            "review_flags": flags,
        }

    def _split_diagnostics(self) -> dict[str, Any]:
        if not self.split_column:
            return {"status": "not_run", "reason": "split_column was not supplied"}
        table = pd.crosstab(
            self.data[self.split_column],
            self.data[self.sensitive_attributes[0]],
            normalize="index",
        )
        differences = {
            str(group): float(table[group].max() - table[group].min())
            for group in table.columns
        }
        return {
            "status": "complete",
            "split_column": self.split_column,
            "sensitive_attribute": self.sensitive_attributes[0],
            "group_shares_by_split": table.to_dict(orient="index"),
            "maximum_share_difference_by_group": differences,
        }

    def _threshold_analysis(self) -> dict[str, Any]:
        if self.task_type != "binary_classification" or not self.score_column:
            return {"status": "not_run", "reason": "Binary probability scores were not supplied"}
        frame = self.data.dropna(
            subset=[self.score_column, self.sensitive_attributes[0]]
        ).copy()
        frame["_score"] = pd.to_numeric(frame[self.score_column], errors="coerce")
        frame = frame.dropna(subset=["_score"])
        rows = []
        for threshold in np.linspace(0.1, 0.9, 9):
            rates = frame.groupby(self.sensitive_attributes[0])["_score"].apply(
                lambda values: float((values >= threshold).mean())
            )
            rows.append(
                {
                    "threshold": float(round(threshold, 2)),
                    "demographic_parity_difference": _difference(rates.tolist()),
                    "disparate_impact_ratio": _ratio(rates.tolist()),
                }
            )
        return {
            "status": "complete",
            "sensitive_attribute": self.sensitive_attributes[0],
            "thresholds": rows,
        }

    def build(self) -> dict[str, Any]:
        analyses = [self._analyze(attributes) for attributes in self._attribute_sets()]
        all_flags = [
            {"attributes": analysis["attributes"], "message": message}
            for analysis in analyses
            for message in analysis["review_flags"]
        ]
        proxy_exclusions = [
            column
            for column in (
                self.target_column,
                self.prediction_column,
                self.score_column,
                self.split_column,
                self.counterfactual_pair_column,
                self.video_id_column,
                self.subject_id_column,
                self.timestamp_column,
            )
            if column
        ] + self.sensitive_attributes
        report = {
            "schema_version": "1.0",
            "report_type": "automl_bias_detection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "review_recommended" if all_flags else "no_threshold_flags",
            "summary": {
                "rows": len(self.data),
                "sensitive_attributes": self.sensitive_attributes,
                "target_column": self.target_column,
                "prediction_column": self.prediction_column,
                "outcome_source": "prediction" if self.prediction_column else "observed_target",
                "positive_label": self.positive_label,
                "minimum_group_size": self.minimum_group_size,
                "task_type": self.task_type,
                "modality": self.modality,
                "score_column": self.score_column,
                "analyses_run": len(analyses),
                "flags_count": len(all_flags),
            },
            "flags": all_flags,
            "analyses": analyses,
            "dataset_diagnostics": {
                "split_distribution": self._split_diagnostics(),
                "group_data_quality": group_data_quality(
                    self.data,
                    self.sensitive_attributes,
                    self.target_column,
                    self.positive_label,
                ),
                "proxy_predictability": {
                    attribute: proxy_predictability(
                        self.data, attribute, proxy_exclusions
                    )
                    for attribute in self.sensitive_attributes
                },
            },
            "score_threshold_analysis": self._threshold_analysis(),
            "modality_analysis": {
                "text_counterfactual": (
                    text_counterfactual_metrics(
                        self.data,
                        self.counterfactual_pair_column,
                        self.prediction_column,
                        self.score_column,
                    )
                    if self.modality == "tabular_text"
                    else {"status": "not_applicable"}
                ),
                "vision": (
                    vision_metrics(
                        self.data,
                        self.vision_task,
                        self.iou_column,
                        self.detected_column,
                        self.dice_column,
                        self.sensitive_attributes,
                        self.average_precision_column,
                        self.boundary_error_column,
                    )
                    if self.modality == "image"
                    else {"status": "not_applicable"}
                ),
                "video": (
                    video_metrics(
                        self.data,
                        self.video_id_column,
                        self.subject_id_column,
                        self.timestamp_column,
                        self.duration_column,
                        self.target_column,
                        self.prediction_column,
                        self.detection_delay_column,
                        self.sensitive_attributes,
                        self.track_lost_column,
                    )
                    if self.modality == "video"
                    else {"status": "not_applicable"}
                ),
            },
            "methodology": {
                "purpose": "Descriptive screening for group-level disparities.",
                "thresholds": {
                    "disparate_impact_ratio": 0.8,
                    "demographic_parity_difference": 0.10,
                },
                "limitations": [
                    "Indicators are not proof of discrimination or fairness.",
                    "Results depend on the supplied sensitive attributes, labels, predictions, and sample size.",
                    "Small groups, label errors, and unmeasured confounders can make results misleading.",
                    "Bootstrap intervals quantify sampling variability, not causal uncertainty.",
                    "Counterfactual text pairs require semantic validation.",
                    "Image demographic attributes should come from appropriately governed metadata, not unvalidated inference.",
                    "Video frames from the same subject are correlated; subject-level review remains necessary.",
                    "Domain, legal, and stakeholder review is required before decisions are made.",
                ],
            },
        }
        return _clean(report)
