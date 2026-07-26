"""Diagnostics that require text, image, or video metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def text_counterfactual_metrics(
    frame: pd.DataFrame,
    pair_column: str | None,
    prediction_column: str | None,
    score_column: str | None,
) -> dict[str, Any]:
    if not pair_column:
        return {"status": "not_run", "reason": "counterfactual_pair_column was not supplied"}
    if pair_column not in frame:
        return {"status": "not_run", "reason": f"Column '{pair_column}' was not found"}
    if not prediction_column and not score_column:
        return {"status": "not_run", "reason": "Predictions or scores are required"}
    groups = [group for _, group in frame.dropna(subset=[pair_column]).groupby(pair_column) if len(group) >= 2]
    if not groups:
        return {"status": "not_run", "reason": "No counterfactual pairs were found"}
    flips, score_differences = [], []
    for group in groups:
        if prediction_column:
            flips.append(float(group[prediction_column].astype(str).nunique() > 1))
        if score_column:
            scores = pd.to_numeric(group[score_column], errors="coerce").dropna()
            if len(scores) >= 2:
                score_differences.append(float(scores.max() - scores.min()))
    return {
        "status": "complete",
        "pairs_analyzed": len(groups),
        "counterfactual_flip_rate": float(np.mean(flips)) if flips else None,
        "mean_absolute_score_difference": (
            float(np.mean(score_differences)) if score_differences else None
        ),
        "maximum_score_difference": max(score_differences, default=None),
        "interpretation": "Pairs should differ only in the tested identity reference and remain semantically valid.",
    }


def vision_metrics(
    frame: pd.DataFrame,
    vision_task: str | None,
    iou_column: str | None,
    detected_column: str | None,
    dice_column: str | None,
    sensitive_attributes: list[str] | None = None,
    average_precision_column: str | None = None,
    boundary_error_column: str | None = None,
) -> dict[str, Any]:
    if not vision_task:
        return {"status": "not_run", "reason": "vision_task was not supplied"}
    result: dict[str, Any] = {"status": "complete", "vision_task": vision_task}
    if vision_task == "classification":
        result["note"] = "Classification disparities are reported in the core group analyses."
    elif vision_task == "object_detection":
        if detected_column and detected_column in frame:
            detected = frame[detected_column].astype("string").str.lower().isin(["1", "true", "yes"])
            result["detection_rate"] = float(detected.mean())
            result["missed_detection_rate"] = float(1 - detected.mean())
        if iou_column and iou_column in frame:
            iou = pd.to_numeric(frame[iou_column], errors="coerce").dropna()
            result["mean_iou"] = float(iou.mean()) if len(iou) else None
            result["localization_success_at_iou_50"] = float((iou >= 0.5).mean()) if len(iou) else None
        if average_precision_column and average_precision_column in frame:
            values = pd.to_numeric(frame[average_precision_column], errors="coerce").dropna()
            result["mean_average_precision"] = float(values.mean()) if len(values) else None
        if len(result) == 2:
            return {"status": "not_run", "reason": "Detection outcome or IoU columns are required"}
    elif vision_task == "segmentation":
        found = False
        for name, column in (("mean_iou", iou_column), ("mean_dice", dice_column)):
            if column and column in frame:
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                result[name] = float(values.mean()) if len(values) else None
                found = True
        if boundary_error_column and boundary_error_column in frame:
            values = pd.to_numeric(frame[boundary_error_column], errors="coerce").dropna()
            result["mean_boundary_error"] = float(values.mean()) if len(values) else None
            found = True
        if not found:
            return {"status": "not_run", "reason": "IoU or Dice columns are required"}
    else:
        return {"status": "not_run", "reason": f"Unsupported vision_task '{vision_task}'"}
    group_metrics: dict[str, Any] = {}
    for attribute in sensitive_attributes or []:
        group_metrics[attribute] = {}
        for group_name, group in frame.dropna(subset=[attribute]).groupby(attribute):
            metrics: dict[str, Any] = {"sample_count": len(group)}
            if detected_column and detected_column in group:
                detected = group[detected_column].astype("string").str.lower().isin(
                    ["1", "true", "yes"]
                )
                metrics["detection_rate"] = float(detected.mean())
            for name, column in (
                ("mean_iou", iou_column),
                ("mean_dice", dice_column),
                ("mean_average_precision", average_precision_column),
                ("mean_boundary_error", boundary_error_column),
            ):
                if column and column in group:
                    values = pd.to_numeric(group[column], errors="coerce").dropna()
                    metrics[name] = float(values.mean()) if len(values) else None
            group_metrics[attribute][str(group_name)] = metrics
    result["group_metrics"] = group_metrics
    return result


def video_metrics(
    frame: pd.DataFrame,
    video_id_column: str | None,
    subject_id_column: str | None,
    timestamp_column: str | None,
    duration_column: str | None,
    target_column: str | None,
    prediction_column: str | None,
    delay_column: str | None,
    sensitive_attributes: list[str] | None = None,
    track_lost_column: str | None = None,
) -> dict[str, Any]:
    if not video_id_column:
        return {"status": "not_run", "reason": "video_id_column was not supplied"}
    if video_id_column not in frame:
        return {"status": "not_run", "reason": f"Column '{video_id_column}' was not found"}
    result: dict[str, Any] = {
        "status": "complete",
        "videos": int(frame[video_id_column].nunique()),
        "subjects": (
            int(frame[subject_id_column].nunique())
            if subject_id_column and subject_id_column in frame
            else None
        ),
    }
    if delay_column and delay_column in frame:
        delays = pd.to_numeric(frame[delay_column], errors="coerce").dropna()
        result["mean_time_to_detection"] = float(delays.mean()) if len(delays) else None
        result["p90_time_to_detection"] = float(delays.quantile(0.9)) if len(delays) else None
    if (
        timestamp_column
        and timestamp_column in frame
        and prediction_column
        and prediction_column in frame
    ):
        ordered = frame.sort_values([video_id_column, timestamp_column])
        prediction_changed = ordered.groupby(video_id_column)[prediction_column].transform(
            lambda values: values.astype(str).ne(values.astype(str).shift())
        )
        eligible = pd.Series(True, index=ordered.index)
        if target_column and target_column in ordered:
            eligible = ~ordered.groupby(video_id_column)[target_column].transform(
                lambda values: values.astype(str).ne(values.astype(str).shift())
            )
        # Exclude the first frame in each video.
        first = ordered.groupby(video_id_column).cumcount() == 0
        stable = eligible & ~first
        result["temporal_prediction_flip_rate"] = (
            float(prediction_changed[stable].mean()) if stable.any() else None
        )
    if duration_column and duration_column in frame and target_column and prediction_column:
        duration = pd.to_numeric(frame[duration_column], errors="coerce").dropna().sum()
        errors = (frame[target_column].astype(str) != frame[prediction_column].astype(str)).sum()
        result["errors_per_minute"] = float(errors / (duration / 60)) if duration > 0 else None
    if track_lost_column and track_lost_column in frame:
        lost = frame[track_lost_column].astype("string").str.lower().isin(["1", "true", "yes"])
        result["track_loss_rate"] = float(lost.mean())

    group_metrics: dict[str, Any] = {}
    for attribute in sensitive_attributes or []:
        group_metrics[attribute] = {}
        for group_name, group in frame.dropna(subset=[attribute]).groupby(attribute):
            metrics: dict[str, Any] = {
                "frames": len(group),
                "videos": int(group[video_id_column].nunique()),
            }
            if delay_column and delay_column in group:
                values = pd.to_numeric(group[delay_column], errors="coerce").dropna()
                metrics["mean_time_to_detection"] = float(values.mean()) if len(values) else None
            if track_lost_column and track_lost_column in group:
                lost = group[track_lost_column].astype("string").str.lower().isin(
                    ["1", "true", "yes"]
                )
                metrics["track_loss_rate"] = float(lost.mean())
            if duration_column and duration_column in group and target_column and prediction_column:
                duration = pd.to_numeric(group[duration_column], errors="coerce").dropna().sum()
                errors = (
                    group[target_column].astype(str) != group[prediction_column].astype(str)
                ).sum()
                metrics["errors_per_minute"] = (
                    float(errors / (duration / 60)) if duration > 0 else None
                )
            group_metrics[attribute][str(group_name)] = metrics
    result["group_metrics"] = group_metrics
    return result
