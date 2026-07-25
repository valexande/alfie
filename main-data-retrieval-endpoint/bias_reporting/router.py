"""FastAPI endpoint for generating a bias detection report."""

from __future__ import annotations

import io
from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .html_report import render_html
from .service import BiasReportService

router = APIRouter(tags=["Bias Reporting"])


@router.post(
    "/bias-report",
    summary="Generate group-disparity indicators from AutoML results",
    responses={
        200: {"description": "Structured JSON or standalone HTML bias report"},
        400: {"description": "Invalid data or configuration"},
    },
)
async def create_bias_report(
    data_file: UploadFile = File(..., description="AutoML results in CSV format"),
    sensitive_attributes: str = Form(
        ..., description="Comma-separated sensitive attribute column names"
    ),
    target_column: str | None = Form(
        None, description="Observed/ground-truth label column"
    ),
    prediction_column: str | None = Form(
        None, description="AutoML predicted-label column"
    ),
    positive_label: str = Form("1", description="Value treated as the positive outcome"),
    minimum_group_size: int = Form(20, ge=1, le=100000),
    include_intersections: bool = Form(True),
    task_type: Literal[
        "binary_classification", "multiclass_classification", "regression"
    ] = Form("binary_classification"),
    modality: Literal["tabular", "tabular_text", "image", "video"] = Form("tabular"),
    score_column: str | None = Form(
        None, description="Positive-class probability for calibration and threshold analysis"
    ),
    split_column: str | None = Form(
        None, description="Train/validation/test split column for representation-shift checks"
    ),
    context_attributes: str = Form(
        "",
        description="Optional comma-separated quality/context columns such as lighting or blur",
    ),
    bootstrap_iterations: int = Form(500, ge=0, le=10000),
    counterfactual_pair_column: str | None = Form(
        None, description="Identifier linking original and identity-swapped text rows"
    ),
    vision_task: Literal["classification", "object_detection", "segmentation"] | None = Form(None),
    iou_column: str | None = Form(None),
    detected_column: str | None = Form(None),
    dice_column: str | None = Form(None),
    average_precision_column: str | None = Form(None),
    boundary_error_column: str | None = Form(None),
    video_id_column: str | None = Form(None),
    subject_id_column: str | None = Form(None),
    timestamp_column: str | None = Form(None),
    duration_column: str | None = Form(None, description="Observation duration in seconds"),
    detection_delay_column: str | None = Form(None),
    track_lost_column: str | None = Form(None),
    response_format: Literal["json", "html"] = Form("json"),
):
    """Generate descriptive fairness indicators; no sensitive data is persisted."""
    try:
        content = await data_file.read()
        if not content:
            raise ValueError("The uploaded CSV file is empty.")
        frame = pd.read_csv(io.BytesIO(content))
        attributes = [
            item.strip() for item in sensitive_attributes.split(",") if item.strip()
        ]
        contexts = [item.strip() for item in context_attributes.split(",") if item.strip()]
        report = BiasReportService(
            data=frame,
            sensitive_attributes=attributes,
            target_column=target_column or None,
            prediction_column=prediction_column or None,
            positive_label=positive_label,
            minimum_group_size=minimum_group_size,
            include_intersections=include_intersections,
            task_type=task_type,
            modality=modality,
            score_column=score_column or None,
            split_column=split_column or None,
            context_attributes=contexts,
            bootstrap_iterations=bootstrap_iterations,
            counterfactual_pair_column=counterfactual_pair_column or None,
            vision_task=vision_task,
            iou_column=iou_column or None,
            detected_column=detected_column or None,
            dice_column=dice_column or None,
            video_id_column=video_id_column or None,
            subject_id_column=subject_id_column or None,
            timestamp_column=timestamp_column or None,
            duration_column=duration_column or None,
            detection_delay_column=detection_delay_column or None,
            average_precision_column=average_precision_column or None,
            boundary_error_column=boundary_error_column or None,
            track_lost_column=track_lost_column or None,
        ).build()
        if response_format == "html":
            return HTMLResponse(render_html(report))
        return JSONResponse(report)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
