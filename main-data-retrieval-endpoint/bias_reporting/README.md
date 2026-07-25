# Modality-aware AutoML bias reporting

`POST /bias-report` creates a JSON or standalone HTML screening report from an
AutoML evaluation CSV. The endpoint does not persist the uploaded data.

## Core input

| Form field | Required | Meaning |
|---|---:|---|
| `data_file` | yes | AutoML results CSV |
| `sensitive_attributes` | yes | Comma-separated governed metadata columns |
| `target_column` | task-dependent | Observed ground-truth value |
| `prediction_column` | task-dependent | Predicted label or regression value |
| `task_type` | no | `binary_classification`, `multiclass_classification`, or `regression` |
| `modality` | no | `tabular`, `tabular_text`, `image`, or `video` |
| `positive_label` | binary only | Value treated as the favorable/positive outcome |
| `score_column` | no | Positive-class probability in `[0, 1]` |
| `split_column` | no | Train/validation/test split indicator |
| `context_attributes` | no | Comma-separated quality strata such as lighting or blur |
| `minimum_group_size` | no | Groups below this size are excluded from disparity summaries |
| `bootstrap_iterations` | no | Resamples used for uncertainty intervals; default `500` |
| `response_format` | no | `json` or `html` |

Core analyses include representation, group missingness and label prevalence,
intersectional groups, proxy predictability, split-distribution shift, group
performance, disparity metrics, and bootstrap uncertainty.

Binary classification additionally supports ROC-AUC, Brier score, expected
calibration error, and fairness across decision thresholds when `score_column`
is provided. Multiclass reports per-class and macro metrics. Regression reports
MAE, RMSE, median error, signed error, residual quantiles, and over/under-
prediction rates.

## Tabular data with text

Supply:

```text
modality=tabular_text
counterfactual_pair_column=pair_id
```

Each `pair_id` should link semantically equivalent rows that differ only in the
identity reference being tested. The report computes counterfactual prediction
flip rate, mean score change, and maximum score change. Pair construction and
semantic validity remain the caller's responsibility.

## Images

Supply `modality=image` and one of:

- `vision_task=classification`: uses the core classification analysis.
- `vision_task=object_detection`: optionally supply `detected_column` and
  `iou_column`; use `average_precision_column` when AutoML provides per-sample
  or per-batch AP values.
- `vision_task=segmentation`: supply `iou_column`, `dice_column`, or both.
  `boundary_error_column` is also supported.

Use `context_attributes=lighting,blur,occlusion,pose` to obtain the same
disaggregated evaluation across image-quality conditions. Sensitive attributes
should come from appropriately governed dataset metadata, not unvalidated
inference from pixels.

## Video

Supply `modality=video` and:

| Field | Purpose |
|---|---|
| `video_id_column` | Required to identify correlated frames |
| `subject_id_column` | Subject-level count |
| `timestamp_column` | Temporal ordering and prediction instability |
| `duration_column` | Observation seconds for errors per minute |
| `detection_delay_column` | Mean and 90th-percentile time to detection |
| `track_lost_column` | Boolean tracking-loss indicator |

The report computes video/subject counts, temporal prediction flip rate during
stable labels, errors per minute, track-loss rate, and detection-delay
statistics. Video and vision diagnostics are also broken down by sensitive
attribute. Core group analysis still supplies outcome and performance
disparities.

## Example

```bash
curl -X POST http://localhost:8000/bias-report \
  -F data_file=@automl_results.csv \
  -F sensitive_attributes=gender,age_group \
  -F target_column=label \
  -F prediction_column=prediction \
  -F score_column=approved_probability \
  -F split_column=split \
  -F task_type=binary_classification \
  -F modality=tabular \
  -F positive_label=approved \
  -F response_format=json
```

## Interpretation

The output is a descriptive screening report, not proof of fairness,
discrimination, or causality. Thresholds are review triggers. Metric selection
must reflect the actual harm, decision context, affected population, and legal
requirements. Small groups, correlated observations, label errors, invalid
counterfactuals, and unmeasured confounders can materially change conclusions.
