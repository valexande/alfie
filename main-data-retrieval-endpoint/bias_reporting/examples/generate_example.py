"""Generate the checked example reports from the sample AutoML results."""

import json
from pathlib import Path

import pandas as pd

from bias_reporting.html_report import render_html
from bias_reporting.service import BiasReportService


example_dir = Path(__file__).resolve().parent
data = pd.read_csv(example_dir / "automl_results.csv")
report = BiasReportService(
    data=data,
    sensitive_attributes=["gender", "age_group"],
    target_column="label",
    prediction_column="prediction",
    positive_label="approved",
    minimum_group_size=4,
    include_intersections=True,
).build()

(example_dir / "sample_bias_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
(example_dir / "sample_bias_report.html").write_text(
    render_html(report), encoding="utf-8"
)
