"""Generate a comprehensive synthetic AutoML bias-report example."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bias_reporting.html_report import render_html
from bias_reporting.service import BiasReportService


rng = np.random.default_rng(20260725)
example_dir = Path(__file__).resolve().parent
rows = []

# Every case has a counterfactual pair. The business features and ground truth
# remain constant while the identity term changes.
for pair_id in range(180):
    age_group = rng.choice(["18-34", "35-54", "55+"], p=[0.38, 0.42, 0.20])
    region = rng.choice(["urban", "rural"], p=[0.68, 0.32])
    split = rng.choice(["train", "validation", "test"], p=[0.58, 0.17, 0.25])
    experience = int(rng.integers(1, 26))
    income = float(
        26000
        + experience * 2100
        + (5500 if region == "urban" else 0)
        + rng.normal(0, 5500)
    )
    debt_ratio = float(np.clip(rng.beta(2.2, 4.2), 0.02, 0.95))
    true_logit = (
        -1.25
        + 0.055 * experience
        + 0.000012 * income
        - 2.1 * debt_ratio
        + (0.22 if age_group == "35-54" else 0)
    )
    true_probability = 1 / (1 + np.exp(-true_logit))
    label = "approved" if rng.random() < true_probability else "rejected"

    for gender in ("Female", "Male"):
        # Deliberate synthetic disparity: the score changes even though the
        # paired applicant's legitimate attributes and label are identical.
        identity_shift = 0.13 if gender == "Male" else -0.04
        age_shift = -0.07 if age_group == "55+" else 0
        noise = rng.normal(0, 0.035)
        score = float(np.clip(true_probability + identity_shift + age_shift + noise, 0.01, 0.99))
        prediction = "approved" if score >= 0.50 else "rejected"
        identity_word = "man" if gender == "Male" else "woman"
        rows.append(
            {
                "pair_id": f"case-{pair_id:03d}",
                "gender": gender,
                "age_group": age_group,
                "region": region,
                "split": split,
                "years_experience": experience,
                "income": round(income, 2),
                "debt_ratio": round(debt_ratio, 4),
                "application_text": (
                    f"The applicant is a {identity_word} with {experience} years "
                    "of experience requesting a standard account review."
                ),
                "label": label,
                "prediction": prediction,
                "approved_probability": round(score, 5),
            }
        )

data = pd.DataFrame(rows)
csv_path = example_dir / "comprehensive_automl_results.csv"
data.to_csv(csv_path, index=False)

report = BiasReportService(
    data=data,
    sensitive_attributes=["gender", "age_group"],
    target_column="label",
    prediction_column="prediction",
    positive_label="approved",
    minimum_group_size=25,
    include_intersections=True,
    task_type="binary_classification",
    modality="tabular_text",
    score_column="approved_probability",
    split_column="split",
    context_attributes=["region"],
    bootstrap_iterations=1000,
    counterfactual_pair_column="pair_id",
).build()

(example_dir / "comprehensive_bias_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
(example_dir / "comprehensive_bias_report.html").write_text(
    render_html(report), encoding="utf-8"
)
