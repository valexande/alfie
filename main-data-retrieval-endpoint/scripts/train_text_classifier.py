#!/usr/bin/env python3
"""Train a raw-text sklearn pipeline and generate an XAI HTML report."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from xai_core.explainer_factory import ExplainerFactory
from xai_core.data_interpretability_service import DataInterpretabilityService
from xai_core.report_builder import ReportBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="CSV containing raw text and labels")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--target-column", default="labels")
    parser.add_argument("--model-output", type=Path, default=Path("text_classifier.joblib"))
    parser.add_argument("--report-output", type=Path, default=Path("text_xai_report.html"))
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    required = {args.text_column, args.target_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[[args.text_column]].copy()
    X[args.text_column] = X[args.text_column].fillna("").astype(str)
    y = df[args.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    preprocessing = ColumnTransformer([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=40000,
        ), args.text_column),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])

    print(f"Training on {len(X_train):,} rows; evaluating on {len(X_test):,} rows")
    pipeline.fit(X_train, y_train)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)

    explainer = ExplainerFactory.create(
        pipeline, X_test.reset_index(drop=True), y_test.reset_index(drop=True),
        max_samples=300,
    )
    data_service = DataInterpretabilityService(df)
    html = ReportBuilder(explainer, data_service=data_service).build(mode="expert")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(html, encoding="utf-8")

    metrics = explainer.get_metrics()
    print(f"Model:  {args.model_output.resolve()}")
    print(f"Report: {args.report_output.resolve()}")
    print(f"Accuracy:     {metrics.get('accuracy', 0):.2%}")
    print(f"Weighted F1:  {metrics.get('f1', 0):.4f}")
    print(f"Macro F1:     {metrics.get('macro_f1', 0):.4f}")
    print(f"Macro recall: {metrics.get('macro_recall', 0):.4f}")


if __name__ == "__main__":
    main()
