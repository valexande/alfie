#!/usr/bin/env python3
"""
AutoGluon explainability report generator.

Loads an AutoGluon TabularPredictor and generates a comprehensive HTML report with:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- ROC curve (binary or multiclass micro-average)
- PCA visualization (variance + 2D scatter)
- SHAP feature importance (optional)

Usage:
  python ag_explain.py \
    --predictor /path/to/predictor \
    --data /path/to/data.csv \
    --target-col target_column \
    --output report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_predictor(predictor_path: Path):
    """Load AutoGluon predictor from directory or pickle file."""
    from autogluon.tabular import TabularPredictor

    if predictor_path.is_dir():
        print(f"Loading predictor from directory: {predictor_path}")
        return TabularPredictor.load(str(predictor_path), require_py_version_match=False)

    # Fallback to pickle
    print(f"Loading predictor from pickle: {predictor_path}")
    import joblib
    try:
        return joblib.load(predictor_path)
    except Exception:
        import pickle
        with predictor_path.open("rb") as f:
            return pickle.load(f)


def validate_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target, ensuring target column exists."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    print(f"Data shape: {X.shape}, Target classes: {sorted(y.unique())}")
    return X, y


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_confusion_matrix(y_true, y_pred, labels) -> str:
    """Generate confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_roc_curve(y_true, y_proba, classes) -> Optional[str]:
    """Generate ROC curve (binary or multiclass micro-average)."""
    try:
        if len(classes) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
            auc = roc_auc_score(y_true, y_proba[:, 1])

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()

            return fig_to_base64(fig)

        else:
            # Multiclass: micro-average
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] <= 1:
                return None

            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            auc = roc_auc_score(y_bin, y_proba, average="micro", multi_class="ovr")

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, linewidth=2, label=f"Micro-avg AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curve (Micro-Average)", fontsize=14, fontweight="bold")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()

            return fig_to_base64(fig)

    except Exception as e:
        print(f"Warning: Could not generate ROC curve: {e}")
        return None


def plot_pca_analysis(X: pd.DataFrame, y: pd.Series) -> tuple[Optional[str], Optional[str]]:
    """Generate PCA variance and scatter plots for numeric features."""
    # Select numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print(f"Warning: Need at least 2 numeric features for PCA, found {len(numeric_cols)}")
        return None, None

    X_num = X[numeric_cols].dropna()
    if len(X_num) < 3:
        print("Warning: Insufficient data rows for PCA after dropping NaNs")
        return None, None

    # Fit PCA
    n_components = min(10, len(numeric_cols))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_num)
    var_ratio = pca.explained_variance_ratio_

    # Variance bar plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("PCA Explained Variance", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(1, len(var_ratio) + 1))

    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, var_ratio)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)

    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    variance_plot = fig_to_base64(fig1)

    # 2D scatter plot (PC1 vs PC2)
    scatter_plot = None
    if components.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        # Get aligned labels
        y_aligned = y.loc[X_num.index]
        class_labels, class_ids = pd.factorize(y_aligned)

        scatter = ax2.scatter(components[:, 0], components[:, 1],
                            c=class_labels, cmap="viridis", alpha=0.6,
                            edgecolors='k', linewidth=0.5, s=50)

        ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
        ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
        ax2.set_title("PCA Projection (PC1 vs PC2)", fontsize=14, fontweight="bold")

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=scatter.cmap(scatter.norm(i)),
                            markersize=8, label=str(label))
                  for i, label in enumerate(class_ids)]
        ax2.legend(handles=handles, title="Class", loc="best")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        scatter_plot = fig_to_base64(fig2)

    return variance_plot, scatter_plot


def plot_shap_importance(predictor, X: pd.DataFrame, max_samples: int = 300) -> Optional[str]:
    """Generate SHAP feature importance plot."""
    if not SHAP_AVAILABLE:
        print("SHAP not installed, skipping feature importance plot")
        return None

    try:
        print("Generating SHAP feature importance...")

        # Sample data if needed
        X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)
        print(f"Using {len(X_sample)} samples for SHAP analysis")

        # Create prediction wrapper for AutoGluon
        def predict_fn(data):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X_sample.columns)
            return predictor.predict_proba(data).values

        # Try TreeExplainer first (faster), fallback to KernelExplainer
        explainer = None
        shap_values = None

        try:
            # Attempt TreeExplainer (works with tree-based models)
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X_sample)
            print("Using TreeExplainer")
        except Exception as e:
            print(f"TreeExplainer unavailable: {e}")
            print("Falling back to KernelExplainer (this may take a few minutes)...")

            # Use KernelExplainer (works with any model but slower)
            background = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
            explainer = shap.KernelExplainer(predict_fn, background)
            X_sample = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
            print("Using KernelExplainer")

        # Handle multiclass output (list of arrays)
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = shap_values[0]  # Use first class for visualization

        # Create summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                         plot_type="bar",
                         max_display=20,
                         show=False)
        plt.title("Feature Importance (SHAP)", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        return fig_to_base64(fig)

    except Exception as e:
        print(f"Warning: SHAP analysis failed: {e}")
        return None


def build_html_report(model_name: str,
                     metrics: dict,
                     confusion_matrix_img: Optional[str],
                     roc_curve_img: Optional[str],
                     pca_variance_img: Optional[str],
                     pca_scatter_img: Optional[str],
                     shap_importance_img: Optional[str]) -> str:
    """Build complete HTML report with all visualizations."""

    def image_section(img_b64: str, title: str) -> str:
        return f'''
        <div class="section">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{img_b64}" alt="{title}"/>
        </div>
        '''

    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoGluon Explainability Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.95; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .section h3 {{
            color: #764ba2;
            font-size: 1.4em;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:hover {{ background: #f8f9fa; }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            display: block;
            margin: 15px auto;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-top: 5px;
        }}
        .info-box {{
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .info-box h4 {{
            color: #1976D2;
            margin-bottom: 10px;
        }}
        .info-box ul {{
            margin-left: 20px;
        }}
        .info-box li {{
            margin: 8px 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AutoGluon Explainability Report</h1>
            <p>Model: <strong>{model_name}</strong></p>
        </div>

        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Accuracy</div>
                    <div class="value">{metrics.get('accuracy', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Macro F1</div>
                    <div class="value">{metrics.get('f1_macro', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Weighted F1</div>
                    <div class="value">{metrics.get('f1_weighted', 'N/A')}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Classification Report</h2>
            {metrics.get('report_html', '<em>No report available</em>')}
        </div>
'''

    # Add visualization sections
    if confusion_matrix_img:
        html += image_section(confusion_matrix_img, "Confusion Matrix")

    if roc_curve_img:
        html += image_section(roc_curve_img, "ROC Curve")

    if shap_importance_img:
        html += image_section(shap_importance_img, "Feature Importance (SHAP)")

    if pca_variance_img:
        html += image_section(pca_variance_img, "PCA Explained Variance")

    if pca_scatter_img:
        html += image_section(pca_scatter_img, "PCA Projection")

    # Add explanation section
    html += '''
        <div class="section">
            <h2>Understanding the Metrics</h2>
            <div class="info-box">
                <h4>Classification Metrics</h4>
                <ul>
                    <li><strong>Accuracy</strong>: Overall fraction of correct predictions</li>
                    <li><strong>Precision</strong>: Of predicted positives, how many are actually correct</li>
                    <li><strong>Recall</strong>: Of actual positives, how many were correctly identified</li>
                    <li><strong>F1 Score</strong>: Harmonic mean of precision and recall
                        <ul>
                            <li><em>Macro</em>: Unweighted average across all classes</li>
                            <li><em>Weighted</em>: Class-frequency weighted average</li>
                        </ul>
                    </li>
                </ul>
            </div>

            <div class="info-box">
                <h4>Visualizations</h4>
                <ul>
                    <li><strong>Confusion Matrix</strong>: Shows prediction vs actual counts. Diagonal = correct predictions, off-diagonal = errors</li>
                    <li><strong>ROC Curve</strong>: Trade-off between true positive rate and false positive rate. AUC closer to 1.0 indicates better model separability</li>
                    <li><strong>SHAP Feature Importance</strong>: Shows which features have the most impact on predictions (based on Shapley values)</li>
                    <li><strong>PCA Variance</strong>: How much information each principal component captures</li>
                    <li><strong>PCA Projection</strong>: Data visualization in 2D space to reveal patterns and clustering</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate AutoGluon model explainability report",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictor", required=True,
                       help="Path to AutoGluon predictor directory or pickle file")
    parser.add_argument("--data", required=True,
                       help="Path to CSV data file")
    parser.add_argument("--target-col", required=True,
                       help="Name of the target column in CSV")
    parser.add_argument("--output", required=True,
                       help="Path to save HTML report")
    parser.add_argument("--max-shap-samples", type=int, default=300,
                       help="Maximum samples for SHAP analysis (default: 300)")

    args = parser.parse_args()

    # Resolve paths
    predictor_path = Path(args.predictor).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    # Validate inputs
    if not predictor_path.exists():
        print(f"Error: Predictor path does not exist: {predictor_path}")
        return 1

    if not data_path.exists():
        print(f"Error: Data file does not exist: {data_path}")
        return 1

    print("=" * 60)
    print("AutoGluon Explainability Report Generator")
    print("=" * 60)

    # Load predictor
    predictor = load_predictor(predictor_path)

    # Load and validate data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    X, y = validate_data(df, args.target_col)

    # Generate predictions
    print("\nGenerating predictions...")
    # Use only LightGBM model to avoid FastAI path issues on Windows
    try:
        y_pred = predictor.predict(X, model='LightGBMXT')  # Use only LightGBM model
        proba_df = predictor.predict_proba(X, model='LightGBMXT')
        print("Using LightGBMXT model for predictions")
    except Exception as e:
        print(f"Warning: Could not use LightGBMXT model ({e}), trying default...")
        y_pred = predictor.predict(X)
        proba_df = predictor.predict_proba(X)

    classes = list(proba_df.columns)
    y_proba = proba_df.values

    # Calculate metrics
    print("Calculating metrics...")
    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": f"{report_dict.get('accuracy', 0):.3f}",
        "f1_macro": f"{report_dict.get('macro avg', {}).get('f1-score', 0):.3f}",
        "f1_weighted": f"{report_dict.get('weighted avg', {}).get('f1-score', 0):.3f}",
        "report_html": pd.DataFrame(report_dict).T.to_html(
            classes="table",
            float_format="%.3f",
            border=0
        ),
    }

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("  - Confusion matrix")
    confusion_matrix_img = plot_confusion_matrix(y, y_pred, labels=classes)

    print("  - ROC curve")
    roc_curve_img = plot_roc_curve(y, y_proba, classes)

    print("  - PCA analysis")
    pca_variance_img, pca_scatter_img = plot_pca_analysis(X, y)

    print("  - SHAP feature importance")
    shap_importance_img = plot_shap_importance(predictor, X, max_samples=args.max_shap_samples)

    # Build and save report
    print("\nBuilding HTML report...")
    html = build_html_report(
        model_name=type(predictor).__name__,
        metrics=metrics,
        confusion_matrix_img=confusion_matrix_img,
        roc_curve_img=roc_curve_img,
        pca_variance_img=pca_variance_img,
        pca_scatter_img=pca_scatter_img,
        shap_importance_img=shap_importance_img
    )

    output_path.write_text(html, encoding="utf-8")

    print("=" * 60)
    print(f"[OK] Report successfully generated: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
