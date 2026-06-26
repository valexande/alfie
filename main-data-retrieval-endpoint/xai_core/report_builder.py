"""
Report Builder - Unified HTML report generation for all explainers.

Layout (in order):
  1. Header + Executive Summary  (natural language)
  2. MODEL BASICS                (metrics, feature importance, predictions)
  3. ADVANCED ANALYSIS           (SHAP, PCA, embeddings) — for data scientists
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from xai_core.base_explainer import BaseModelExplainer
    from xai_core.data_interpretability_service import DataInterpretabilityService

warnings.filterwarnings('ignore')


class ReportBuilder:
    """
    Unified report builder for model explainability.

    Generates model-only HTML reports from any explainer:
      - Executive summary in plain English
      - Model basics (metrics, feature importance, predictions)
      - Advanced section (SHAP, PCA) — at the end

    Example:
        >>> explainer = AutoGluonTabularExplainer(model, X, y)
        >>> html = ReportBuilder(explainer).build(mode='expert')
    """

    def __init__(
        self,
        explainer: "BaseModelExplainer",
        data_service: Optional["DataInterpretabilityService"] = None,
    ):
        self.explainer = explainer
        self.data_service = data_service

    # =========================================================================
    # Public entry point
    # =========================================================================

    def build(self, mode: str = 'expert') -> str:
        metrics = self.explainer.get_metrics()
        plots   = self.explainer.generate_plots()
        return self._build_report(metrics, plots, mode)

    def build_timeseries_report(self, mode: str = 'expert') -> str:
        metrics = self.explainer.get_metrics()
        plots   = self.explainer.generate_plots()
        return self._build_timeseries_html(metrics, plots, mode)

    # =========================================================================
    # Main report assembly
    # =========================================================================

    def _build_report(self, metrics: Dict, plots: Dict, mode: str) -> str:
        header          = self._build_header(metrics, mode)
        executive       = self._build_executive_summary(metrics, plots)
        model_basics    = self._build_model_basics(metrics, plots, mode)
        advanced        = self._build_advanced_section(metrics, plots, mode)

        title = (
            f"Beginner Report — {metrics.get('model_type', 'Unknown')}"
            if mode == 'beginner'
            else f"Expert Report — {metrics.get('model_type', 'Unknown')}"
        )
        return self._wrap_html(
            header + executive + model_basics + advanced,
            title=title
        )

    # =========================================================================
    # Section builders
    # =========================================================================

    def _build_header(self, metrics: Dict, mode: str = 'expert') -> str:
        model_type   = metrics.get('model_type', 'Unknown')
        problem_type = metrics.get('problem_type', 'Unknown')
        n_features   = metrics.get('n_features', '?')
        n_samples    = metrics.get('n_samples', '?')

        if mode == 'beginner':
            mode_badge = '<span class="badge" style="background:#f59e0b;color:#fff">Beginner Guide</span>'
            subtitle   = '<p style="color:#6b7280;margin-top:4px">Plain-language explanation — no statistics background needed.</p>'
        else:
            mode_badge = '<span class="badge" style="background:#6366f1;color:#fff">Expert Analysis</span>'
            subtitle   = '<p style="color:#6b7280;margin-top:4px">Full technical report — metrics, SHAP values, and advanced diagnostics.</p>'

        return f'''
        <div class="header">
            <h1>Model Explainability Report</h1>
            {subtitle}
            <p>
                {mode_badge}
                <span class="badge badge-primary">{model_type}</span>
                <span class="badge badge-success">{problem_type}</span>
                <span class="badge badge-info">{n_features} features</span>
                <span class="badge badge-info">{n_samples:,} samples</span>
            </p>
        </div>'''

    def _build_executive_summary(self, metrics: Dict, plots: Dict) -> str:
        """Auto-generate a plain-English summary of what was found."""
        problem_type = metrics.get('problem_type', 'unknown')
        model_type   = metrics.get('model_type', 'unknown')
        n_features   = metrics.get('n_features', '?')
        n_samples    = metrics.get('n_samples', '?')

        lines = [
            f"We analysed a <strong>{problem_type}</strong> model "
            f"(<em>{model_type}</em>) trained on "
            f"<strong>{n_samples:,} rows</strong> and "
            f"<strong>{n_features} input features</strong>."
        ]

        # Performance headline
        if problem_type == 'classification':
            acc = metrics.get('accuracy')
            auc = metrics.get('roc_auc')
            if acc is not None:
                grade = "excellent" if acc >= 0.9 else ("good" if acc >= 0.75 else "moderate")
                lines.append(
                    f"Overall accuracy is <strong>{acc:.1%}</strong> — {grade} performance."
                )
            if auc is not None:
                lines.append(
                    f"The ROC AUC score is <strong>{auc:.3f}</strong> "
                    f"({'near-perfect' if auc >= 0.95 else 'strong' if auc >= 0.85 else 'acceptable'} "
                    f"discrimination between classes)."
                )
        elif problem_type == 'regression':
            r2   = metrics.get('r2')
            rmse = metrics.get('rmse')
            if r2 is not None:
                lines.append(
                    f"The model explains <strong>{r2:.1%}</strong> of the variance in the target "
                    f"(R² = {r2:.3f})."
                )
            if rmse is not None:
                lines.append(
                    f"Average prediction error (RMSE): <strong>{rmse:.4f}</strong>."
                )

        # Top feature hint
        if 'feature_importance' in plots:
            try:
                fi = self.explainer.get_feature_importance()
                if fi is not None and len(fi) > 0:
                    top = fi.iloc[0]['feature']
                    lines.append(
                        f"The single most influential input feature is "
                        f"<strong>{top}</strong>."
                    )
            except Exception:
                pass

        bullets = "".join(f"<li>{l}</li>" for l in lines)
        return f'''
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="exec-summary">
                <ul>{bullets}</ul>
            </div>
        </div>'''

    # -------------------------------------------------------------------------
    # DATA SECTION
    # -------------------------------------------------------------------------

    def _build_data_section(self, mode: str = 'expert') -> str:
        if self.data_service is None:
            return ""

        ds      = self.data_service
        info    = ds.data_info
        col_inf = ds.column_info
        results = ds.analysis_results
        dplots  = ds.plots

        n_rows, n_cols = info['shape']
        missing_total  = sum(info['missing_values'].values())
        numeric_cols   = [c for c, v in col_inf.items() if v['type'] == 'numeric']
        cat_cols       = [c for c, v in col_inf.items() if v['type'] == 'categorical']

        # ── Overview cards ──
        missing_pct = (missing_total / (n_rows * n_cols) * 100) if n_cols else 0
        missing_label = (
            f'<span style="color:#e53e3e;">{missing_total} ({missing_pct:.1f}%)</span>'
            if missing_total > 0 else
            f'<span style="color:#38a169;">None</span>'
        )
        overview = f'''
        <div class="section-divider"><span>SECTION 1 — YOUR DATA</span></div>
        <div class="section">
            <h2>Dataset Overview</h2>
            {self._narrative(
                "Before we explain the model, let's understand the data it was trained on. "
                "A good dataset is the foundation of a reliable model — here's what yours looks like at a glance."
            )}
            <div class="metrics-grid">
                <div class="metric-card"><div class="value">{n_rows:,}</div><div class="label">Rows</div></div>
                <div class="metric-card"><div class="value">{n_cols}</div><div class="label">Columns</div></div>
                <div class="metric-card"><div class="value">{len(numeric_cols)}</div><div class="label">Numeric Features</div></div>
                <div class="metric-card"><div class="value">{len(cat_cols)}</div><div class="label">Categorical Features</div></div>
                <div class="metric-card"><div class="value">{missing_label}</div><div class="label">Missing Values</div></div>
            </div>
        </div>'''

        # ── Column table ──
        rows_html = ""
        for col, ci in col_inf.items():
            warn = ' <span style="color:#e53e3e;">⚠</span>' if ci['null_percentage'] > 10 else ''
            rows_html += f'''
            <tr>
                <td><strong>{col}</strong></td>
                <td><span class="type-badge type-{ci["type"]}">{ci["type"]}</span></td>
                <td>{ci["null_count"]} ({ci["null_percentage"]:.1f}%){warn}</td>
                <td>{ci["unique_count"]}</td>
            </tr>'''

        col_table = f'''
        <div class="section">
            <h2>Column Summary</h2>
            {self._narrative(
                "Each row in the table below is one column from your dataset. "
                "Red warning marks appear when more than 10% of values are missing — "
                "those columns may need imputation before training."
            )}
            <table>
                <tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique Values</th></tr>
                {rows_html}
            </table>
        </div>'''

        # ── Distributions ──
        dist_html = ""
        if 'numeric_distributions' in dplots:
            # Build outlier callout
            outlier_notes = []
            if 'outliers' in results:
                for col, oi in results['outliers'].items():
                    if oi['percentage'] > 5:
                        outlier_notes.append(
                            f"<strong>{col}</strong> has {oi['percentage']:.1f}% outliers (values far from the average)."
                        )
            outlier_note = (
                "Key outlier observations: " + " ".join(outlier_notes)
                if outlier_notes else
                "No severe outliers detected — distributions look well-behaved."
            )

            dist_html = f'''
            <div class="section">
                <h2>Feature Distributions</h2>
                {self._narrative(
                    "A distribution chart shows how often each value occurs for a given feature. "
                    "Tall narrow peaks indicate most values cluster around one point; "
                    "wide flat distributions indicate high variability. "
                    "Skewed shapes (long tail on one side) often mean outliers are present. "
                    + outlier_note
                )}
                <img src="data:image/png;base64,{dplots['numeric_distributions']}" alt="Distributions"/>
                {self._caption(
                    "What to look for",
                    "Bell-shaped (normal) distributions are easy for models to learn. "
                    "Heavily skewed distributions may benefit from log-transformation.",
                    "Warning signs",
                    [
                        "Very narrow spike at one value = most rows have the same value (low information).",
                        "Bimodal (two humps) = two distinct sub-populations — consider splitting by group.",
                        "Long right tail = outliers pulling the average up.",
                    ],
                    "Example",
                    "If <em>age</em> shows a spike at 0 with a long tail, it likely means many records "
                    "defaulted to age=0 rather than being truly zero-aged."
                )}
            </div>'''

        # ── Correlation heatmap ──
        corr_html = ""
        if 'correlation_heatmap' in dplots:
            # Find strongest correlations from the matrix
            strong_pairs = []
            if 'correlation_matrix' in results:
                corr = results['correlation_matrix']
                for i, c1 in enumerate(corr.columns):
                    for c2 in corr.columns[i+1:]:
                        val = corr.loc[c1, c2]
                        if abs(val) >= 0.7:
                            direction = "positively" if val > 0 else "negatively"
                            strong_pairs.append(
                                f"<strong>{c1}</strong> and <strong>{c2}</strong> "
                                f"are {direction} correlated ({val:.2f})"
                            )

            corr_note = (
                "Notable relationships: " + "; ".join(strong_pairs[:3]) + "."
                if strong_pairs else
                "No strong correlations (>0.7) detected — features appear fairly independent."
            )

            corr_html = f'''
            <div class="section">
                <h2>Feature Correlations</h2>
                {self._narrative(
                    "Correlation tells us how much two features move together. "
                    "A value near +1 means they rise and fall together; near -1 means one rises when the other falls; "
                    "near 0 means they are independent. "
                    "Highly correlated features carry redundant information — keeping both can confuse the model. "
                    + corr_note
                )}
                <img src="data:image/png;base64,{dplots['correlation_heatmap']}" alt="Correlation Heatmap"/>
                {self._caption(
                    "How to read the heatmap",
                    "Dark red = strong positive correlation. Dark blue = strong negative. White/light = little relationship.",
                    "Action points",
                    [
                        "Pairs above 0.9 correlation: consider dropping one — they carry the same information.",
                        "High correlation with the target column = very useful predictive feature.",
                        "Perfect 1.0 on diagonal (each feature with itself) is expected.",
                    ],
                    "Example",
                    "If <em>total_spend</em> and <em>num_transactions</em> show 0.95 correlation, "
                    "one can be dropped without losing predictive power."
                )}
            </div>'''

        # ── Categorical distributions ──
        cat_html = ""
        if 'categorical_distributions' in dplots:
            cat_html = f'''
            <div class="section">
                <h2>Categorical Feature Distributions</h2>
                {self._narrative(
                    "Categorical features are labels or groups rather than numbers — for example, "
                    "country, status, or product type. "
                    "The bar chart below shows how many rows fall into each category. "
                    "Heavily imbalanced categories (one bar vastly taller than others) can bias the model "
                    "toward the dominant category."
                )}
                <img src="data:image/png;base64,{dplots['categorical_distributions']}" alt="Categorical Distributions"/>
                {self._caption(
                    "What to look for",
                    "Balanced bars = the model sees roughly equal examples of each category.",
                    "Warning signs",
                    [
                        "One category has 90%+ of rows: the model may always predict that category.",
                        "Many categories with very few examples: may need grouping into 'Other'.",
                    ],
                    "Example",
                    "If <em>country</em> is 95% 'US', the model learns little from that feature — "
                    "it just echoes the majority."
                )}
            </div>'''

        # Beginner: skip correlation heatmap (too technical)
        if mode == 'beginner':
            corr_html = ""

        return overview + col_table + dist_html + corr_html + cat_html

    # -------------------------------------------------------------------------
    # MODEL BASICS
    # -------------------------------------------------------------------------

    def _build_model_basics(self, metrics: Dict, plots: Dict, mode: str) -> str:
        problem_type = metrics.get('problem_type', 'unknown')

        metrics_html     = self._format_metrics_cards(metrics)
        metrics_narrative = self._build_metrics_narrative(metrics, mode)

        importance_html = ""
        if 'feature_importance' in plots:
            importance_html = f'''
            <div class="section">
                <h2>Feature Importance</h2>
                {self._narrative(
                    "Feature importance answers the key question: "
                    "<em>which inputs does this model actually rely on?</em> "
                    "The chart ranks every feature from most to least influential. "
                    "You can use this to simplify the model (by dropping unimportant features) "
                    "or to verify the model is using the right signals."
                )}
                <img src="data:image/png;base64,{plots['feature_importance']}" alt="Feature Importance"/>
                {self._caption(
                    "What is this?",
                    "Bar length = how much this feature drives predictions. "
                    "Longer bar = stronger influence.",
                    "How to read it",
                    [
                        "Top feature = the single biggest driver of predictions.",
                        "Features with ~0 importance can be safely removed.",
                        "If an important feature surprises you, investigate whether the model learned a spurious correlation.",
                    ],
                    "Example",
                    "In a house-price model, <em>square_footage</em> bar at 0.45 vs <em>garage_colour</em> at 0.001 "
                    "confirms that size matters far more than colour."
                )}
            </div>'''

        # Classification plots
        pred_html = ""
        if problem_type == 'classification':
            if 'confusion_matrix' in plots:
                if mode == 'beginner':
                    cm_narrative = (
                        "Think of this as a score card. Each box shows how often the model "
                        "got a prediction right or wrong. The boxes along the diagonal "
                        "(top-left to bottom-right) are correct predictions — you want those "
                        "to be the biggest numbers. Any other box is a mistake."
                    )
                    cm_caption_body = [
                        "Big numbers on the diagonal = the model is doing well.",
                        "Numbers off the diagonal = types of mistakes being made.",
                    ]
                else:
                    cm_narrative = (
                        "The confusion matrix is the most direct way to see where the model succeeds and fails. "
                        "Each cell shows how many times the model predicted one class when the true answer was another. "
                        "We want large numbers on the diagonal (correct predictions) and small numbers everywhere else."
                    )
                    cm_caption_body = [
                        "Diagonal cells = correct predictions (want these large).",
                        "Off-diagonal cells = mistakes; row = true class, column = predicted class.",
                        "Large off-diagonal number between two classes = the model frequently confuses them.",
                    ]

                pred_html += f'''
                <div class="section">
                    <h2>Confusion Matrix — What did the model get right?</h2>
                    {self._narrative(cm_narrative)}
                    <img src="data:image/png;base64,{plots['confusion_matrix']}" alt="Confusion Matrix"/>
                    {self._caption(
                        "What is this?",
                        "A grid counting every predicted class vs. actual class across all test samples.",
                        "How to read it",
                        cm_caption_body,
                        "Example",
                        "Actual=Spam / Predicted=Ham = 80 means 80 spam emails slipped through as legitimate — a false-negative problem."
                    )}
                </div>'''

            # ROC curve — expert only (too technical for beginners)
            if 'roc_curve' in plots and mode == 'expert':
                pred_html += f'''
                <div class="section">
                    <h2>ROC Curve — How reliably does the model separate classes?</h2>
                    {self._narrative(
                        "The ROC curve shows the trade-off between catching true positives and accidentally flagging false positives. "
                        "Every point on the curve represents a different decision threshold. "
                        "A perfect model hugs the top-left corner; a random guess follows the diagonal."
                    )}
                    <img src="data:image/png;base64,{plots['roc_curve']}" alt="ROC Curve"/>
                    {self._caption(
                        "What is this?",
                        "True Positive Rate (sensitivity) vs False Positive Rate at every threshold.",
                        "How to read it",
                        [
                            "Curve close to top-left corner = excellent discrimination.",
                            "AUC 0.9–1.0 = excellent · 0.7–0.9 = good · 0.5–0.7 = poor.",
                            "Diagonal dashed line = random guessing (AUC = 0.5).",
                        ],
                        "Example",
                        "A cancer screening model with AUC = 0.97 catches 97% of true cases "
                        "before false-alarm rates become problematic."
                    )}
                </div>'''

        elif problem_type == 'regression':
            if 'residuals' in plots:
                pred_html = f'''
                <div class="section">
                    <h2>Prediction Analysis — How accurate are the predictions?</h2>
                    {self._narrative(
                        "For regression models, we compare what the model predicted against what actually happened. "
                        "The left chart plots predictions against true values — ideally all points lie on the diagonal. "
                        "The right chart shows the distribution of errors (residuals). "
                        "Errors should be small, centred at zero, and bell-shaped — any other pattern hints at a systematic bias."
                    )}
                    <img src="data:image/png;base64,{plots['residuals']}" alt="Residuals"/>
                    {self._caption(
                        "What is this?",
                        "Left: predicted vs actual scatter. Right: error distribution histogram.",
                        "How to read it",
                        [
                            "Points near the red diagonal = accurate predictions.",
                            "Fan shape (wider scatter at high values) = model is less reliable for larger values.",
                            "Residual histogram centred at zero = no systematic over/under-prediction.",
                        ],
                        "Example",
                        "If residuals are systematically +5,000, the model consistently under-predicts — "
                        "a missing feature (e.g. years of experience) likely explains the gap."
                    )}
                </div>'''

        if mode == 'beginner':
            divider = '<div class="section-divider"><span>SECTION 2 — HOW THE MODEL PERFORMS</span></div>'
            expert_note = '''
            <div class="section" style="background:#f0f9ff;border-left:4px solid #3b82f6;padding:16px;">
                <p style="margin:0;color:#1e40af;">
                    <strong>Want more detail?</strong>
                    Re-run this report with <code>user_level=expert</code> to see ROC curves,
                    SHAP value analysis, PCA variance plots, and all technical metrics.
                </p>
            </div>'''
        else:
            divider = '<div class="section-divider"><span>SECTION 2 — MODEL PERFORMANCE</span></div>'
            expert_note = ""

        # ---- Per-class breakdown (populated by VisionClassifierExplainer and
        #      any other explainer that adds 'per_class' to its metrics dict) ----
        per_class_html = ""
        per_class = metrics.get("per_class")
        if per_class and mode == "expert":
            rows_html = ""
            for cls_name, cls_metrics in per_class.items():
                rows_html += (
                    f"<tr>"
                    f"<td><strong>{cls_name}</strong></td>"
                    f"<td>{cls_metrics.get('accuracy', '—'):.3f}</td>"
                    f"<td>{cls_metrics.get('precision', '—'):.3f}</td>"
                    f"<td>{cls_metrics.get('recall', '—'):.3f}</td>"
                    f"<td>{cls_metrics.get('f1', '—'):.3f}</td>"
                    f"<td>{cls_metrics.get('support', '—')}</td>"
                    f"</tr>"
                )
            per_class_html = f'''
            <div class="section">
                <h2>Per-Class Performance</h2>
                {self._narrative(
                    "Breaking accuracy down by class reveals whether the model performs "
                    "equally well across all categories or struggles with specific ones."
                )}
                <table style="width:100%;border-collapse:collapse;margin-top:12px">
                    <thead>
                        <tr style="background:#f3f4f6">
                            <th style="padding:8px;text-align:left;border:1px solid #e5e7eb">Class</th>
                            <th style="padding:8px;text-align:center;border:1px solid #e5e7eb">Accuracy</th>
                            <th style="padding:8px;text-align:center;border:1px solid #e5e7eb">Precision</th>
                            <th style="padding:8px;text-align:center;border:1px solid #e5e7eb">Recall</th>
                            <th style="padding:8px;text-align:center;border:1px solid #e5e7eb">F1</th>
                            <th style="padding:8px;text-align:center;border:1px solid #e5e7eb">Support</th>
                        </tr>
                    </thead>
                    <tbody style="font-size:0.92em">{rows_html}</tbody>
                </table>
            </div>'''

        # ---- Class distribution bar chart (vision only, rendered as section) ----
        class_dist_html = ""
        if "class_distribution" in plots:
            class_dist_html = f'''
            <div class="section">
                <h2>Class Distribution — Ground Truth vs Predicted</h2>
                {self._narrative(
                    "Comparing how many images belong to each class (ground truth) against "
                    "how many were predicted shows whether the model systematically "
                    "over- or under-predicts any category."
                )}
                <img src="data:image/png;base64,{plots['class_distribution']}"
                     alt="Class Distribution" style="max-width:100%"/>
            </div>'''

        return divider + metrics_html + metrics_narrative + importance_html + per_class_html + class_dist_html + pred_html + expert_note

    def _build_metrics_narrative(self, metrics: Dict, mode: str = 'expert') -> str:
        problem_type = metrics.get('problem_type', 'unknown')
        lines = []

        if problem_type == 'classification':
            acc = metrics.get('accuracy')
            f1  = metrics.get('f1')
            auc = metrics.get('roc_auc')
            if acc is not None:
                pct = f"{acc:.1%}"
                if mode == 'beginner':
                    verdict = ("Great — the model is very reliable." if acc >= 0.9
                               else "Decent — mostly correct with some room to improve." if acc >= 0.7
                               else "Needs work — the model gets it wrong too often.")
                    lines.append(
                        f"The model gets the right answer <strong>{pct} of the time</strong>. {verdict}"
                    )
                else:
                    verdict = ("The model is highly reliable." if acc >= 0.9
                               else "Performance is acceptable but there is room for improvement." if acc >= 0.7
                               else "Performance is below par — consider retraining with more data or better features.")
                    lines.append(f"Accuracy of <strong>{pct}</strong> means the model predicts the correct class "
                                  f"<strong>{pct}</strong> of the time. {verdict}")
            if mode == 'expert':
                if f1 is not None:
                    lines.append(f"F1 Score of <strong>{f1:.3f}</strong> balances precision and recall — "
                                  f"useful when classes are imbalanced.")
                if auc is not None:
                    lines.append(f"ROC AUC of <strong>{auc:.3f}</strong> measures discrimination ability independent of threshold.")

        elif problem_type == 'regression':
            r2   = metrics.get('r2')
            rmse = metrics.get('rmse')
            mae  = metrics.get('mae')
            if r2 is not None:
                if mode == 'beginner':
                    verdict = ("Strong — the model captures most of the pattern." if r2 >= 0.8
                               else "Moderate — useful but misses some patterns." if r2 >= 0.5
                               else "Weak — the model struggles to find the pattern.")
                    lines.append(
                        f"The model explains <strong>{r2:.0%} of the variation</strong> in outcomes. {verdict}"
                    )
                else:
                    verdict = ("The model explains most of the variance — strong fit." if r2 >= 0.8
                               else "Moderate explanatory power — useful but imperfect." if r2 >= 0.5
                               else "Low explanatory power — the model misses important patterns.")
                    lines.append(f"R² of <strong>{r2:.3f}</strong>: the model explains <strong>{r2:.1%}</strong> "
                                  f"of the variance in the target. {verdict}")
            if rmse is not None and mae is not None:
                if mode == 'beginner':
                    lines.append(f"On average, predictions are off by about <strong>{mae:.4f}</strong>.")
                else:
                    lines.append(f"On average, predictions are off by <strong>~{mae:.4f}</strong> (MAE) "
                                  f"or <strong>~{rmse:.4f}</strong> (RMSE — penalises large errors more).")

        if not lines:
            return ""

        bullets = "".join(f"<li>{l}</li>" for l in lines)
        return f'<div class="section"><div class="exec-summary"><ul>{bullets}</ul></div></div>'

    # -------------------------------------------------------------------------
    # ADVANCED SECTION
    # -------------------------------------------------------------------------

    def _build_advanced_section(self, metrics: Dict, plots: Dict, mode: str) -> str:
        # Beginners don't need SHAP, PCA, or embeddings — skip the whole section
        if mode == 'beginner':
            return ""

        vision_plots = any(k in plots for k in ('gradcam_gallery', 'misclassified_gallery'))
        has_advanced = any(k in plots for k in ('shap_summary', 'pca_variance',
                                                 'embeddings_pca', 'embeddings_tsne',
                                                 'text_explanations'))
        if not has_advanced and not vision_plots:
            return ""

        divider = '<div class="section-divider"><span>SECTION 3 — ADVANCED ANALYSIS (for data scientists)</span></div>'

        intro = f'''
        <div class="section">
            {self._narrative(
                "The charts in this section go deeper than standard metrics — they reveal <em>how</em> the model "
                "thinks at a feature-by-feature and sample-by-sample level. "
                "They require some statistical background to interpret fully, "
                "which is why they appear here at the end rather than upfront."
            )}
        </div>'''

        shap_html = ""
        # skip_shap_section is set by models where pixel-level SHAP is meaningless
        if metrics.get("skip_shap_section"):
            shap_html = ""
        elif 'shap_summary' in plots:
            shap_html = f'''
            <div class="section">
                <h2>SHAP Value Analysis</h2>
                {self._narrative(
                    "SHAP (SHapley Additive exPlanations) goes beyond feature importance by showing "
                    "<em>how much</em> and <em>in which direction</em> each feature pushes a specific prediction "
                    "away from the model's average output. "
                    "It is grounded in game theory: each feature gets a 'fair share' of the credit or blame "
                    "for every individual prediction. "
                    "Unlike feature importance (which is a single number per feature), SHAP gives you one value "
                    "per feature <em>per sample</em> — so you can see why the model was confident for one row "
                    "but uncertain for another."
                )}
                <img src="data:image/png;base64,{plots['shap_summary']}" alt="SHAP Summary"/>
                {self._caption(
                    "What is this?",
                    "Each dot = one sample. X-axis = SHAP value (positive pushes prediction up, negative pushes it down). "
                    "Colour = feature value (red = high, blue = low). Features ranked by mean absolute SHAP.",
                    "How to read it",
                    [
                        "Red dots on the right = high feature value strongly increases the prediction.",
                        "Blue dots on the left = low feature value strongly decreases the prediction.",
                        "Wide spread of dots = this feature has variable influence depending on its value.",
                        "Tight cluster near zero = this feature barely matters for any prediction.",
                    ],
                    "Example",
                    "In a loan default model, <em>credit_score</em> red dots appear on the far left — "
                    "high scores strongly suppress default risk. <em>debt_ratio</em> red dots appear on the far right — "
                    "high debt strongly increases default risk."
                )}
            </div>'''

        text_explanations_html = ""
        if 'text_explanations' in plots:
            text_explanations_html = f'''
            <div class="section">
                <h2>Text Token Explanations</h2>
                {self._narrative(
                    "For text classification models, this section highlights which words changed the model's "
                    "confidence for each displayed prediction. The method masks one token at a time and measures "
                    "the drop or increase in the predicted-class probability. Green tokens support the predicted "
                    "label; red tokens push against it."
                )}
                <div class="info-box">
                    These are local explanations for individual examples. They should be read together with
                    per-class metrics and the confusion matrix, especially when the dataset is imbalanced.
                </div>
                {plots['text_explanations']}
            </div>'''

        pca_html = ""
        if 'pca_variance' in plots:
            pca_content = ""
            if 'pca_variance' in plots:
                pca_content += f'<img src="data:image/png;base64,{plots["pca_variance"]}" alt="PCA Variance"/>'
            pca_html = f'''
            <div class="section">
                <h2>PCA Analysis — Variance Structure</h2>
                {self._narrative(
                    "Principal Component Analysis (PCA) compresses all your features into a small number of "
                    "'summary axes' that capture as much variation as possible. "
                    "Think of it as finding the best angle to photograph a complex 3D shape so that the photo "
                    "preserves the most detail. "
                    "The bar chart shows how much of the total data variance each component captures."
                )}
                {pca_content}
                {self._caption(
                    "What is this?",
                    "Bar chart: variance explained per component.",
                    "How to read it",
                    [
                        "If the first 2–3 bars cover 80%+ variance, your data has a compact structure — "
                        "a simpler model may suffice.",
                    ],
                    "Example",
                    "In a customer dataset, a small number of PCA components explaining most variance means "
                    "the strongest patterns can be summarized with fewer dimensions."
                )}
            </div>'''

        embedding_html = ""
        if 'embeddings_pca' in plots or 'embeddings_tsne' in plots:
            emb_content = ""
            if 'embeddings_pca' in plots:
                emb_content += f'<img src="data:image/png;base64,{plots["embeddings_pca"]}" alt="Embeddings PCA"/>'
            if 'embeddings_tsne' in plots:
                emb_content += f'<img src="data:image/png;base64,{plots["embeddings_tsne"]}" alt="Embeddings t-SNE"/>'
            embedding_html = f'''
            <div class="section">
                <h2>Embedding Visualisation</h2>
                {self._narrative(
                    "Multimodal models convert raw inputs (text, images, tables) into high-dimensional numeric vectors "
                    "called embeddings. These are the model's internal 'understanding' of each sample. "
                    "Because embeddings can have hundreds of dimensions, we project them to 2D: "
                    "PCA (fast, linear) for global structure, and t-SNE (slower, non-linear) for local clusters."
                )}
                {emb_content}
                {self._caption(
                    "What is this?",
                    "Left: PCA projection of model embeddings. Right: t-SNE projection. Colours = target classes.",
                    "How to read it",
                    [
                        "Tight, well-separated clusters per class = model learned meaningful representations.",
                        "Mixed clusters = model struggles to separate those classes internally.",
                        "t-SNE is better for spotting sub-clusters; PCA is better for global spread.",
                    ],
                    "Example",
                    "In a sentiment model, positive reviews cluster tightly on the right, negative on the left — "
                    "the model has learned semantically meaningful embeddings."
                )}
            </div>'''

        # ---- Vision-model galleries (GradCAM + misclassifications) ----
        gradcam_html = ""
        if 'gradcam_gallery' in plots:
            gradcam_html = f'''
            <div class="section">
                <h2>GradCAM — Where the Model Looks</h2>
                {self._narrative(
                    "GradCAM (Gradient-weighted Class Activation Mapping) highlights the image regions that "
                    "were most important for each prediction. "
                    "Red areas = highest influence; blue = lowest. "
                    "Each column shows one representative correctly-classified image per class."
                )}
                <img src="data:image/png;base64,{plots['gradcam_gallery']}"
                     alt="GradCAM Gallery" style="max-width:100%;border-radius:6px"/>
            </div>'''

        misclassified_html = ""
        if 'misclassified_gallery' in plots:
            misclassified_html = f'''
            <div class="section">
                <h2>Misclassified Examples</h2>
                {self._narrative(
                    "These images were predicted incorrectly. "
                    "Reviewing mistakes helps identify systematic failure modes — "
                    "for example, the model may struggle with images that share visual features "
                    "across two classes."
                )}
                <img src="data:image/png;base64,{plots['misclassified_gallery']}"
                     alt="Misclassified Gallery" style="max-width:100%;border-radius:6px"/>
            </div>'''

        return (
            divider + intro + shap_html + text_explanations_html + pca_html
            + embedding_html + gradcam_html + misclassified_html
        )

    # =========================================================================
    # Time series report (separate path)
    # =========================================================================

    def _build_timeseries_html(self, metrics: Dict, plots: Dict, mode: str) -> str:
        forecast_html = ""
        if 'forecast' in plots:
            forecast_html = f'''
            <div class="section">
                <h2>Forecast Visualisation</h2>
                {self._narrative(
                    "The solid line shows the model's predicted future values based on historical patterns. "
                    "The shaded band around it is the uncertainty interval — the model's honest estimate of how "
                    "wrong it might be. A narrow band means high confidence; a wide band means the future is harder to predict."
                )}
                <img src="data:image/png;base64,{plots['forecast']}" alt="Forecast"/>
                {self._caption(
                    "What is this?",
                    "Predicted future values (line) with a confidence interval (shaded band).",
                    "How to read it",
                    [
                        "Narrow band = confident forecast.",
                        "Wide band = high uncertainty — treat predictions with caution.",
                        "Where the line meets historical data = the forecast start point.",
                    ],
                    "Example",
                    "Monthly sales forecast of 12,000 units with a band of 10,500–13,500 means the model "
                    "is fairly confident but acknowledges ±12.5% seasonal uncertainty."
                )}
            </div>'''

        info_box = '''
        <div class="info-box">
            <strong>About Time Series Models</strong><br>
            Time series forecasting models predict future values from historical sequences.
            Traditional feature importance and SHAP analysis are not applicable here —
            what matters is the shape, trend, and seasonality of the forecast.
        </div>'''

        return self._wrap_html(f'''
            <div class="header">
                <h1>Time Series Forecast Report</h1>
                <p>
                    <span class="badge badge-primary">{metrics.get('model_type', 'Time Series')}</span>
                    <span class="badge badge-success">Forecasting</span>
                </p>
            </div>
            <div class="section-divider"><span>FORECAST</span></div>
            {info_box}
            {self._format_metrics_cards(metrics)}
            {forecast_html}
        ''', title="Time Series Report")

    # =========================================================================
    # Beginner report (simplified)
    # =========================================================================

    def _build_beginner_report(self, metrics: Dict, plots: Dict) -> str:
        insights = self._generate_insights(metrics)

        importance_html = ""
        if 'feature_importance' in plots:
            importance_html = f'''
            <div class="section">
                <h2>What Features Matter Most?</h2>
                {self._narrative(
                    "This chart shows which inputs the model relies on most. "
                    "The longest bar = the most important feature."
                )}
                <img src="data:image/png;base64,{plots['feature_importance']}" alt="Feature Importance"/>
                {self._caption(
                    "In plain English",
                    "Think of this as a list of clues the model uses.",
                    "What to look for",
                    ["Longest bar = most important clue.",
                     "Very short bars = those features barely matter."],
                    "Example",
                    "If <em>number_of_rooms</em> has the longest bar in a house-price model, "
                    "adding rooms raises the price more than any other factor."
                )}
            </div>'''

        roc_html = ""
        if 'roc_curve' in plots:
            roc_html = f'''
            <div class="section">
                <h2>How Well Does the Model Distinguish Outcomes?</h2>
                {self._narrative(
                    "This curve shows how good the model is at separating outcomes. "
                    "A curve hugging the top-left corner is excellent."
                )}
                <img src="data:image/png;base64,{plots['roc_curve']}" alt="ROC Curve"/>
            </div>'''

        confusion_html = ""
        if 'confusion_matrix' in plots:
            confusion_html = f'''
            <div class="section">
                <h2>Prediction Accuracy Breakdown</h2>
                {self._narrative(
                    "This grid shows how often the model gets each category right or wrong. "
                    "Big numbers on the diagonal = good accuracy."
                )}
                <img src="data:image/png;base64,{plots['confusion_matrix']}" alt="Confusion Matrix"/>
            </div>'''

        return self._wrap_html(f'''
            <div class="header">
                <h1>Model Explainability Report</h1>
                <p><strong>Model:</strong> {metrics.get("model_type", "Unknown")}
                   &nbsp;|&nbsp; <strong>Task:</strong> {metrics.get("problem_type", "Unknown")}</p>
            </div>
            <div class="section">
                <h2>Key Insights</h2>
                <div class="exec-summary">{insights}</div>
            </div>
            {self._format_metrics_cards(metrics)}
            {importance_html}
            {roc_html}
            {confusion_html}
        ''', title="Model Report — Beginner")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _narrative(self, text: str) -> str:
        """Highlighted plain-English paragraph shown BEFORE a chart."""
        return f'<div class="narrative"><p>{text}</p></div>'

    def _caption(
        self,
        what_label: str,
        what_text: str,
        howto_label: str,
        howto_bullets: list,
        example_label: str,
        example_text: str,
    ) -> str:
        """3-column explanation box shown BELOW a chart."""
        bullets_html = "".join(f"<li>{b}</li>" for b in howto_bullets)
        return f'''
        <table class="caption-table">
            <tr>
                <td class="caption-col">
                    <span class="caption-label">{what_label}</span>
                    <p>{what_text}</p>
                </td>
                <td class="caption-col">
                    <span class="caption-label">{howto_label}</span>
                    <ul>{bullets_html}</ul>
                </td>
                <td class="caption-col caption-example">
                    <span class="caption-label">{example_label}</span>
                    <p>{example_text}</p>
                </td>
            </tr>
        </table>'''

    def _generate_insights(self, metrics: Dict) -> str:
        insights = []
        problem_type = metrics.get('problem_type', 'unknown')
        if problem_type == 'classification':
            acc = metrics.get('accuracy')
            if acc is not None:
                grade = "excellent" if acc >= 0.9 else ("good" if acc >= 0.7 else "could be improved")
                insights.append(f"<li>Model accuracy is {grade} at <strong>{acc:.1%}</strong>.</li>")
            auc = metrics.get('roc_auc')
            if auc is not None and auc >= 0.7:
                insights.append(f"<li>The model distinguishes classes well (AUC = {auc:.3f}).</li>")
        elif problem_type == 'regression':
            r2 = metrics.get('r2')
            if r2 is not None:
                insights.append(f"<li>The model explains <strong>{r2:.1%}</strong> of outcome variance (R² = {r2:.3f}).</li>")
        n_feat = metrics.get('n_features', '?')
        n_samp = metrics.get('n_samples', '?')
        insights.append(f"<li>Trained on <strong>{n_samp:,}</strong> samples with <strong>{n_feat}</strong> features.</li>")
        return f"<ul>{''.join(insights)}</ul>" if insights else "<p>Analysis complete.</p>"

    def _format_metrics_cards(self, metrics: Dict) -> str:
        metric_config = {
            'accuracy': ('Accuracy',  lambda x: f"{x:.1%}"),
            'precision': ('Precision', lambda x: f"{x:.3f}"),
            'recall':    ('Recall',    lambda x: f"{x:.3f}"),
            'f1':        ('F1 Score',  lambda x: f"{x:.3f}"),
            'roc_auc':   ('ROC AUC',   lambda x: f"{x:.3f}"),
            'mae':       ('MAE',       lambda x: f"{x:.4f}"),
            'rmse':      ('RMSE',      lambda x: f"{x:.4f}"),
            'r2':        ('R² Score',  lambda x: f"{x:.4f}"),
            'n_features':('Features',  lambda x: str(x)),
            'n_samples': ('Samples',   lambda x: f"{x:,}"),
        }
        cards = []
        for key, (label, fmt) in metric_config.items():
            if key in metrics and metrics[key] is not None:
                try:
                    cards.append(f'''
                    <div class="metric-card">
                        <div class="value">{fmt(metrics[key])}</div>
                        <div class="label">{label}</div>
                    </div>''')
                except Exception:
                    pass
        if not cards:
            return ""
        return f'''
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">{"".join(cards)}</div>
        </div>'''

    # =========================================================================
    # HTML wrapper + CSS
    # =========================================================================

    def _wrap_html(self, body: str, title: str = "Model Report") -> str:
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f0f2f5; padding: 20px;
        }}
        .container {{
            max-width: 1200px; margin: 0 auto; background: white;
            padding: 40px; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }}
        /* Header */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 32px; border-radius: 10px; margin-bottom: 32px;
        }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        /* Section divider */
        .section-divider {{
            display: flex; align-items: center; margin: 40px 0 24px;
        }}
        .section-divider::before, .section-divider::after {{
            content: ""; flex: 1; height: 2px; background: #667eea;
        }}
        .section-divider span {{
            padding: 4px 16px; background: #667eea; color: white;
            border-radius: 20px; font-size: 0.78em; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
            margin: 0 12px;
        }}
        /* Sections */
        .section {{ margin: 28px 0; }}
        .section h2 {{
            color: #667eea; font-size: 1.6em; margin-bottom: 12px;
            padding-bottom: 8px; border-bottom: 3px solid #e8eaf6;
        }}
        /* Narrative box (before chart) */
        .narrative {{
            background: #f7f8ff; border-left: 4px solid #667eea;
            padding: 14px 18px; border-radius: 0 8px 8px 0;
            margin-bottom: 16px; font-size: 0.95em; color: #2d3748; line-height: 1.65;
        }}
        /* Executive summary */
        .exec-summary {{
            background: #eef9f0; border: 1px solid #c6e6cb;
            border-radius: 8px; padding: 18px 22px;
        }}
        .exec-summary ul {{ padding-left: 20px; }}
        .exec-summary li {{ margin-bottom: 6px; font-size: 0.95em; }}
        /* Metrics grid */
        .metrics-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px; margin: 16px 0;
        }}
        .metric-card {{
            background: #f8f9fa; padding: 18px; border-radius: 8px;
            text-align: center; border-left: 4px solid #667eea;
        }}
        .metric-card .value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .metric-card .label {{
            color: #666; font-size: 0.82em; margin-top: 4px;
            text-transform: uppercase; letter-spacing: 0.5px;
        }}
        /* Images */
        img {{
            max-width: 100%; height: auto; border: 1px solid #e0e0e0;
            border-radius: 8px; margin: 12px 0; display: block;
        }}
        /* Column type badges */
        .type-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 0.8em; font-weight: 600;
        }}
        .type-numeric     {{ background: #e3f2fd; color: #1565c0; }}
        .type-categorical {{ background: #f3e5f5; color: #6a1b9a; }}
        .type-boolean     {{ background: #e8f5e9; color: #2e7d32; }}
        .type-datetime    {{ background: #fff8e1; color: #f57f17; }}
        .type-unknown     {{ background: #fafafa; color: #666; }}
        /* Tables */
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.9em; }}
        th, td {{ padding: 10px 14px; text-align: left; border: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
        tr:hover {{ background: #fafafa; }}
        /* Caption table (below charts) */
        .caption-table {{ margin-top: 10px; border: 1px solid #c5d0f5; border-radius: 8px; overflow: hidden; }}
        .caption-table td {{ border: none; border-right: 1px solid #c5d0f5; padding: 14px 16px; vertical-align: top; width: 33%; background: #f0f4ff; }}
        .caption-table td:last-child {{ border-right: none; }}
        .caption-label {{
            display: block; font-weight: 700; font-size: 0.75em;
            text-transform: uppercase; letter-spacing: 0.6px; color: #4a5568; margin-bottom: 5px;
        }}
        .caption-col p, .caption-col ul {{ font-size: 0.87em; color: #374151; line-height: 1.55; }}
        .caption-col ul {{ padding-left: 16px; margin: 0; }}
        .caption-col ul li {{ margin-bottom: 3px; }}
        .caption-example {{ background: #eef9f0 !important; border-left: 3px solid #38a169 !important; }}
        .caption-example p {{ color: #276749; }}
        /* Badges */
        .badge {{
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.82em; font-weight: 500; margin-right: 6px;
        }}
        .badge-primary {{ background: rgba(255,255,255,0.25); color: white; }}
        .badge-success {{ background: rgba(56,161,105,0.35); color: white; }}
        .badge-info    {{ background: rgba(255,255,255,0.15); color: white; }}
        /* Info box */
        .info-box {{
            background: #e3f2fd; border-left: 4px solid #2196f3;
            padding: 16px 20px; margin: 16px 0; border-radius: 4px;
        }}
        /* Text token explanations */
        .text-explanations {{
            display: grid; gap: 16px; margin-top: 16px;
        }}
        .text-card {{
            border: 1px solid #d7def2; border-radius: 8px; padding: 16px;
            background: #fbfcff;
        }}
        .text-card-meta {{
            display: flex; flex-wrap: wrap; gap: 10px 18px;
            color: #4b5563; font-size: 0.86em; margin-bottom: 12px;
        }}
        .token-highlight {{
            font-size: 0.98em; line-height: 2.05; color: #1f2937;
        }}
        .token-highlight span {{
            display: inline-block; padding: 0 4px; border-radius: 4px;
            margin: 1px 0;
        }}
        .token-positive {{
            background: rgba(34, 197, 94, var(--token-alpha));
            border-bottom: 2px solid rgba(22, 163, 74, 0.65);
        }}
        .token-negative {{
            background: rgba(239, 68, 68, var(--token-alpha));
            border-bottom: 2px solid rgba(220, 38, 38, 0.65);
        }}
        .token-neutral {{ background: transparent; }}
        .token-table {{
            margin-top: 14px; max-width: 520px; font-size: 0.86em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {body}
    </div>
</body>
</html>'''
