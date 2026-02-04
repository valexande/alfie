"""
Report Builder - Unified HTML report generation for all explainers.

Works with any BaseModelExplainer subclass to generate
comprehensive explainability reports.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')


class ReportBuilder:
    """
    Unified report builder for model explainability.
    
    Generates HTML reports from any explainer with:
    - Model information and metrics
    - Feature importance visualization
    - SHAP analysis (if available)
    - Classification/regression specific plots
    - PCA analysis
    
    Example:
        >>> explainer = TreeBasedExplainer(model, X, y)
        >>> builder = ReportBuilder(explainer)
        >>> html = builder.build(mode='expert')
    """
    
    def __init__(self, explainer: "BaseModelExplainer"):
        """
        Initialize report builder.
        
        Args:
            explainer: Any BaseModelExplainer instance
        """
        self.explainer = explainer
    
    def build(self, mode: str = 'expert') -> str:
        """
        Build HTML report.
        
        Args:
            mode: 'beginner' for simplified, 'expert' for full details
            
        Returns:
            HTML string
        """
        # Get all data from explainer
        metrics = self.explainer.get_metrics()
        plots = self.explainer.generate_plots()
        
        if mode == 'beginner':
            return self._build_beginner_report(metrics, plots)
        else:
            return self._build_expert_report(metrics, plots)
    
    def build_timeseries_report(self, mode: str = 'expert') -> str:
        """Build specialized report for time series models."""
        metrics = self.explainer.get_metrics()
        plots = self.explainer.generate_plots()
        
        return self._build_timeseries_html(metrics, plots, mode)
    
    def _build_beginner_report(self, metrics: Dict, plots: Dict) -> str:
        """Build simplified beginner-friendly report."""
        
        # Key insights section
        insights = self._generate_insights(metrics)
        
        # Feature importance plot
        importance_html = ""
        if 'feature_importance' in plots:
            importance_html = f'''
            <div class="section">
                <h2>What Features Matter Most?</h2>
                <p>The chart below shows which features (inputs) have the biggest impact on the model's predictions.</p>
                <img src="data:image/png;base64,{plots['feature_importance']}" alt="Feature Importance"/>
            </div>
            '''
        
        # ROC curve for classification
        roc_html = ""
        if 'roc_curve' in plots:
            roc_html = f'''
            <div class="section">
                <h2>Model Performance</h2>
                <p>This curve shows how well the model distinguishes between different outcomes. 
                   A curve closer to the top-left corner indicates better performance.</p>
                <img src="data:image/png;base64,{plots['roc_curve']}" alt="ROC Curve"/>
            </div>
            '''
        
        # Confusion matrix for classification
        confusion_html = ""
        if 'confusion_matrix' in plots:
            confusion_html = f'''
            <div class="section">
                <h2>Prediction Accuracy</h2>
                <p>This matrix shows how often the model makes correct predictions. 
                   Numbers on the diagonal represent correct predictions.</p>
                <img src="data:image/png;base64,{plots['confusion_matrix']}" alt="Confusion Matrix"/>
            </div>
            '''
        
        return self._wrap_html(f'''
            <div class="header">
                <h1>Model Explainability Report</h1>
                <p><strong>Model Type:</strong> {metrics.get('model_type', 'Unknown')}</p>
                <p><strong>Task:</strong> {metrics.get('problem_type', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                <div class="info-box">
                    {insights}
                </div>
            </div>
            
            {self._format_metrics_cards(metrics)}
            
            {importance_html}
            {roc_html}
            {confusion_html}
        ''', title="Model Report - Beginner")
    
    def _build_expert_report(self, metrics: Dict, plots: Dict) -> str:
        """Build detailed expert report."""
        
        # Metrics section
        metrics_html = self._format_metrics_cards(metrics)
        
        # Feature importance
        importance_html = ""
        if 'feature_importance' in plots:
            importance_html = self._section(
                "Feature Importance",
                f'<img src="data:image/png;base64,{plots["feature_importance"]}" alt="Feature Importance"/>',
                "Shows which features have the most impact on model predictions."
            )
        
        # SHAP analysis
        shap_html = ""
        if 'shap_summary' in plots:
            shap_html = self._section(
                "SHAP Value Analysis",
                f'<img src="data:image/png;base64,{plots["shap_summary"]}" alt="SHAP Summary"/>',
                "SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions using game theory."
            )
        
        # Classification plots
        classification_html = ""
        if 'confusion_matrix' in plots:
            classification_html += self._section(
                "Confusion Matrix",
                f'<img src="data:image/png;base64,{plots["confusion_matrix"]}" alt="Confusion Matrix"/>',
                "Shows prediction vs actual counts. Diagonal values = correct predictions."
            )
        if 'roc_curve' in plots:
            classification_html += self._section(
                "ROC Curve",
                f'<img src="data:image/png;base64,{plots["roc_curve"]}" alt="ROC Curve"/>',
                "Trade-off between true positive rate and false positive rate. AUC closer to 1.0 is better."
            )
        
        # Regression plots
        regression_html = ""
        if 'residuals' in plots:
            regression_html += self._section(
                "Residual Analysis",
                f'<img src="data:image/png;base64,{plots["residuals"]}" alt="Residuals"/>',
                "Left: Predicted vs Actual values. Right: Distribution of prediction errors."
            )
        
        # PCA analysis
        pca_html = ""
        if 'pca_variance' in plots or 'pca_scatter' in plots:
            pca_content = ""
            if 'pca_variance' in plots:
                pca_content += f'<img src="data:image/png;base64,{plots["pca_variance"]}" alt="PCA Variance"/>'
            if 'pca_scatter' in plots:
                pca_content += f'<img src="data:image/png;base64,{plots["pca_scatter"]}" alt="PCA Scatter"/>'
            pca_html = self._section(
                "PCA Analysis",
                pca_content,
                "Principal Component Analysis reveals data structure and variance distribution."
            )
        
        # Embedding plots (for multimodal)
        embedding_html = ""
        if 'embeddings_pca' in plots or 'embeddings_tsne' in plots:
            emb_content = ""
            if 'embeddings_pca' in plots:
                emb_content += f'<img src="data:image/png;base64,{plots["embeddings_pca"]}" alt="Embeddings PCA"/>'
            if 'embeddings_tsne' in plots:
                emb_content += f'<img src="data:image/png;base64,{plots["embeddings_tsne"]}" alt="Embeddings t-SNE"/>'
            embedding_html = self._section(
                "Embedding Visualization",
                emb_content,
                "Learned representations from the model projected to 2D."
            )
        
        return self._wrap_html(f'''
            <div class="header">
                <h1>Model Explainability Report</h1>
                <p><span class="badge badge-primary">{metrics.get('model_type', 'Unknown')}</span>
                   <span class="badge badge-success">{metrics.get('problem_type', 'Unknown')}</span></p>
            </div>
            
            {metrics_html}
            {importance_html}
            {shap_html}
            {classification_html}
            {regression_html}
            {pca_html}
            {embedding_html}
        ''', title="Model Report - Expert")
    
    def _build_timeseries_html(self, metrics: Dict, plots: Dict, mode: str) -> str:
        """Build time series specific report."""
        
        # Forecast plot
        forecast_html = ""
        if 'forecast' in plots:
            forecast_html = self._section(
                "Forecast Visualization",
                f'<img src="data:image/png;base64,{plots["forecast"]}" alt="Forecast"/>',
                "Predicted future values with uncertainty bands."
            )
        
        # Metrics
        metrics_html = self._format_metrics_cards(metrics)
        
        info_box = '''
        <div class="info-box">
            <h3>About Time Series Models</h3>
            <p>Time series forecasting models predict future values based on historical patterns.</p>
            <p><strong>Note:</strong> SHAP-based feature importance is not applicable to time series models.</p>
        </div>
        '''
        
        return self._wrap_html(f'''
            <div class="header">
                <h1>Time Series Forecast Report</h1>
                <p><span class="badge badge-primary">{metrics.get('model_type', 'Time Series')}</span>
                   <span class="badge badge-success">Forecasting</span></p>
            </div>
            
            {info_box}
            {metrics_html}
            {forecast_html}
        ''', title="Time Series Report")
    
    def _generate_insights(self, metrics: Dict) -> str:
        """Generate beginner-friendly insights from metrics."""
        insights = []
        
        problem_type = metrics.get('problem_type', 'unknown')
        
        if problem_type == 'classification':
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                if acc >= 0.9:
                    insights.append(f"<p>✓ Your model has excellent accuracy ({acc:.1%})</p>")
                elif acc >= 0.7:
                    insights.append(f"<p>✓ Your model has good accuracy ({acc:.1%})</p>")
                else:
                    insights.append(f"<p>⚠ Your model's accuracy could be improved ({acc:.1%})</p>")
            
            if 'roc_auc' in metrics:
                auc = metrics['roc_auc']
                if auc >= 0.9:
                    insights.append("<p>✓ The model is excellent at distinguishing between classes</p>")
                elif auc >= 0.7:
                    insights.append("<p>✓ The model has good discrimination ability</p>")
        
        elif problem_type == 'regression':
            if 'r2' in metrics:
                r2 = metrics['r2']
                if r2 >= 0.8:
                    insights.append(f"<p>✓ Your model explains {r2:.1%} of the variance in the data</p>")
                elif r2 >= 0.5:
                    insights.append(f"<p>✓ Your model has moderate explanatory power (R² = {r2:.2f})</p>")
        
        insights.append(f"<p>The model has {metrics.get('n_features', 'unknown')} features and was evaluated on {metrics.get('n_samples', 'unknown')} samples.</p>")
        
        return "\n".join(insights) if insights else "<p>Model analysis complete.</p>"
    
    def _format_metrics_cards(self, metrics: Dict) -> str:
        """Format metrics as card grid."""
        
        # Define which metrics to show
        metric_config = {
            'accuracy': ('Accuracy', lambda x: f"{x:.1%}"),
            'precision': ('Precision', lambda x: f"{x:.3f}"),
            'recall': ('Recall', lambda x: f"{x:.3f}"),
            'f1': ('F1 Score', lambda x: f"{x:.3f}"),
            'roc_auc': ('ROC AUC', lambda x: f"{x:.3f}"),
            'mae': ('MAE', lambda x: f"{x:.4f}"),
            'rmse': ('RMSE', lambda x: f"{x:.4f}"),
            'r2': ('R² Score', lambda x: f"{x:.4f}"),
            'n_features': ('Features', lambda x: str(x)),
            'n_samples': ('Samples', lambda x: str(x)),
        }
        
        cards = []
        for key, (label, formatter) in metric_config.items():
            if key in metrics and metrics[key] is not None:
                try:
                    value = formatter(metrics[key])
                    cards.append(f'''
                    <div class="metric-card">
                        <div class="value">{value}</div>
                        <div class="label">{label}</div>
                    </div>
                    ''')
                except Exception:
                    pass
        
        if not cards:
            return ""
        
        return f'''
        <div class="section">
            <h2>Model Performance</h2>
            <div class="metrics-grid">
                {"".join(cards)}
            </div>
        </div>
        '''
    
    def _section(self, title: str, content: str, description: str = "") -> str:
        """Create a report section."""
        desc_html = f'<p class="plot-description">{description}</p>' if description else ''
        return f'''
        <div class="section">
            <h2>{title}</h2>
            {desc_html}
            {content}
        </div>
        '''
    
    def _wrap_html(self, body: str, title: str = "Model Report") -> str:
        """Wrap content in full HTML document with styles."""
        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
        .plot-description {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 15px;
            font-style: italic;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin: 15px 0;
            display: block;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
            margin-right: 8px;
        }}
        .badge-primary {{ background: rgba(255,255,255,0.2); color: white; }}
        .badge-success {{ background: rgba(40,167,69,0.3); color: white; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        {body}
    </div>
</body>
</html>
'''
