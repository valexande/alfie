"""
Explainer Service - Unified model explainability for AutoGluon and sklearn models.

For AutoGluon models:
- Feature importance via TabularPredictor.feature_importance()
- SHAP values via auto-selected explainer (Tree/Linear/Kernel)

For sklearn models:
- Permutation importance via sklearn.inspection
- Native feature importance for tree-based models
- Comprehensive performance metrics and visualizations
"""

from typing import Any, Dict, Optional, List, Tuple
import warnings
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import label_binarize

from xai_core.model_loader import ModelInfo
from xai_core.utils import safe_compute, fig_to_base64, ensure_numeric

# Optional imports with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP analysis will be skipped.")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

# Try to import AutoGluon EDA
try:
    from autogluon.eda import auto
    AUTOGLUON_EDA_AVAILABLE = True
except ImportError:
    AUTOGLUON_EDA_AVAILABLE = False

warnings.filterwarnings('ignore')


# =============================================================================
# SHAP EXPLAINER SELECTION
# =============================================================================

def select_shap_explainer(model: Any, X_sample: pd.DataFrame, model_type: str) -> Tuple[Any, str]:
    """
    Auto-select best SHAP explainer for model type.
    
    Args:
        model: The model to explain
        X_sample: Sample data for background
        model_type: Type of model ('tree_ensemble', 'xgboost', 'lightgbm', etc.)
        
    Returns:
        Tuple[explainer, explainer_name] or (None, None) if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        return None, None
    
    print(f"Selecting SHAP explainer for model type: {model_type}")
    
    # Tree-based models: Use TreeExplainer (fast, exact)
    if model_type in ['xgboost', 'lightgbm', 'catboost', 'tree_ensemble']:
        try:
            explainer = shap.TreeExplainer(model)
            print("  Using TreeExplainer (fast, exact)")
            return explainer, 'TreeExplainer'
        except Exception as e:
            print(f"  TreeExplainer failed: {e}, trying fallback")
    
    # Linear models: Use LinearExplainer
    if model_type == 'linear':
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            print("  Using LinearExplainer")
            return explainer, 'LinearExplainer'
        except Exception as e:
            print(f"  LinearExplainer failed: {e}, trying fallback")
    
    # Fallback: KernelExplainer (slow but universal)
    try:
        background = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
        
        # Create prediction function
        if hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
            
        explainer = shap.KernelExplainer(predict_fn, background)
        print("  Using KernelExplainer (slow but universal)")
        return explainer, 'KernelExplainer'
    except Exception as e:
        print(f"  KernelExplainer failed: {e}")
        return None, None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, classes: List) -> Optional[str]:
    """
    Generate ROC curve visualization.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        classes: List of class labels
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if len(classes) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            
            ax.plot(fpr, tpr, linewidth=2, color='#667eea', label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        else:
            # Multiclass: micro-average ROC
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] <= 1:
                return None
            
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            auc = roc_auc_score(y_bin, y_proba, average="micro", multi_class="ovr")
            
            ax.plot(fpr, tpr, linewidth=2, color='#667eea', label=f"Micro-avg AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"ROC curve plot failed: {e}")
        return None


def plot_pca_analysis(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate PCA variance and scatter plots.
    
    Args:
        X: Feature DataFrame
        y: Optional target Series for coloring
        
    Returns:
        Tuple of (variance_plot_base64, scatter_plot_base64)
    """
    try:
        X_numeric = ensure_numeric(X)
        
        if X_numeric.shape[1] < 2:
            print(f"Need at least 2 numeric features for PCA, found {X_numeric.shape[1]}")
            return None, None
        
        X_clean = X_numeric.dropna()
        if len(X_clean) < 3:
            print("Insufficient data rows for PCA after dropping NaNs")
            return None, None
        
        # Fit PCA
        n_components = min(10, X_clean.shape[1])
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_clean)
        var_ratio = pca.explained_variance_ratio_
        
        # Variance bar plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color="#667eea", alpha=0.8)
        ax1.set_xlabel("Principal Component", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.set_title("PCA Explained Variance", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(1, len(var_ratio) + 1))
        
        # Add value labels
        for bar, val in zip(bars, var_ratio):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        variance_plot = fig_to_base64(fig1)
        
        # 2D scatter plot
        scatter_plot = None
        if components.shape[1] >= 2:
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            
            if y is not None:
                y_aligned = y.loc[X_clean.index] if hasattr(y, 'loc') else y
                class_labels, class_ids = pd.factorize(y_aligned)
                scatter = ax2.scatter(components[:, 0], components[:, 1],
                                    c=class_labels, cmap="viridis", alpha=0.6,
                                    edgecolors='k', linewidth=0.5, s=50)
                
                # Create legend
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=scatter.cmap(scatter.norm(i)),
                                    markersize=8, label=str(label))
                          for i, label in enumerate(class_ids)]
                ax2.legend(handles=handles, title="Class", loc="best")
            else:
                ax2.scatter(components[:, 0], components[:, 1],
                          alpha=0.6, edgecolors='k', linewidth=0.5, s=50, color='#667eea')
            
            ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
            ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
            ax2.set_title("PCA Projection (PC1 vs PC2)", fontsize=14, fontweight="bold")
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            scatter_plot = fig_to_base64(fig2)
        
        return variance_plot, scatter_plot
    except Exception as e:
        print(f"PCA analysis failed: {e}")
        return None, None


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> Optional[str]:
    """
    Generate residual analysis plots for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        residuals = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs Actual
        ax1.scatter(y_pred, y_true, alpha=0.5, edgecolors='k', linewidth=0.5, color='#667eea')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('Actual', fontsize=12)
        ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Residual distribution
        ax2.hist(residuals, bins=30, color='#764ba2', alpha=0.7, edgecolor='k')
        ax2.axvline(0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Residual plot failed: {e}")
        return None


def plot_shap_summary(explainer: Any, X_sample: pd.DataFrame, max_samples: int = 300) -> Optional[str]:
    """
    Generate SHAP summary plot.
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data to explain
        max_samples: Maximum samples to plot
        
    Returns:
        Base64-encoded PNG string or None
    """
    if not SHAP_AVAILABLE or explainer is None:
        return None
    
    try:
        X_plot = X_sample.sample(n=min(max_samples, len(X_sample)), random_state=42)
        shap_values = explainer.shap_values(X_plot)
        
        # Handle multi-class output
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = shap_values[0]
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_plot, plot_type="bar", max_display=20, show=False)
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"SHAP summary plot failed: {e}")
        return None


class ExplainerService:
    """
    Unified explainability service for AutoGluon and sklearn models.
    
    Example:
        >>> model_info = load_model("my_model")
        >>> service = ExplainerService(model_info, X_test, y_test, predictor=predictor)
        >>> html_report = service.generate_html_report(mode='expert')
    """
    
    def __init__(
        self, 
        model_info: ModelInfo, 
        X: pd.DataFrame, 
        y: pd.Series,
        predictor: Any = None,
        max_samples: int = 1000,
        label: Optional[str] = None
    ):
        """
        Initialize explainer service.
        
        Args:
            model_info: ModelInfo from load_model()
            X: Feature DataFrame
            y: Target Series
            predictor: Raw AutoGluon predictor (for native EDA functions)
            max_samples: Max samples for SHAP computation (for performance)
            label: Target column name
        """
        self.model_info = model_info
        self.X = X
        self.y = y
        self.predictor = predictor
        self.max_samples = max_samples
        self.label = label or (y.name if hasattr(y, 'name') and y.name else 'target')
        
        # Build full DataFrame with target
        self.data = X.copy()
        self.data[self.label] = y.values
        
        # Cache for computed values
        self._feature_importance = None
        self._shap_values = None
    
    @property
    def model(self) -> Any:
        """Get the raw model/predictor."""
        return self.predictor if self.predictor else self.model_info.model
    
    @property
    def is_classification(self) -> bool:
        """Check if model is a classifier."""
        return self.model_info.problem_type == 'classification'
    
    @property
    def is_timeseries(self) -> bool:
        """Check if model is a time series forecaster."""
        return self.model_info.problem_type == 'forecasting'
    
    @property
    def is_autogluon(self) -> bool:
        """Check if model is an AutoGluon predictor."""
        return self.model_info.is_autogluon
    
    def generate_html_report(self, mode: str = 'expert') -> str:
        """
        Generate HTML explainability report.
        
        Args:
            mode: 'beginner' or 'expert' - controls detail level
            
        Returns:
            HTML string with full report
        """
        # Time series needs custom report
        if self.is_timeseries:
            return self._generate_timeseries_report(mode)
        
        # Generate report using native AutoGluon methods + SHAP
        if self.is_autogluon:
            return self._generate_autogluon_report(mode)
        else:
            return self._generate_generic_report(mode)
    
    def _generate_autogluon_report(self, mode: str) -> str:
        """Generate report using AutoGluon's native feature_importance + SHAP + advanced visualizations."""
        
        # Sample data for performance
        data_sample = self._sample_data_df()
        
        # Check for version mismatch warning
        version_warning = ""
        if self.model_info.has_version_mismatch:
            version_warning = f"""
            <div class="warning-box" style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 15px 0;">
                <h4 style="color: #856404; margin-top: 0;">⚠️ AutoGluon Version Mismatch Detected</h4>
                <p><strong>Model trained with:</strong> AutoGluon {self.model_info.model_version or 'unknown'}</p>
                <p><strong>Currently installed:</strong> AutoGluon {self.model_info.current_version or 'unknown'}</p>
                <p>Some features (SHAP analysis, predictions) may be unavailable. Feature importance from training data is still shown.</p>
            </div>
            """
        
        # Get feature importance using native method
        feature_importance_df = self.get_feature_importance()
        feature_importance_plot = self._create_feature_importance_plot(feature_importance_df)
        
        # Get SHAP explanations for sample rows (only if versions compatible)
        if self.model_info.version_compatible:
            shap_explanations = self._get_shap_explanations(data_sample, mode)
        else:
            shap_explanations = self._generate_version_mismatch_message("SHAP analysis")
        
        # Get model metrics
        metrics = self.get_metrics()
        metrics_html = self._format_metrics_html(metrics)
        
        # Get predictions vs actual (for evaluation)
        if self.model_info.version_compatible:
            predictions_plot = self._create_predictions_plot(data_sample)
        else:
            predictions_plot = self._generate_version_mismatch_message("Prediction analysis")
        
        # Generate ROC curve for classification
        roc_curve_section = ""
        if self.is_classification and self.model_info.version_compatible:
            roc_curve_html = self._create_roc_curve_plot(data_sample)
            if roc_curve_html:
                roc_curve_section = f"""
                <div class="section">
                    <h2>ROC Curve Analysis</h2>
                    <p>The ROC curve shows the trade-off between true positive rate and false positive rate. AUC closer to 1.0 indicates better performance.</p>
                    {roc_curve_html}
                </div>
                """
        
        # Generate PCA analysis
        pca_section = ""
        if mode == 'expert':
            pca_variance, _pca_scatter = plot_pca_analysis(self.X, self.y)
            if pca_variance:
                pca_content = ""
                if pca_variance:
                    pca_content += f'<img src="data:image/png;base64,{pca_variance}" alt="PCA Variance"/>'
                pca_section = f"""
                <div class="section">
                    <h2>PCA Analysis</h2>
                    <p>Principal Component Analysis reveals the variance distribution in your data.</p>
                    {pca_content}
                </div>
                """
        
        # Generate residual plot for regression
        residual_section = ""
        if not self.is_classification and self.model_info.version_compatible and mode == 'expert':
            residual_html = self._create_residual_plot(data_sample)
            if residual_html:
                residual_section = f"""
                <div class="section">
                    <h2>Residual Analysis</h2>
                    <p>Residual analysis helps identify patterns in prediction errors and model performance.</p>
                    {residual_html}
                </div>
                """
        
        # Build HTML report
        beginner_section = ""
        expert_section = ""
        
        if mode == 'beginner':
            beginner_section = f"""
            <div class="section">
                <h2>Key Insights</h2>
                <div class="info-box">
                    <p>This report shows which features are most important for the model's predictions.</p>
                    <p>Features at the top of the importance chart have the biggest impact on predictions.</p>
                </div>
            </div>
            {roc_curve_section}
            """
        else:
            expert_section = f"""
            <div class="section">
                <h2>SHAP Value Analysis</h2>
                <p>SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions.</p>
                {shap_explanations}
            </div>
            
            {roc_curve_section}
            
            <div class="section">
                <h2>Prediction Analysis</h2>
                {predictions_plot}
            </div>
            
            {residual_section}
            
            {pca_section}
            """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Model Explainability Report</title>
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
                .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
                .header p {{ opacity: 0.9; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{
                    color: #667eea;
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
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
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                .table th, .table td {{
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                .table th {{ background: #f8f9fa; font-weight: 600; }}
                .table tr:hover {{ background: #f5f5f5; }}
                img {{ max-width: 100%; border-radius: 8px; margin: 15px 0; }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 4px;
                }}
                .warning-box {{
                    background: #fff3e0;
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
                }}
                .badge-primary {{ background: #667eea; color: white; }}
                .badge-success {{ background: #28a745; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Explainability Report</h1>
                    <p><span class="badge badge-primary">{self.model_info.model_type}</span>
                       <span class="badge badge-success">{self.model_info.problem_type}</span>
                       {' <span class="badge badge-primary">AutoGluon</span>' if self.is_autogluon else ''}
                       {f' <span class="badge" style="background:#ffc107;color:#856404;">v{self.model_info.model_version}</span>' if self.model_info.model_version else ''}</p>
                </div>
                
                {version_warning}
                
                {beginner_section}
                
                <div class="section">
                    <h2>Model Performance</h2>
                    {metrics_html}
                </div>
                
                <div class="section">
                    <h2>Feature Importance</h2>
                    <p>Shows which features have the most impact on model predictions (permutation importance).</p>
                    {feature_importance_plot}
                </div>
                
                {expert_section}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_generic_report(self, mode: str) -> str:
        """Generate report for non-AutoGluon models using sklearn metrics + advanced visualizations."""
        
        # Sample data
        data_sample = self._sample_data_df()
        
        # Get feature importance
        feature_importance_df = self.get_feature_importance()
        feature_importance_plot = self._create_feature_importance_plot(feature_importance_df)
        
        # Get model metrics
        metrics = self.get_metrics()
        metrics_html = self._format_metrics_html(metrics)
        
        # Create predictions plot
        predictions_plot = self._create_predictions_plot(data_sample)
        
        # SHAP analysis with smart explainer selection
        shap_section = ""
        if mode == 'expert' and SHAP_AVAILABLE:
            shap_html = self._get_shap_explanations(data_sample, mode)
            if shap_html and 'error' not in shap_html.lower():
                shap_section = f"""
                <div class="section">
                    <h2>SHAP Value Analysis</h2>
                    <p>SHAP values show how each feature contributes to predictions using game theory (Shapley values).</p>
                    {shap_html}
                </div>
                """
        
        # ROC curve for classification
        roc_section = ""
        if self.is_classification:
            roc_html = self._create_roc_curve_plot(data_sample)
            if roc_html:
                roc_section = f"""
                <div class="section">
                    <h2>ROC Curve Analysis</h2>
                    <p>The ROC curve shows model discrimination ability. AUC closer to 1.0 indicates better performance.</p>
                    {roc_html}
                </div>
                """
        
        # Residual analysis for regression
        residual_section = ""
        if not self.is_classification and mode == 'expert':
            residual_html = self._create_residual_plot(data_sample)
            if residual_html:
                residual_section = f"""
                <div class="section">
                    <h2>Residual Analysis</h2>
                    <p>Residual plots help identify patterns in prediction errors and potential model issues.</p>
                    {residual_html}
                </div>
                """
        
        # PCA analysis
        pca_section = ""
        if mode == 'expert':
            pca_variance, _pca_scatter = plot_pca_analysis(self.X, self.y)
            if pca_variance:
                pca_content = ""
                if pca_variance:
                    pca_content += f'<img src="data:image/png;base64,{pca_variance}" alt="PCA Variance"/>'
                pca_section = f"""
                <div class="section">
                    <h2>PCA Analysis</h2>
                    <p>Principal Component Analysis reveals the variance distribution in your data.</p>
                    {pca_content}
                </div>
                """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Model Explainability Report</title>
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
                .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{
                    color: #667eea;
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
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
                }}
                img {{ max-width: 100%; border-radius: 8px; margin: 15px 0; }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Explainability Report</h1>
                    <p><strong>Model Type:</strong> {self.model_info.model_type} | 
                       <strong>Problem:</strong> {self.model_info.problem_type}</p>
                </div>
                
                <div class="section">
                    <h2>Model Performance</h2>
                    {metrics_html}
                </div>
                
                <div class="section">
                    <h2>Feature Importance</h2>
                    {feature_importance_plot}
                </div>
                
                {roc_section}
                
                {shap_section}
                
                {'<div class="section"><h2>Predictions Analysis</h2>' + predictions_plot + '</div>' if mode == 'expert' else ''}
                
                {residual_section}
                
                {pca_section}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance using native AutoGluon method or permutation importance.
        
        Returns DataFrame with columns: feature, importance
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        # Try AutoGluon native feature_importance first
        if self.is_autogluon and self.predictor is not None:
            try:
                # Use native feature_importance method
                importance = self.predictor.feature_importance(
                    data=self.data,
                    subsample_size=min(self.max_samples, len(self.data)),
                    num_shuffle_sets=5,
                    silent=True
                )
                if isinstance(importance, pd.DataFrame):
                    # Rename columns if needed
                    if 'importance' in importance.columns:
                        self._feature_importance = importance.reset_index()
                        self._feature_importance.columns = ['feature', 'importance'] + list(self._feature_importance.columns[2:])
                    else:
                        importance = importance.reset_index()
                        importance.columns = ['feature'] + list(importance.columns[1:])
                        if len(importance.columns) > 1:
                            importance['importance'] = importance.iloc[:, 1]
                        self._feature_importance = importance[['feature', 'importance']]
                    return self._feature_importance
            except Exception as e:
                print(f"AutoGluon feature_importance failed: {e}")
        
        # Fall back to permutation importance for sklearn models
        try:
            from sklearn.inspection import permutation_importance
            
            model = self.model_info.model
            X_sample, y_sample = self._sample_data()
            
            result = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
            
            self._feature_importance = importance_df
            return importance_df
            
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            return None
    
    def _get_shap_explanations(self, data_sample: pd.DataFrame, mode: str) -> str:
        """Get SHAP explanations using auto-selected explainer."""
        
        if not SHAP_AVAILABLE:
            return "<p>SHAP library not available. Install with: pip install shap</p>"
        
        try:
            # Select rows to explain
            n_explain = min(100, len(data_sample)) if mode == 'expert' else min(50, len(data_sample))
            X_explain = data_sample.drop(columns=[self.label], errors='ignore').head(n_explain)
            
            # Test if predictor can make predictions (version compatibility check)
            if self.is_autogluon and self.predictor is not None:
                try:
                    test_pred = self.predictor.predict(X_explain.head(1))
                except AttributeError as ve:
                    if 'passthrough' in str(ve) or 'AsTypeFeatureGenerator' in str(ve):
                        return self._generate_version_mismatch_message("SHAP analysis")
                    raise
            
            # Determine model type for smart explainer selection
            model_type = self.model_info.model_type if self.model_info else 'unknown'
            
            # Get the model to explain
            if self.is_autogluon and self.predictor is not None:
                # For AutoGluon, create a wrapper function
                def predict_fn(X):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=self.X.columns)
                    result = self.predictor.predict(X)
                    return result.values if hasattr(result, 'values') else np.array(result)
                
                # Use KernelExplainer for AutoGluon (can't use TreeExplainer directly)
                background = self.X.sample(min(50, len(self.X)), random_state=42)
                explainer = shap.KernelExplainer(predict_fn, background)
                explainer_name = 'KernelExplainer'
            else:
                # For sklearn models, use smart selection
                model = self.model_info.model if self.model_info else None
                if model is None:
                    return "<p>No model available for SHAP analysis.</p>"
                
                X_background = ensure_numeric(self.X.sample(min(100, len(self.X)), random_state=42))
                explainer, explainer_name = select_shap_explainer(model, X_background, model_type)
                
                if explainer is None:
                    return "<p>Could not create SHAP explainer for this model type.</p>"
            
            # Get SHAP values
            X_explain_numeric = ensure_numeric(X_explain)
            shap_values = explainer.shap_values(X_explain_numeric)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]
            
            # Create summary plot
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_explain_numeric, plot_type="bar", max_display=20, show=False)
            plt.title(f"SHAP Feature Importance ({explainer_name})", fontsize=14, fontweight="bold", pad=20)
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close('all')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f'<img src="data:image/png;base64,{img_base64}" alt="SHAP Summary"/>'
            
        except Exception as e:
            error_msg = str(e)
            if 'passthrough' in error_msg or 'AsTypeFeatureGenerator' in error_msg:
                return self._generate_version_mismatch_message("SHAP analysis")
            print(f"SHAP analysis error: {e}")
            return f"<p>Could not generate SHAP plot: {e}</p>"
    
    def _generate_version_mismatch_message(self, analysis_type: str) -> str:
        """Generate informative message about AutoGluon version mismatch."""
        return f"""
        <div class="warning-box" style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="color: #856404; margin-top: 0;">⚠️ AutoGluon Version Mismatch</h4>
            <p><strong>{analysis_type}</strong> is unavailable due to a version compatibility issue.</p>
            <p>The model was trained with a different AutoGluon version than the one currently installed.</p>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Retrain the model with the current AutoGluon version</li>
                <li>Install the AutoGluon version that matches the model (check metadata.json)</li>
            </ul>
            <p><em>Feature importance from model training is still available above.</em></p>
        </div>
        """
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Returns dict with metrics appropriate for model type.
        """
        metrics = {
            'model_type': self.model_info.model_type,
            'problem_type': self.model_info.problem_type,
            'is_autogluon': self.model_info.is_autogluon,
            'n_features': len(self.X.columns),
            'n_samples': len(self.X),
        }
        
        if self.is_timeseries:
            return metrics
        
        # Get predictions
        try:
            if self.predictor is not None:
                y_pred = self.predictor.predict(self.X)
            else:
                y_pred = self.model_info.model.predict(self.X)
        except AttributeError as ve:
            if 'passthrough' in str(ve) or 'AsTypeFeatureGenerator' in str(ve):
                metrics['warning'] = 'AutoGluon version mismatch - metrics unavailable'
            else:
                print(f"Prediction failed: {ve}")
            return metrics
        except Exception as e:
            print(f"Prediction failed: {e}")
            return metrics
        
        # Calculate metrics based on problem type
        if self.is_classification:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            try:
                metrics['accuracy'] = safe_compute(lambda: round(accuracy_score(self.y, y_pred), 4))
                metrics['precision'] = safe_compute(lambda: round(precision_score(self.y, y_pred, average='weighted', zero_division=0), 4))
                metrics['recall'] = safe_compute(lambda: round(recall_score(self.y, y_pred, average='weighted', zero_division=0), 4))
                metrics['f1'] = safe_compute(lambda: round(f1_score(self.y, y_pred, average='weighted', zero_division=0), 4))
            except Exception as e:
                print(f"Classification metrics error: {e}")
        else:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            try:
                metrics['r2'] = safe_compute(lambda: round(r2_score(self.y, y_pred), 4))
                metrics['mae'] = safe_compute(lambda: round(mean_absolute_error(self.y, y_pred), 4))
                metrics['rmse'] = safe_compute(lambda: round(np.sqrt(mean_squared_error(self.y, y_pred)), 4))
            except Exception as e:
                print(f"Regression metrics error: {e}")
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.get_metrics()
    
    def _sample_data(self) -> tuple:
        """Sample X and y data for faster computation."""
        if len(self.X) <= self.max_samples:
            return self.X, self.y
        
        indices = np.random.choice(len(self.X), self.max_samples, replace=False)
        return self.X.iloc[indices], self.y.iloc[indices]
    
    def _sample_data_df(self) -> pd.DataFrame:
        """Sample full DataFrame for faster computation."""
        if len(self.data) <= self.max_samples:
            return self.data
        
        return self.data.sample(n=self.max_samples, random_state=42)
    
    def _create_feature_importance_plot(self, importance_df: Optional[pd.DataFrame]) -> str:
        """Create feature importance bar chart as base64 image."""
        
        if importance_df is None or importance_df.empty:
            return "<p>Feature importance not available.</p>"
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        # Sort and limit to top 20 features
        df = importance_df.nlargest(20, 'importance')
        
        # Create horizontal bar chart
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df)))
        bars = ax.barh(df['feature'], df['importance'], color=colors[::-1])
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for bar, val in zip(bars, df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance"/>'
    
    def _create_predictions_plot(self, data_sample: pd.DataFrame) -> str:
        """Create predictions vs actual plot."""
        
        try:
            X_sample = data_sample.drop(columns=[self.label])
            y_actual = data_sample[self.label]
            
            # Try to get predictions
            try:
                if self.predictor is not None:
                    y_pred = self.predictor.predict(X_sample)
                else:
                    y_pred = self.model_info.model.predict(X_sample)
            except AttributeError as ve:
                if 'passthrough' in str(ve) or 'AsTypeFeatureGenerator' in str(ve):
                    return self._generate_version_mismatch_message("Prediction analysis")
                raise
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if self.is_classification:
                # Confusion matrix for classification
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                cm = confusion_matrix(y_actual, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            else:
                # Scatter plot for regression
                ax.scatter(y_actual, y_pred, alpha=0.5, edgecolors='none')
                
                # Add perfect prediction line
                min_val = min(y_actual.min(), y_pred.min())
                max_val = max(y_actual.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual', fontsize=12)
                ax.set_ylabel('Predicted', fontsize=12)
                ax.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
                ax.legend()
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f'<img src="data:image/png;base64,{img_base64}" alt="Predictions"/>'
            
        except Exception as e:
            error_msg = str(e)
            if 'passthrough' in error_msg or 'AsTypeFeatureGenerator' in error_msg:
                return self._generate_version_mismatch_message("Prediction analysis")
            return f"<p>Could not create predictions plot: {e}</p>"
    
    def _create_roc_curve_plot(self, data_sample: pd.DataFrame) -> Optional[str]:
        """Create ROC curve plot for classification models."""
        try:
            X_sample = data_sample.drop(columns=[self.label], errors='ignore')
            y_actual = data_sample[self.label] if self.label in data_sample.columns else self.y
            
            # Get probability predictions
            if self.predictor is not None:
                y_proba_df = self.predictor.predict_proba(X_sample)
                if isinstance(y_proba_df, pd.DataFrame):
                    classes = list(y_proba_df.columns)
                    y_proba = y_proba_df.values
                else:
                    return None
            elif hasattr(self.model_info.model, 'predict_proba'):
                X_numeric = ensure_numeric(X_sample)
                y_proba = self.model_info.model.predict_proba(X_numeric)
                classes = list(np.unique(y_actual))
            else:
                return None
            
            # Generate ROC curve
            return plot_roc_curve(y_actual, y_proba, classes)
            
        except Exception as e:
            print(f"ROC curve generation failed: {e}")
            return None
    
    def _create_residual_plot(self, data_sample: pd.DataFrame) -> Optional[str]:
        """Create residual analysis plot for regression models."""
        try:
            X_sample = data_sample.drop(columns=[self.label], errors='ignore')
            y_actual = data_sample[self.label] if self.label in data_sample.columns else self.y
            
            # Get predictions
            if self.predictor is not None:
                y_pred = self.predictor.predict(X_sample)
            else:
                X_numeric = ensure_numeric(X_sample)
                y_pred = self.model_info.model.predict(X_numeric)
            
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            # Generate residual plot
            return plot_residuals(y_actual, y_pred)
            
        except Exception as e:
            print(f"Residual plot generation failed: {e}")
            return None
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML cards."""
        
        # Filter out non-numeric/non-metric keys
        display_metrics = {}
        metric_labels = {
            'accuracy': 'Accuracy',
            'precision': 'Precision', 
            'recall': 'Recall',
            'f1': 'F1 Score',
            'r2': 'R² Score',
            'mae': 'Mean Abs Error',
            'rmse': 'Root MSE',
            'n_features': 'Features',
            'n_samples': 'Samples'
        }
        
        for key, label in metric_labels.items():
            if key in metrics and metrics[key] is not None:
                display_metrics[label] = metrics[key]
        
        if not display_metrics:
            return "<p>No metrics available.</p>"
        
        cards_html = '<div class="metrics-grid">'
        for label, value in display_metrics.items():
            if isinstance(value, float):
                formatted = f"{value:.4f}" if value < 1 else f"{value:.2f}"
            else:
                formatted = str(value)
            cards_html += f'''
            <div class="metric-card">
                <div class="value">{formatted}</div>
                <div class="label">{label}</div>
            </div>
            '''
        cards_html += '</div>'
        
        return cards_html
    
    def _generate_timeseries_report(self, mode: str) -> str:
        """Generate custom report for time series models."""
        
        # Get forecasts
        forecasts = safe_compute(lambda: self.predictor.predict(self.X) if self.predictor else None)
        
        # Generate forecast plot
        forecast_plot = ""
        if forecasts is not None:
            forecast_plot = self._create_forecast_plot(forecasts)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Time Series Forecast Report</title>
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
                .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{
                    color: #667eea;
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                img {{ max-width: 100%; border-radius: 8px; margin: 15px 0; }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Time Series Forecast Report</h1>
                    <p><strong>Model Type:</strong> {self.model_info.model_type}</p>
                </div>
                
                <div class="info-box">
                    <p><strong>Note:</strong> SHAP-based explainability is not applicable to time series forecasting models. 
                    This report shows forecast outputs and model performance.</p>
                </div>
                
                <div class="section">
                    <h2>Forecast Visualization</h2>
                    {forecast_plot}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def generate_eda_report(self) -> str:
        """
        Generate dataset health report using AutoGluon EDA concepts.
        """
        if not AUTOGLUON_EDA_AVAILABLE:
            warnings.warn("AutoGluon EDA not available, falling back to standard analysis")

        try:
            # 1. Dataset Overview
            desc_df = self.X.describe()
            desc = desc_df.to_html(classes='table')
            
            # 2. Missing Values Analysis
            missing_plot = ""
            try:
                null_counts = self.X.isnull().sum()
                if null_counts.sum() > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    null_counts[null_counts > 0].plot(kind='bar', ax=ax, color='#667eea')
                    ax.set_title("Missing Values per Column")
                    ax.set_ylabel("Count")
                    plt.tight_layout()
                    missing_plot = fig_to_base64(fig)
                    missing_plot = f'<img src="data:image/png;base64,{missing_plot}" alt="Missing Values"/>'
            except Exception as e:
                print(f"Missing analysis failed: {e}")

            # 3. Correlation Analysis
            corr_plot = ""
            try:
                numeric_df = self.X.select_dtypes(include=[np.number])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.2f')
                    ax.set_title("Feature Correlations")
                    plt.tight_layout()
                    corr_plot = fig_to_base64(fig)
                    corr_plot = f'<img src="data:image/png;base64,{corr_plot}" alt="Correlation Matrix"/>'
            except Exception as e:
                print(f"Correlation analysis failed: {e}")

            # 4. Target Distribution (if available)
            target_plot = ""
            try:
                if self.y is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if self.is_classification:
                        self.y.value_counts().plot(kind='bar', ax=ax, color='#764ba2')
                        ax.set_title("Target Class Distribution")
                    else:
                        sns.histplot(self.y, kde=True, ax=ax, color='#764ba2')
                        ax.set_title("Target Value Distribution")
                    plt.tight_layout()
                    target_plot = fig_to_base64(fig)
                    target_plot = f'<img src="data:image/png;base64,{target_plot}" alt="Target Distribution"/>'
            except Exception as e:
                print(f"Target analysis failed: {e}")

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dataset Analysis Report</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background: #f5f5f5;
                        padding: 40px;
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
                        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }}
                    .section {{ margin: 40px 0; }}
                    h1 {{ margin: 0; }}
                    h2 {{ 
                        color: #11998e;
                        border-bottom: 2px solid #11998e;
                        padding-bottom: 10px;
                        margin-bottom: 20px;
                    }}
                    .table {{ 
                        width: 100%; 
                        border-collapse: collapse; 
                        display: block;
                        overflow-x: auto;
                    }}
                    .table th, .table td {{ 
                        padding: 12px; 
                        border: 1px solid #ddd; 
                        text-align: right;
                    }}
                    .table th {{ background: #f8f9fa; text-align: center; }}
                    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Dataset Analysis Report</h1>
                        <p>Detailed analysis of input data distribution and health.</p>
                    </div>
                    
                    <div class="section">
                        <h2>Statistical Summary</h2>
                        {desc}
                    </div>

                    <div class="section">
                        <h2>Target Distribution</h2>
                        {target_plot if target_plot else "<p>Target variable not available.</p>"}
                    </div>

                    <div class="section">
                        <h2>Correlation Analysis</h2>
                        {corr_plot if corr_plot else "<p>Insufficient numeric data for correlation analysis.</p>"}
                    </div>

                    <div class="section">
                        <h2>Missing Values</h2>
                        {missing_plot if missing_plot else "<p>No missing values detected.</p>"}
                    </div>
                </div>
            </body>
            </html>
            """
            return html

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"<h2>EDA Generation Failed</h2><pre>{str(e)}</pre>"
    
    def _create_forecast_plot(self, forecasts: pd.DataFrame) -> str:
        """Create forecast visualization as base64 image."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean forecast if available
        if 'mean' in forecasts.columns:
            ax.plot(forecasts.index, forecasts['mean'], 'b-', linewidth=2, label='Forecast')
        else:
            # Plot first numeric column
            numeric_cols = forecasts.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.plot(forecasts.index, forecasts[numeric_cols[0]], 'b-', linewidth=2, label='Forecast')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Time Series Forecast', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_base64}" alt="Forecast Plot"/>'
