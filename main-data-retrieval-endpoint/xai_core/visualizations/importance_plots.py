"""
Feature importance and SHAP visualization functions.
"""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from xai_core.utils import fig_to_base64

warnings.filterwarnings('ignore')

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
    color: str = "#667eea"
) -> Optional[str]:
    """
    Create feature importance bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to show
        color: Bar color
        
    Returns:
        Base64-encoded PNG string or None on error
    """
    if importance_df is None or importance_df.empty:
        return None
    
    try:
        # Limit to top N features
        df_plot = importance_df.nlargest(top_n, 'importance')
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.35)))
        
        # Create horizontal bar chart
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df_plot)))
        bars = ax.barh(
            range(len(df_plot)), 
            df_plot['importance'].values, 
            color=colors[::-1]
        )
        
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'].values)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, df_plot['importance'].values):
            ax.text(
                bar.get_width() + 0.001, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', 
                va='center', 
                fontsize=9
            )
        
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Feature importance plot failed: {e}")
        return None


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    plot_type: str = "bar"
) -> Optional[str]:
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        max_display: Maximum features to display
        plot_type: 'bar' or 'dot'
        
    Returns:
        Base64-encoded PNG string or None
    """
    if not SHAP_AVAILABLE:
        return None
    
    if shap_values is None:
        return None
    
    try:
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            plot_type=plot_type, 
            max_display=max_display, 
            show=False
        )
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"SHAP summary plot failed: {e}")
        return None


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    instance_idx: int = 0
) -> Optional[str]:
    """
    Create SHAP waterfall plot for a single instance.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        instance_idx: Index of instance to explain
        
    Returns:
        Base64-encoded PNG string or None
    """
    if not SHAP_AVAILABLE:
        return None
    
    if shap_values is None:
        return None
    
    try:
        fig = plt.figure(figsize=(10, 8))
        
        # Create Explanation object for waterfall
        if instance_idx < len(shap_values):
            explanation = shap.Explanation(
                values=shap_values[instance_idx],
                data=X.iloc[instance_idx].values,
                feature_names=list(X.columns)
            )
            shap.plots.waterfall(explanation, show=False)
            
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"SHAP waterfall plot failed: {e}")
        return None
