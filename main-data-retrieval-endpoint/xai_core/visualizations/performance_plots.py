"""
Performance visualization functions for classification and regression.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

from xai_core.utils import fig_to_base64

warnings.filterwarnings('ignore')


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: Optional[List] = None,
    title: str = "Confusion Matrix"
) -> Optional[str]:
    """
    Create confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class labels
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        if classes is None:
            classes = np.unique(y_true)
        
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=classes
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")
        return None


def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: Optional[List] = None,
    title: str = "ROC Curve"
) -> Optional[str]:
    """
    Create ROC curve visualization.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        classes: List of class labels
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        if classes is None:
            classes = np.unique(y_true)
        
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
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"ROC curve plot failed: {e}")
        return None


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Residual Analysis"
) -> Optional[str]:
    """
    Create residual analysis plots for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs Actual
        ax1.scatter(y_pred, y_true, alpha=0.5, edgecolors='k', linewidth=0.5, color='#667eea')
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
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
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Residual plot failed: {e}")
        return None


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual"
) -> Optional[str]:
    """
    Create scatter plot of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5, color='#667eea')
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Predictions plot failed: {e}")
        return None
