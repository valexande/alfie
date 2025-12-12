#!/usr/bin/env python3
"""
Universal Model Explainability Flask API

Supports:
- AutoGluon predictors (TabularPredictor, MultiModalPredictor, TimeSeriesPredictor)
- Generic pickled models (sklearn, XGBoost, LightGBM, CatBoost, etc.)

Auto-detects model type and applies appropriate explainability techniques.
Generates comprehensive HTML reports with metrics, visualizations, and interpretations.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, request
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# Windows compatibility fix for PosixPath in AutoGluon models
# AutoGluon models trained on Linux contain PosixPath objects that can't be instantiated on Windows
if sys.platform == 'win32':
    from pathlib import WindowsPath
    import pathlib
    
    # Create a compatibility class that allows PosixPath to work on Windows
    # This wraps WindowsPath but accepts Posix-style paths
    class CompatiblePosixPath(WindowsPath):
        """Windows-compatible PosixPath for loading Linux-trained AutoGluon models."""
        def __new__(cls, *args, **kwargs):
            if args:
                # Convert forward slashes to backslashes for Windows
                path_str = str(args[0]).replace('/', '\\')
                # Remove leading slash if present (absolute path)
                if path_str.startswith('\\') and len(path_str) > 1 and path_str[1] != '\\':
                    path_str = path_str[1:]
                args = (path_str,) + args[1:]
            return WindowsPath.__new__(WindowsPath, *args, **kwargs)
    
    # Patch pathlib.PosixPath to use our compatible version
    # This allows unpickling of models with PosixPath objects
    pathlib.PosixPath = CompatiblePosixPath

# Optional imports with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP analysis will be skipped.")

try:
    from autogluon.tabular import TabularPredictor
    TABULAR_AVAILABLE = True
except ImportError:
    TABULAR_AVAILABLE = False
    print("Warning: AutoGluon Tabular not available.")

try:
    from autogluon.multimodal import MultiModalPredictor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Warning: AutoGluon MultiModal not available.")

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    print("Warning: AutoGluon TimeSeries not available.")

try:
    from sklearn.inspection import permutation_importance, partial_dependence
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False
    print("Warning: sklearn.inspection not available.")

# Flask app
app = Flask(__name__)

# Constants
DEFAULT_MAX_SHAP_SAMPLES = 300
DEFAULT_MAX_EMBEDDING_SAMPLES = 1000
DEFAULT_PORT = 5005


# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def safe_compute(func, error_list: List[str], default_value=None, error_prefix="Operation"):
    """Execute function with error handling and logging."""
    try:
        return func()
    except Exception as e:
        error_msg = f"{error_prefix}: {str(e)}"
        error_list.append(error_msg)
        print(f"Warning: {error_msg}")
        return default_value


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detect likely target column from common names."""
    common_names = [
        'target', 'label', 'class', 'y', 'outcome',
        'prediction', 'alert', 'target_variable'
    ]

    for name in common_names:
        if name in df.columns:
            return name

    # Fallback: last column
    return df.columns[-1] if len(df.columns) > 0 else None


# =============================================================================
# SECTION 2: MODEL TYPE DETECTION
# =============================================================================

def detect_model_type(model) -> Tuple[str, str]:
    """
    Detect model type for appropriate explainability selection.

    Returns:
        Tuple[str, str]: (primary_type, sub_type)
        - primary_type: 'autogluon' or 'generic'
        - sub_type: specific model type
    """
    class_name = type(model).__name__
    module_path = type(model).__module__

    print(f"Detecting model type: class={class_name}, module={module_path}")

    # Check AutoGluon first
    if 'TabularPredictor' in class_name or 'autogluon.tabular' in module_path:
        return 'autogluon', 'tabular'
    elif 'MultiModalPredictor' in class_name or 'autogluon.multimodal' in module_path:
        return 'autogluon', 'multimodal'
    elif 'TimeSeriesPredictor' in class_name or 'autogluon.timeseries' in module_path:
        return 'autogluon', 'timeseries'

    # Check generic models - tree-based
    if any(x in class_name for x in ['XGB', 'xgboost', 'XGBoost']):
        return 'generic', 'xgboost'
    elif any(x in class_name for x in ['LGB', 'lightgbm', 'LightGBM']):
        return 'generic', 'lightgbm'
    elif any(x in class_name for x in ['RandomForest', 'GradientBoosting', 'DecisionTree', 'ExtraTree']):
        return 'generic', 'sklearn_tree'
    elif 'CatBoost' in class_name:
        return 'generic', 'catboost'

    # Linear models
    elif any(x in class_name for x in ['Linear', 'Logistic', 'Ridge', 'Lasso', 'ElasticNet', 'SGD']):
        return 'generic', 'linear'

    # Ensemble models
    elif any(x in class_name for x in ['Voting', 'Stacking', 'Bagging', 'AdaBoost']):
        return 'generic', 'ensemble'

    # Neural networks
    elif any(x in class_name for x in ['Sequential', 'Model', 'Keras', 'torch', 'pytorch']):
        return 'generic', 'neural_network'

    # Other models
    elif any(x in class_name for x in ['SVC', 'SVR', 'SVM']):
        return 'generic', 'svm'
    elif any(x in class_name for x in ['KNeighbors', 'KNN']):
        return 'generic', 'knn'
    elif any(x in class_name for x in ['NaiveBayes', 'GaussianNB', 'MultinomialNB']):
        return 'generic', 'naive_bayes'

    return 'generic', 'unknown'


# =============================================================================
# SECTION 3: MODEL LOADING
# =============================================================================

def load_model(model_path: str) -> Tuple[Any, str, str, List[str]]:
    """
    Universal model loader for AutoGluon predictors and generic pickled models.

    Args:
        model_path: Path to predictor directory or pickle file

    Returns:
        Tuple[model, primary_type, sub_type, errors]
    """
    path = Path(model_path).expanduser().resolve()
    errors = []

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    # Strategy 1: Try loading as AutoGluon directory
    if path.is_dir():
        # Try TabularPredictor
        if TABULAR_AVAILABLE:
            try:
                print(f"Attempting to load as TabularPredictor from {path}")
                model = TabularPredictor.load(
                    str(path),
                    require_py_version_match=False,
                    require_version_match=False  # Allow version mismatch
                )
                primary_type, sub_type = detect_model_type(model)
                print(f"Successfully loaded as {primary_type}/{sub_type}")
                return model, primary_type, sub_type, errors
            except Exception as e:
                errors.append(f"TabularPredictor load failed: {e}")

        # Try MultiModalPredictor
        if MULTIMODAL_AVAILABLE:
            try:
                print(f"Attempting to load as MultiModalPredictor from {path}")
                model = MultiModalPredictor.load(str(path))
                primary_type, sub_type = detect_model_type(model)
                print(f"Successfully loaded as {primary_type}/{sub_type}")
                return model, primary_type, sub_type, errors
            except Exception as e:
                errors.append(f"MultiModalPredictor load failed: {e}")

        # Try TimeSeriesPredictor
        if TIMESERIES_AVAILABLE:
            try:
                print(f"Attempting to load as TimeSeriesPredictor from {path}")
                model = TimeSeriesPredictor.load(str(path))
                primary_type, sub_type = detect_model_type(model)
                print(f"Successfully loaded as {primary_type}/{sub_type}")
                return model, primary_type, sub_type, errors
            except Exception as e:
                errors.append(f"TimeSeriesPredictor load failed: {e}")

    # Strategy 2: Load as pickle/joblib file
    if path.is_file():
        # Try joblib
        try:
            import joblib
            print(f"Attempting to load with joblib from {path}")
            model = joblib.load(path)
            primary_type, sub_type = detect_model_type(model)
            print(f"Successfully loaded as {primary_type}/{sub_type}")
            return model, primary_type, sub_type, errors
        except Exception as e:
            errors.append(f"Joblib load failed: {e}")

        # Try pickle
        try:
            import pickle
            print(f"Attempting to load with pickle from {path}")
            with path.open("rb") as f:
                model = pickle.load(f)
            primary_type, sub_type = detect_model_type(model)
            print(f"Successfully loaded as {primary_type}/{sub_type}")
            return model, primary_type, sub_type, errors
        except Exception as e:
            errors.append(f"Pickle load failed: {e}")

    # All strategies failed
    error_details = "\n".join(errors) if errors else "Unknown error"
    raise ValueError(f"Could not load model from {path}. Errors:\n{error_details}")


# =============================================================================
# SECTION 4: DATA HANDLING
# =============================================================================

def load_and_prepare_data(
    csv_file,
    primary_type: str,
    sub_type: str,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any], List[str]]:
    """
    Load CSV and prepare data based on model type.

    Returns:
        Tuple[X, y, data_info, errors]
    """
    df = pd.read_csv(csv_file)
    errors = []
    data_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    print(f"Loaded data: shape={df.shape}, columns={len(df.columns)}")

    # Handle AutoGluon predictors
    if primary_type == 'autogluon':
        if sub_type == 'tabular':
            # TabularPredictor: requires target column
            if not target_col:
                target_col = detect_target_column(df)

            if target_col and target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
                print(f"TabularPredictor: X shape={X.shape}, y shape={y.shape}")
                return X, y, data_info, errors
            else:
                errors.append(f"Target column '{target_col}' not found in data")
                return df, None, data_info, errors

        elif sub_type == 'multimodal':
            # MultiModalPredictor: target optional
            if target_col and target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                y = None
                X = df
            print(f"MultiModalPredictor: X shape={X.shape}, y={y.shape if y is not None else None}")
            return X, y, data_info, errors

        elif sub_type == 'timeseries':
            # TimeSeriesPredictor: expects TimeSeriesDataFrame format
            # Try to convert if it's not already
            if TIMESERIES_AVAILABLE:
                try:
                    if not isinstance(df, TimeSeriesDataFrame):
                        ts_df = TimeSeriesDataFrame.from_data_frame(df)
                    else:
                        ts_df = df
                    print(f"TimeSeriesPredictor: TimeSeriesDataFrame shape={ts_df.shape}")
                    return ts_df, None, data_info, errors
                except Exception as e:
                    errors.append(f"TimeSeriesDataFrame conversion failed: {e}")
                    return df, None, data_info, errors
            else:
                return df, None, data_info, errors

    # Handle generic models
    else:  # primary_type == 'generic'
        if not target_col:
            target_col = detect_target_column(df)

        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            print(f"Generic model: X shape={X.shape}, y shape={y.shape}")
            return X, y, data_info, errors
        else:
            # No target column - return full dataframe
            print(f"Generic model: No target column, X shape={df.shape}")
            return df, None, data_info, errors


def ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric for visualization."""
    X_numeric = X.copy()

    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object':
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
            except:
                # Drop column if encoding fails
                X_numeric = X_numeric.drop(columns=[col])

    return X_numeric


# =============================================================================
# SECTION 5: SHAP EXPLAINER SELECTION
# =============================================================================

def select_shap_explainer(model, X_sample: pd.DataFrame, model_subtype: str):
    """
    Auto-select best SHAP explainer for model type.

    Returns:
        Tuple[explainer, explainer_name] or (None, None) if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        return None, None

    print(f"Selecting SHAP explainer for model type: {model_subtype}")

    # Tree-based models: Use TreeExplainer (fast, exact)
    if model_subtype in ['xgboost', 'lightgbm', 'catboost', 'sklearn_tree']:
        try:
            explainer = shap.TreeExplainer(model)
            print("Using TreeExplainer")
            return explainer, 'TreeExplainer'
        except Exception as e:
            print(f"TreeExplainer failed: {e}, trying fallback")

    # Linear models: Use LinearExplainer
    if model_subtype == 'linear':
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            print("Using LinearExplainer")
            return explainer, 'LinearExplainer'
        except Exception as e:
            print(f"LinearExplainer failed: {e}, trying fallback")

    # Fallback: KernelExplainer (slow but universal)
    try:
        background = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict, background)
        print("Using KernelExplainer (slow but universal)")
        return explainer, 'KernelExplainer'
    except Exception as e:
        print(f"KernelExplainer failed: {e}")
        return None, None


# =============================================================================
# SECTION 6: METRICS COMPUTATION
# =============================================================================

def compute_classification_metrics(y_true, y_pred, y_proba, classes) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    metrics = {}

    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        metrics['accuracy'] = None

    try:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report_dict
        metrics['f1_macro'] = report_dict.get('macro avg', {}).get('f1-score', None)
        metrics['f1_weighted'] = report_dict.get('weighted avg', {}).get('f1-score', None)
        metrics['precision_macro'] = report_dict.get('macro avg', {}).get('precision', None)
        metrics['recall_macro'] = report_dict.get('macro avg', {}).get('recall', None)
    except:
        metrics['classification_report'] = None

    try:
        if len(classes) == 2 and y_proba is not None:
            # Binary classification AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        elif y_proba is not None:
            # Multiclass AUC (one-vs-rest)
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
    except:
        metrics['roc_auc'] = None

    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Compute comprehensive regression metrics."""
    metrics = {}

    try:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
    except Exception as e:
        print(f"Regression metrics computation failed: {e}")

    return metrics


def compute_forecast_statistics(forecasts) -> Dict[str, Any]:
    """Compute statistics for time series forecasts."""
    stats = {}

    try:
        if hasattr(forecasts, 'mean'):
            mean_col = forecasts['mean'] if 'mean' in forecasts.columns else forecasts.iloc[:, -1]
            stats['mean_forecast'] = float(mean_col.mean())
            stats['std_forecast'] = float(mean_col.std())
            stats['min_forecast'] = float(mean_col.min())
            stats['max_forecast'] = float(mean_col.max())
    except Exception as e:
        print(f"Forecast statistics computation failed: {e}")

    return stats


# TO BE CONTINUED IN NEXT SECTION...
# (This file will continue with visualization functions, explainability pipelines, etc.)

# =============================================================================
# SECTION 7: FEATURE IMPORTANCE EXTRACTION
# =============================================================================

def get_native_feature_importance(model, feature_names: List[str], model_subtype: str) -> Optional[pd.DataFrame]:
    """Extract native feature importance if available."""
    
    # Tree-based models with feature_importances_
    if hasattr(model, 'feature_importances_'):
        try:
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print(f"Extracted native feature importance ({model_subtype})")
                return df
        except Exception as e:
            print(f"Failed to extract feature_importances_: {e}")
    
    # Linear models with coef_
    if hasattr(model, 'coef_'):
        try:
            coef = model.coef_
            if len(coef.shape) == 1:  # Single output
                importances = np.abs(coef)
            else:  # Multi-output
                importances = np.abs(coef).mean(axis=0)
            
            if len(importances) == len(feature_names):
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print(f"Extracted coefficient-based importance ({model_subtype})")
                return df
        except Exception as e:
            print(f"Failed to extract coef_: {e}")
    
    return None


def get_permutation_importance(model, X, y, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Compute permutation importance (model-agnostic)."""
    if not SKLEARN_INSPECTION_AVAILABLE or y is None:
        return None
    
    try:
        print("Computing permutation importance (model-agnostic)...")
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        print("Permutation importance computed successfully")
        return df
    except Exception as e:
        print(f"Permutation importance failed: {e}")
        return None


# =============================================================================
# SECTION 8: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, classes) -> Optional[str]:
    """Generate confusion matrix visualization."""
    try:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")
        return None


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
        print(f"ROC curve plot failed: {e}")
        return None


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance") -> Optional[str]:
    """Plot feature importance bar chart."""
    try:
        top_n = min(20, len(importance_df))
        df_plot = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(range(len(df_plot)), df_plot['importance'].values, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'].values)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Feature importance plot failed: {e}")
        return None


def plot_pca_analysis(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[Optional[str], Optional[str]]:
    """Generate PCA variance and scatter plots."""
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
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue", alpha=0.8)
        ax1.set_xlabel("Principal Component", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.set_title("PCA Explained Variance", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(1, len(var_ratio) + 1))
        
        for bar, val in zip(bars, var_ratio):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        variance_plot = fig_to_base64(fig1)
        
        # 2D scatter plot
        scatter_plot = None
        if components.shape[1] >= 2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            
            if y is not None:
                y_aligned = y.loc[X_clean.index]
                class_labels, class_ids = pd.factorize(y_aligned)
                scatter = ax2.scatter(components[:, 0], components[:, 1],
                                    c=class_labels, cmap="viridis", alpha=0.6,
                                    edgecolors='k', linewidth=0.5, s=50)
                
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=scatter.cmap(scatter.norm(i)),
                                    markersize=8, label=str(label))
                          for i, label in enumerate(class_ids)]
                ax2.legend(handles=handles, title="Class", loc="best")
            else:
                ax2.scatter(components[:, 0], components[:, 1],
                          alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
            
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


def plot_shap_summary(explainer, X_sample: pd.DataFrame, max_samples: int = 300) -> Optional[str]:
    """Generate SHAP summary plot."""
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


def plot_embeddings_pca(embeddings: np.ndarray, y: Optional[pd.Series] = None) -> Optional[str]:
    """Generate PCA plot of embeddings."""
    try:
        if embeddings.shape[1] < 2:
            return None
        
        pca = PCA(n_components=min(2, embeddings.shape[1]))
        components = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if y is not None:
            class_labels, class_ids = pd.factorize(y)
            scatter = ax.scatter(components[:, 0], components[:, 1],
                                c=class_labels, cmap="viridis", alpha=0.6,
                                edgecolors='k', linewidth=0.5, s=50)
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                markersize=8, label=str(label))
                      for i, label in enumerate(class_ids)]
            ax.legend(handles=handles, title="Class", loc="best")
        else:
            ax.scatter(components[:, 0], components[:, 1],
                      alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        if components.shape[1] > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title("PCA of Embeddings", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Embeddings PCA plot failed: {e}")
        return None


def plot_embeddings_tsne(embeddings: np.ndarray, y: Optional[pd.Series] = None) -> Optional[str]:
    """Generate t-SNE plot of embeddings."""
    try:
        if embeddings.shape[0] < 2:
            return None
        
        tsne = TSNE(n_components=2, random_state=42)
        components = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if y is not None:
            class_labels, class_ids = pd.factorize(y)
            scatter = ax.scatter(components[:, 0], components[:, 1],
                                c=class_labels, cmap="viridis", alpha=0.6,
                                edgecolors='k', linewidth=0.5, s=50)
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                markersize=8, label=str(label))
                      for i, label in enumerate(class_ids)]
            ax.legend(handles=handles, title="Class", loc="best")
        else:
            ax.scatter(components[:, 0], components[:, 1],
                      alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
        
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("t-SNE of Embeddings", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Embeddings t-SNE plot failed: {e}")
        return None


# =============================================================================
# SECTION 9: EXPLAINABILITY PIPELINES
# =============================================================================

def explain_autogluon_tabular(
    predictor,
    X: pd.DataFrame,
    y: pd.Series,
    max_shap_samples: int,
    warnings: List[str]
) -> Dict[str, Any]:
    """
    Complete explainability pipeline for AutoGluon TabularPredictor.

    Returns dict with metrics, plots (base64), and metadata.
    """
    results = {
        'metrics': {},
        'plots': {},
        'is_classification': None,
        'classes': None,
        'feature_importance': None,
        'feature_importance_method': None
    }

    print("Running TabularPredictor explainability pipeline...")

    # Generate predictions
    print("  - Generating predictions...")
    y_pred = safe_compute(
        lambda: predictor.predict(X),
        warnings,
        error_prefix="Prediction generation"
    )
    if y_pred is None:
        warnings.append("Failed to generate predictions")
        return results

    # Determine problem type
    has_predict_proba = hasattr(predictor, 'predict_proba')
    results['is_classification'] = has_predict_proba

    if has_predict_proba:
        print("  - Detected classification problem")
        # Classification pipeline
        y_proba_df = safe_compute(
            lambda: predictor.predict_proba(X),
            warnings,
            error_prefix="Probability prediction"
        )

        if y_proba_df is not None:
            classes = list(y_proba_df.columns)
            y_proba = y_proba_df.values
            results['classes'] = classes

            # Compute classification metrics
            print("  - Computing classification metrics...")
            metrics = compute_classification_metrics(y, y_pred, y_proba, classes)
            results['metrics'] = metrics

            # Confusion matrix
            print("  - Generating confusion matrix...")
            cm_plot = safe_compute(
                lambda: plot_confusion_matrix(y, y_pred, classes),
                warnings,
                error_prefix="Confusion matrix generation"
            )
            if cm_plot:
                results['plots']['confusion_matrix'] = cm_plot

            # ROC curve
            print("  - Generating ROC curve...")
            roc_plot = safe_compute(
                lambda: plot_roc_curve(y, y_proba, classes),
                warnings,
                error_prefix="ROC curve generation"
            )
            if roc_plot:
                results['plots']['roc_curve'] = roc_plot
    else:
        print("  - Detected regression problem")
        # Regression pipeline
        metrics = compute_regression_metrics(y, y_pred)
        results['metrics'] = metrics

        # Residual plot
        print("  - Generating residual plot...")
        def plot_residuals():
            residuals = y - y_pred
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Predicted vs Actual
            ax1.scatter(y_pred, y, alpha=0.5, edgecolors='k', linewidth=0.5)
            ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax1.set_xlabel('Predicted', fontsize=12)
            ax1.set_ylabel('Actual', fontsize=12)
            ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
            ax1.grid(alpha=0.3)

            # Residual distribution
            ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='k')
            ax2.axvline(0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Residuals', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            return fig_to_base64(fig)

        residual_plot = safe_compute(plot_residuals, warnings, error_prefix="Residual plot generation")
        if residual_plot:
            results['plots']['residual_plot'] = residual_plot

    # Feature Importance
    print("  - Extracting feature importance...")
    feature_names = list(X.columns)

    # Try native AutoGluon feature importance
    importance_df = safe_compute(
        lambda: predictor.feature_importance(X),
        warnings,
        error_prefix="Native feature importance (AutoGluon)"
    )

    if importance_df is not None and len(importance_df) > 0:
        importance_df = importance_df.sort_values('importance', ascending=False)
        results['feature_importance'] = importance_df
        results['feature_importance_method'] = 'AutoGluon Native'
        print(f"    ✓ Extracted native feature importance ({len(importance_df)} features)")
    else:
        # Try permutation importance
        print("    - Native failed, trying permutation importance...")
        importance_df = get_permutation_importance(predictor, X, y, feature_names)
        if importance_df is not None:
            results['feature_importance'] = importance_df
            results['feature_importance_method'] = 'Permutation'
            print(f"    ✓ Computed permutation importance ({len(importance_df)} features)")

    # Plot feature importance
    if results['feature_importance'] is not None:
        print("  - Generating feature importance plot...")
        fi_plot = safe_compute(
            lambda: plot_feature_importance(
                results['feature_importance'],
                f"Feature Importance ({results['feature_importance_method']})"
            ),
            warnings,
            error_prefix="Feature importance plot"
        )
        if fi_plot:
            results['plots']['feature_importance'] = fi_plot

    # SHAP Analysis
    if SHAP_AVAILABLE and results['is_classification']:
        print("  - Running SHAP analysis...")
        X_sample = X.sample(n=min(max_shap_samples, len(X)), random_state=42)

        def compute_shap():
            # Create prediction wrapper
            def predict_fn(data):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data, columns=X_sample.columns)
                return predictor.predict_proba(data).values

            # Try TreeExplainer first
            explainer = None
            shap_values = None
            try:
                explainer = shap.TreeExplainer(predictor)
                shap_values = explainer.shap_values(X_sample)
                print("    ✓ Using TreeExplainer")
            except:
                # Fallback to KernelExplainer
                print("    - TreeExplainer failed, using KernelExplainer...")
                background = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
                explainer = shap.KernelExplainer(predict_fn, background)
                X_sample_small = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
                shap_values = explainer.shap_values(X_sample_small)
                print("    ✓ Using KernelExplainer")

            # Handle multiclass
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]

            # Create summary plot
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample if not isinstance(shap_values, list) else X_sample_small,
                            plot_type="bar", max_display=20, show=False)
            plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold", pad=20)
            plt.tight_layout()
            return fig_to_base64(fig)

        shap_plot = safe_compute(compute_shap, warnings, error_prefix="SHAP analysis")
        if shap_plot:
            results['plots']['shap_summary'] = shap_plot

    # PCA Analysis
    print("  - Running PCA analysis...")
    pca_variance, pca_scatter = safe_compute(
        lambda: plot_pca_analysis(X, y),
        warnings,
        (None, None),
        error_prefix="PCA analysis"
    )
    if pca_variance:
        results['plots']['pca_variance'] = pca_variance
    if pca_scatter:
        results['plots']['pca_scatter'] = pca_scatter

    print("  ✓ TabularPredictor pipeline complete")
    return results


def explain_generic_model(
    model,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    model_subtype: str,
    max_shap_samples: int,
    warnings: List[str]
) -> Dict[str, Any]:
    """
    Complete explainability pipeline for generic pickled models.

    Returns dict with metrics, plots (base64), and metadata.
    """
    results = {
        'metrics': {},
        'plots': {},
        'is_classification': None,
        'classes': None,
        'feature_importance': None,
        'feature_importance_method': None
    }

    print("Running generic model explainability pipeline...")

    # Handle feature alignment
    X_model = X.copy()
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            warnings.append(f"Missing features in data: {missing_features}")
            return results
        X_model = X[model_features]

    # Encode categorical columns
    X_numeric = ensure_numeric(X_model)

    # Determine problem type
    has_predict_proba = hasattr(model, 'predict_proba')
    results['is_classification'] = has_predict_proba

    # Generate predictions
    print("  - Generating predictions...")
    y_pred = safe_compute(
        lambda: model.predict(X_numeric),
        warnings,
        error_prefix="Prediction generation"
    )
    if y_pred is None:
        warnings.append("Failed to generate predictions")
        return results

    if y is None:
        warnings.append("No target column provided - skipping metrics")
        return results

    if has_predict_proba:
        print("  - Detected classification problem")
        # Classification pipeline
        y_proba = safe_compute(
            lambda: model.predict_proba(X_numeric),
            warnings,
            error_prefix="Probability prediction"
        )

        if y_proba is not None:
            classes = np.unique(y)
            results['classes'] = list(classes)

            # Compute classification metrics
            print("  - Computing classification metrics...")
            metrics = compute_classification_metrics(y, y_pred, y_proba, classes)
            results['metrics'] = metrics

            # Confusion matrix
            print("  - Generating confusion matrix...")
            cm_plot = safe_compute(
                lambda: plot_confusion_matrix(y, y_pred, classes),
                warnings,
                error_prefix="Confusion matrix generation"
            )
            if cm_plot:
                results['plots']['confusion_matrix'] = cm_plot

            # ROC curve
            print("  - Generating ROC curve...")
            roc_plot = safe_compute(
                lambda: plot_roc_curve(y, y_proba, classes),
                warnings,
                error_prefix="ROC curve generation"
            )
            if roc_plot:
                results['plots']['roc_curve'] = roc_plot
    else:
        print("  - Detected regression problem")
        # Regression pipeline
        metrics = compute_regression_metrics(y, y_pred)
        results['metrics'] = metrics

        # Residual plot
        print("  - Generating residual plot...")
        def plot_residuals():
            residuals = y - y_pred
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Predicted vs Actual
            ax1.scatter(y_pred, y, alpha=0.5, edgecolors='k', linewidth=0.5)
            ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax1.set_xlabel('Predicted', fontsize=12)
            ax1.set_ylabel('Actual', fontsize=12)
            ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
            ax1.grid(alpha=0.3)

            # Residual distribution
            ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='k')
            ax2.axvline(0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Residuals', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            return fig_to_base64(fig)

        residual_plot = safe_compute(plot_residuals, warnings, error_prefix="Residual plot generation")
        if residual_plot:
            results['plots']['residual_plot'] = residual_plot

    # Feature Importance
    print("  - Extracting feature importance...")
    feature_names = list(X_numeric.columns)

    # Try native importance
    importance_df = get_native_feature_importance(model, feature_names, model_subtype)
    if importance_df is not None:
        results['feature_importance'] = importance_df
        results['feature_importance_method'] = 'Native'
        print(f"    ✓ Extracted native feature importance ({len(importance_df)} features)")
    else:
        # Try permutation importance
        print("    - Native failed, trying permutation importance...")
        importance_df = get_permutation_importance(model, X_numeric, y, feature_names)
        if importance_df is not None:
            results['feature_importance'] = importance_df
            results['feature_importance_method'] = 'Permutation'
            print(f"    ✓ Computed permutation importance ({len(importance_df)} features)")

    # Plot feature importance
    if results['feature_importance'] is not None:
        print("  - Generating feature importance plot...")
        fi_plot = safe_compute(
            lambda: plot_feature_importance(
                results['feature_importance'],
                f"Feature Importance ({results['feature_importance_method']})"
            ),
            warnings,
            error_prefix="Feature importance plot"
        )
        if fi_plot:
            results['plots']['feature_importance'] = fi_plot

    # SHAP Analysis
    if SHAP_AVAILABLE:
        print("  - Running SHAP analysis...")
        X_sample = X_numeric.sample(n=min(max_shap_samples, len(X_numeric)), random_state=42)

        explainer, explainer_name = select_shap_explainer(model, X_sample, model_subtype)
        if explainer:
            shap_plot = safe_compute(
                lambda: plot_shap_summary(explainer, X_sample, max_shap_samples),
                warnings,
                error_prefix="SHAP analysis"
            )
            if shap_plot:
                results['plots']['shap_summary'] = shap_plot
                print(f"    ✓ SHAP analysis complete ({explainer_name})")

    # PCA Analysis
    print("  - Running PCA analysis...")
    pca_variance, pca_scatter = safe_compute(
        lambda: plot_pca_analysis(X_numeric, y),
        warnings,
        (None, None),
        error_prefix="PCA analysis"
    )
    if pca_variance:
        results['plots']['pca_variance'] = pca_variance
    if pca_scatter:
        results['plots']['pca_scatter'] = pca_scatter

    print("  ✓ Generic model pipeline complete")
    return results


def explain_autogluon_multimodal(
    predictor,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    max_embedding_samples: int,
    warnings: List[str]
) -> Dict[str, Any]:
    """
    Explainability pipeline for AutoGluon MultiModalPredictor.

    Focus on embeddings extraction and visualization.
    """
    results = {
        'metrics': {},
        'plots': {},
        'embeddings_shape': None
    }

    print("Running MultiModalPredictor explainability pipeline...")
    warnings.append("MultiModal models have limited explainability support")

    # Try predictions
    if y is not None:
        print("  - Attempting predictions...")
        y_pred = safe_compute(
            lambda: predictor.predict(X),
            warnings,
            error_prefix="Prediction generation"
        )

        if y_pred is not None and hasattr(predictor, 'predict_proba'):
            y_proba = safe_compute(
                lambda: predictor.predict_proba(X),
                warnings,
                error_prefix="Probability prediction"
            )
            if y_proba is not None:
                classes = np.unique(y)
                results['classes'] = list(classes)
                metrics = compute_classification_metrics(y, y_pred, y_proba, classes)
                results['metrics'] = metrics

                # Confusion matrix
                cm_plot = safe_compute(
                    lambda: plot_confusion_matrix(y, y_pred, classes),
                    warnings,
                    error_prefix="Confusion matrix generation"
                )
                if cm_plot:
                    results['plots']['confusion_matrix'] = cm_plot

    # Extract embeddings
    print("  - Extracting embeddings...")
    embeddings = safe_compute(
        lambda: predictor.extract_embedding(X.sample(n=min(max_embedding_samples, len(X)), random_state=42)),
        warnings,
        error_prefix="Embeddings extraction"
    )

    if embeddings is not None:
        if isinstance(embeddings, pd.DataFrame):
            embeddings = embeddings.values

        results['embeddings_shape'] = embeddings.shape
        print(f"    ✓ Extracted embeddings: shape={embeddings.shape}")

        # PCA of embeddings
        print("  - Running PCA on embeddings...")
        pca_plot = safe_compute(
            lambda: plot_embeddings_pca(embeddings, y.sample(n=min(max_embedding_samples, len(y)), random_state=42) if y is not None else None),
            warnings,
            error_prefix="Embeddings PCA"
        )
        if pca_plot:
            results['plots']['embeddings_pca'] = pca_plot

        # t-SNE of embeddings
        print("  - Running t-SNE on embeddings...")
        tsne_plot = safe_compute(
            lambda: plot_embeddings_tsne(embeddings, y.sample(n=min(max_embedding_samples, len(y)), random_state=42) if y is not None else None),
            warnings,
            error_prefix="Embeddings t-SNE"
        )
        if tsne_plot:
            results['plots']['embeddings_tsne'] = tsne_plot

    print("  ✓ MultiModal pipeline complete")
    return results


def explain_autogluon_timeseries(
    predictor,
    data,
    warnings: List[str]
) -> Dict[str, Any]:
    """
    Explainability pipeline for AutoGluon TimeSeriesPredictor.

    Focus on forecast visualization with uncertainty.
    """
    results = {
        'metrics': {},
        'plots': {},
        'forecast_shape': None
    }

    print("Running TimeSeriesPredictor explainability pipeline...")
    warnings.append("TimeSeries feature importance is unreliable for ensemble models (per AutoGluon docs)")

    # Generate forecasts
    print("  - Generating forecasts...")
    forecasts = safe_compute(
        lambda: predictor.predict(data),
        warnings,
        error_prefix="Forecast generation"
    )

    if forecasts is not None:
        results['forecast_shape'] = forecasts.shape
        print(f"    ✓ Generated forecasts: shape={forecasts.shape}")

        # Compute statistics
        stats = compute_forecast_statistics(forecasts)
        results['metrics'] = stats

        # Plot forecast with uncertainty
        print("  - Generating forecast plot...")
        def plot_forecast():
            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract quantile columns
            quantile_cols = [col for col in forecasts.columns if 'mean' in str(col) or any(q in str(col) for q in ['0.1', '0.5', '0.9'])]

            if 'mean' in forecasts.columns:
                mean_forecast = forecasts['mean']
                ax.plot(mean_forecast.index, mean_forecast.values, 'b-', linewidth=2, label='Mean Forecast')
            elif len(quantile_cols) > 0:
                # Use first quantile as proxy
                proxy = forecasts[quantile_cols[0]]
                ax.plot(proxy.index, proxy.values, 'b-', linewidth=2, label='Forecast')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Time Series Forecast', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            return fig_to_base64(fig)

        forecast_plot = safe_compute(plot_forecast, warnings, error_prefix="Forecast plot generation")
        if forecast_plot:
            results['plots']['forecast'] = forecast_plot

    print("  ✓ TimeSeries pipeline complete")
    return results


# =============================================================================
# SECTION 10: HTML REPORT BUILDER
# =============================================================================

def build_comprehensive_html_report(
    primary_type: str,
    sub_type: str,
    results: Dict[str, Any],
    data_info: Dict[str, Any],
    user_level: str,
    warnings: List[str]
) -> str:
    """Build comprehensive HTML report with all visualizations and metrics."""

    # Helper function for image sections
    def image_section(img_b64: str, title: str, description: str = "") -> str:
        desc_html = f'<p class="plot-description">{description}</p>' if description else ''
        return f'''
        <div class="section">
            <h3>{title}</h3>
            {desc_html}
            <img src="data:image/png;base64,{img_b64}" alt="{title}"/>
        </div>
        '''

    # Warnings section
    warnings_html = ''
    if warnings:
        warnings_items = ''.join(f'<li>{w}</li>' for w in warnings)
        warnings_html = f'''
        <div class="warning-box">
            <h3>⚠️ Warnings & Limitations</h3>
            <ul>
                {warnings_items}
            </ul>
        </div>
        '''

    # Metrics section
    metrics_html = ''
    if results.get('metrics'):
        metrics = results['metrics']

        if results.get('is_classification'):
            # Classification metrics
            accuracy = metrics.get('accuracy', 'N/A')
            f1_macro = metrics.get('f1_macro', 'N/A')
            f1_weighted = metrics.get('f1_weighted', 'N/A')
            precision = metrics.get('precision_macro', 'N/A')
            recall = metrics.get('recall_macro', 'N/A')
            roc_auc = metrics.get('roc_auc', 'N/A')

            # Format values
            if accuracy != 'N/A':
                accuracy = f"{accuracy:.3f}"
            if f1_macro != 'N/A':
                f1_macro = f"{f1_macro:.3f}"
            if f1_weighted != 'N/A':
                f1_weighted = f"{f1_weighted:.3f}"
            if precision != 'N/A':
                precision = f"{precision:.3f}"
            if recall != 'N/A':
                recall = f"{recall:.3f}"
            if roc_auc != 'N/A':
                roc_auc = f"{roc_auc:.3f}"

            metrics_html = f'''
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Accuracy</div>
                        <div class="value">{accuracy}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Macro F1</div>
                        <div class="value">{f1_macro}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Weighted F1</div>
                        <div class="value">{f1_weighted}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Precision</div>
                        <div class="value">{precision}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Recall</div>
                        <div class="value">{recall}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">ROC AUC</div>
                        <div class="value">{roc_auc}</div>
                    </div>
                </div>
            </div>
            '''

            # Classification report
            if metrics.get('classification_report'):
                report_dict = metrics['classification_report']
                report_df = pd.DataFrame(report_dict).T
                report_html = report_df.to_html(classes="table", float_format="%.3f", border=0)
                metrics_html += f'''
                <div class="section">
                    <h3>Classification Report</h3>
                    {report_html}
                </div>
                '''
        else:
            # Regression metrics
            mae = metrics.get('mae', 'N/A')
            rmse = metrics.get('rmse', 'N/A')
            r2 = metrics.get('r2', 'N/A')
            mse = metrics.get('mse', 'N/A')

            if mae != 'N/A':
                mae = f"{mae:.3f}"
            if rmse != 'N/A':
                rmse = f"{rmse:.3f}"
            if r2 != 'N/A':
                r2 = f"{r2:.3f}"
            if mse != 'N/A':
                mse = f"{mse:.3f}"

            metrics_html = f'''
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">MAE</div>
                        <div class="value">{mae}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">RMSE</div>
                        <div class="value">{rmse}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">R²</div>
                        <div class="value">{r2}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">MSE</div>
                        <div class="value">{mse}</div>
                    </div>
                </div>
            </div>
            '''

    # Visualizations
    plots_html = ''
    plots = results.get('plots', {})

    if plots.get('confusion_matrix'):
        plots_html += image_section(plots['confusion_matrix'], "Confusion Matrix",
                                     "Shows prediction vs actual counts. Diagonal = correct predictions.")

    if plots.get('roc_curve'):
        plots_html += image_section(plots['roc_curve'], "ROC Curve",
                                     "Trade-off between true positive and false positive rate. AUC closer to 1.0 is better.")

    if plots.get('residual_plot'):
        plots_html += image_section(plots['residual_plot'], "Residual Analysis",
                                     "Left: predicted vs actual values. Right: distribution of prediction errors.")

    if plots.get('feature_importance'):
        method = results.get('feature_importance_method', 'Unknown')
        plots_html += image_section(plots['feature_importance'], f"Feature Importance ({method})",
                                     "Shows which features have the most impact on predictions.")

    if plots.get('shap_summary'):
        plots_html += image_section(plots['shap_summary'], "SHAP Feature Importance",
                                     "SHAP values show feature contributions based on game theory (Shapley values).")

    if plots.get('pca_variance'):
        plots_html += image_section(plots['pca_variance'], "PCA Explained Variance",
                                     "Shows how much information each principal component captures.")

    if plots.get('pca_scatter'):
        plots_html += image_section(plots['pca_scatter'], "PCA Projection",
                                     "2D visualization of data in principal component space, colored by target.")

    if plots.get('embeddings_pca'):
        plots_html += image_section(plots['embeddings_pca'], "Embeddings PCA",
                                     "2D projection of learned embeddings from multimodal model.")

    if plots.get('embeddings_tsne'):
        plots_html += image_section(plots['embeddings_tsne'], "Embeddings t-SNE",
                                     "t-SNE nonlinear dimensionality reduction of embeddings.")

    if plots.get('forecast'):
        plots_html += image_section(plots['forecast'], "Time Series Forecast",
                                     "Predicted future values over time.")

    # Build complete HTML
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.95; }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .warning-box h3 {{ color: #f57c00; margin-bottom: 10px; }}
        .warning-box ul {{ margin-left: 20px; }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
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
        .plot-description {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 10px;
            font-style: italic;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Universal Model Explainability Report</h1>
            <p><strong>Model Type:</strong> {primary_type} / {sub_type}</p>
            <p><strong>User Level:</strong> {user_level.title()}</p>
            <p><strong>Data Shape:</strong> {data_info['shape']}</p>
        </div>

        {warnings_html}

        <div class="info-box">
            <h3>Model Information</h3>
            <p><strong>Primary Type:</strong> {primary_type}</p>
            <p><strong>Sub Type:</strong> {sub_type}</p>
            <p><strong>Features:</strong> {len(data_info['columns'])}</p>
        </div>

        {metrics_html}
        {plots_html}

        <div class="section">
            <h2>Understanding the Visualizations</h2>
            <div class="info-box">
                <p><strong>Confusion Matrix:</strong> Shows prediction vs actual counts for each class.</p>
                <p><strong>ROC Curve:</strong> Shows model's ability to distinguish between classes.</p>
                <p><strong>Feature Importance:</strong> Indicates which features most influence predictions.</p>
                <p><strong>SHAP Values:</strong> Game-theory based feature importance (Shapley values).</p>
                <p><strong>PCA:</strong> Dimensionality reduction to visualize high-dimensional data.</p>
            </div>
        </div>
    </div>
</body>
</html>
'''

    return html


# =============================================================================
# SECTION 11: FLASK ENDPOINTS
# =============================================================================

def create_error_html(error_type: str, details: str, suggestions: List[str]) -> str:
    """Generate user-friendly error page."""
    suggestions_html = ''.join(f'<li>{s}</li>' for s in suggestions)
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Explainability Error</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 40px;
                background: #f5f5f5;
            }}
            .error-container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            h1 {{ color: #d32f2f; }}
            h2 {{ color: #333; margin-top: 30px; }}
            .error-details {{
                background: #ffebee;
                border-left: 4px solid #d32f2f;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .error-suggestions {{
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            ul {{ margin: 10px 0; padding-left: 20px; }}
            li {{ margin: 8px 0; }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <h1>Error: {error_type}</h1>
            <div class="error-details">
                <h2>Details</h2>
                <p>{details}</p>
            </div>
            <div class="error-suggestions">
                <h2>Suggestions</h2>
                <ul>
                    {suggestions_html}
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "autogluon-universal-explainability",
        "shap_available": SHAP_AVAILABLE,
        "autogluon_tabular": TABULAR_AVAILABLE,
        "autogluon_multimodal": MULTIMODAL_AVAILABLE,
        "autogluon_timeseries": TIMESERIES_AVAILABLE
    }, 200


@app.route('/explain-model', methods=['POST'])
def explain_model():
    """Main explainability endpoint."""
    try:
        print("="*60)
        print("Explainability request received")
        print("="*60)
        
        # Get parameters
        user_level = request.form.get('user_level', 'expert').lower()
        target_col = request.form.get('target_col')
        max_shap_samples = int(request.form.get('max_shap_samples', DEFAULT_MAX_SHAP_SAMPLES))
        
        # Load model
        model_path = request.form.get('model_path')
        if not model_path and 'model_file' in request.files:
            # Save uploaded file temporarily
            import tempfile
            model_file = request.files['model_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                model_file.save(tmp.name)
                model_path = tmp.name
        
        if not model_path:
            return create_error_html(
                "Missing Model",
                "No model file or path provided",
                [
                    "Provide 'model_path' form field with path to predictor",
                    "Or upload 'model_file' pickle file"
                ]
            ), 400
        
        # Load model
        try:
            model, primary_type, sub_type, load_errors = load_model(model_path)
            print(f"Model loaded: {primary_type}/{sub_type}")
        except Exception as e:
            return create_error_html(
                "Model Loading Failed",
                str(e),
                [
                    "Ensure the model file/directory exists and is accessible",
                    "Check that the model was saved correctly",
                    "For AutoGluon models, provide the predictor directory path",
                    "For generic models, provide the pickle (.pkl) file path"
                ]
            ), 400
        
        # Load data
        if 'data_file' not in request.files:
            return create_error_html(
                "Missing Data",
                "No data file provided",
                ["Upload 'data_file' CSV file with test data"]
            ), 400
        
        data_file = request.files['data_file']
        try:
            X, y, data_info, data_errors = load_and_prepare_data(
                data_file, primary_type, sub_type, target_col
            )
            print(f"Data loaded: X shape={X.shape}, y shape={y.shape if y is not None else None}")
        except Exception as e:
            return create_error_html(
                "Data Loading Failed",
                str(e),
                [
                    "Ensure CSV file is valid and properly formatted",
                    "Check that target column exists if specified",
                    "For time series, ensure 'item_id' and 'timestamp' columns exist"
                ]
            ), 400
        
        # Route to appropriate explainability pipeline based on model type
        warnings = []
        results = {}

        print("Routing to explainability pipeline...")

        if primary_type == 'autogluon':
            if sub_type == 'tabular':
                results = explain_autogluon_tabular(model, X, y, max_shap_samples, warnings)
            elif sub_type == 'multimodal':
                max_embedding_samples = int(request.form.get('max_embedding_samples', DEFAULT_MAX_EMBEDDING_SAMPLES))
                results = explain_autogluon_multimodal(model, X, y, max_embedding_samples, warnings)
            elif sub_type == 'timeseries':
                results = explain_autogluon_timeseries(model, X, warnings)
            else:
                warnings.append(f"Unknown AutoGluon sub-type: {sub_type}")
        else:  # primary_type == 'generic'
            results = explain_generic_model(model, X, y, sub_type, max_shap_samples, warnings)

        # Build comprehensive HTML report
        print("Building HTML report...")
        report_html = build_comprehensive_html_report(
            primary_type, sub_type, results, data_info, user_level, warnings
        )

        print("="*60)
        print("Report generation complete!")
        print("="*60)

        return report_html
        
    except Exception as e:
        print(f"Error in explain_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_error_html(
            "Internal Error",
            str(e),
            ["Check server logs for details", "Ensure all dependencies are installed"]
        ), 500


if __name__ == '__main__':
    print("="*60)
    print("Universal Model Explainability API")
    print("="*60)
    print(f"SHAP Available: {SHAP_AVAILABLE}")
    print(f"AutoGluon Tabular: {TABULAR_AVAILABLE}")
    print(f"AutoGluon MultiModal: {MULTIMODAL_AVAILABLE}")
    print(f"AutoGluon TimeSeries: {TIMESERIES_AVAILABLE}")
    print("="*60)
    print(f"Starting Flask app on port {DEFAULT_PORT}...")
    app.run(host='0.0.0.0', port=DEFAULT_PORT, debug=True)
