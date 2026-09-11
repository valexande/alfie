"""
Base Model Explainer - Abstract base class for all model explainers.

Provides a common interface for model explainability across different model types:
- AutoGluon (Tabular, MultiModal, TimeSeries)
- sklearn (RandomForest, GradientBoosting, Linear models, etc.)
- XGBoost, LightGBM, CatBoost
- PyTorch, TensorFlow
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class BaseModelExplainer(ABC):
    """
    Abstract base class for all model explainers.
    
    Each concrete implementation handles a specific model type with
    optimized explainability methods (e.g., TreeExplainer for tree models).
    
    Example:
        >>> explainer = TreeBasedExplainer(model, X, y)
        >>> importance = explainer.get_feature_importance()
        >>> shap_values = explainer.get_shap_values(X.head(100))
        >>> report = explainer.generate_report(mode='expert')
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        max_samples: int = 1000,
        **kwargs
    ):
        """
        Initialize explainer with model and data.
        
        Args:
            model: The trained model to explain
            X: Feature DataFrame
            y: Target Series
            max_samples: Maximum samples for expensive computations
            **kwargs: Additional configuration options
        """
        self.model = model
        self.X = X
        self.y = y
        self.max_samples = max_samples
        self.config = kwargs
        
        # Cache for computed values
        self._feature_importance: Optional[pd.DataFrame] = None
        self._shap_values: Optional[np.ndarray] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._predictions: Optional[np.ndarray] = None
        
        # Validate inputs
        self._validate()
    
    def _validate(self):
        """Validate model and data compatibility."""
        if self.X is None or len(self.X) == 0:
            raise ValueError("X cannot be empty")
        if self.y is None or len(self.y) == 0:
            raise ValueError("y cannot be empty")
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y must have same length: {len(self.X)} vs {len(self.y)}")
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return model type identifier.
        
        Returns:
            str: e.g., 'xgboost', 'lightgbm', 'autogluon_tabular', 'sklearn_tree'
        """
        pass
    
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """
        Return problem type.
        
        Returns:
            str: 'classification', 'regression', or 'forecasting'
        """
        pass
    
    @property
    def is_classification(self) -> bool:
        """Check if this is a classification problem."""
        return self.problem_type == 'classification'
    
    @property
    def is_regression(self) -> bool:
        """Check if this is a regression problem."""
        return self.problem_type == 'regression'
    
    @property
    def is_forecasting(self) -> bool:
        """Check if this is a forecasting problem."""
        return self.problem_type == 'forecasting'
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names from X."""
        return list(self.X.columns)
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.X.columns)
    
    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return len(self.X)
    
    @property
    def classes(self) -> Optional[np.ndarray]:
        """Get unique classes for classification problems."""
        if self.is_classification:
            return np.unique(self.y)
        return None
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Each implementation should use the best available method:
        - Tree models: native feature_importances_
        - Linear models: coefficient magnitudes
        - AutoGluon: predictor.feature_importance()
        - Fallback: permutation importance
        
        Returns:
            DataFrame with columns ['feature', 'importance'], sorted descending
        """
        pass
    
    @abstractmethod
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get SHAP values using the best available explainer.
        
        Each implementation should use the optimal SHAP explainer:
        - Tree models: shap.TreeExplainer (fast, exact)
        - Linear models: shap.LinearExplainer
        - Neural networks: shap.DeepExplainer or GradientExplainer
        - Fallback: shap.KernelExplainer (slow)
        
        Args:
            X_sample: Samples to explain (defaults to sampled X)
            
        Returns:
            np.ndarray of SHAP values
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Classification: accuracy, precision, recall, f1, roc_auc
        Regression: mae, rmse, r2, mse
        Forecasting: custom metrics
        
        Returns:
            Dict with metric names and values
        """
        pass
    
    # =========================================================================
    # Optional Methods - Can be overridden by subclasses
    # =========================================================================
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get model predictions.
        
        Args:
            X: Data to predict on (defaults to self.X)
            
        Returns:
            np.ndarray of predictions
        """
        if X is None:
            X = self.X
        
        if self._predictions is not None and X is self.X:
            return self._predictions
        
        predictions = self.model.predict(X)
        
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        
        if X is self.X:
            self._predictions = predictions
        
        return predictions
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for classification.
        
        Args:
            X: Data to predict on (defaults to self.X)
            
        Returns:
            np.ndarray of probabilities or None if not classification
        """
        if not self.is_classification:
            return None
        
        if X is None:
            X = self.X
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        proba = self.model.predict_proba(X)
        
        if isinstance(proba, pd.DataFrame):
            return proba.values
        
        return proba
    
    def sample_data(self, n: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sample data for expensive computations.
        
        Args:
            n: Number of samples (defaults to max_samples)
            
        Returns:
            Tuple of (X_sample, y_sample)
        """
        n = n or self.max_samples
        
        if len(self.X) <= n:
            return self.X, self.y
        
        indices = np.random.choice(len(self.X), n, replace=False)
        return self.X.iloc[indices], self.y.iloc[indices]
    
    def generate_plots(self) -> Dict[str, str]:
        """
        Generate all relevant plots as base64-encoded strings.
        
        Returns:
            Dict mapping plot names to base64 PNG strings
        """
        from xai_core.visualizations import (
            plot_feature_importance,
            plot_shap_summary,
            plot_confusion_matrix,
            plot_roc_curve,
            plot_residuals,
            plot_pca_analysis
        )
        
        plots = {}
        
        # Feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            plots['feature_importance'] = plot_feature_importance(importance_df)
        
        # SHAP values
        try:
            X_sample, _ = self.sample_data(min(300, self.max_samples))
            shap_values = self.get_shap_values(X_sample)
            if shap_values is not None:
                plots['shap_summary'] = plot_shap_summary(shap_values, X_sample)
        except Exception as e:
            print(f"SHAP plot failed: {e}")
        
        # Classification-specific plots
        if self.is_classification:
            y_pred = self.get_predictions()
            y_proba = self.get_prediction_probabilities()
            classes = self.classes
            
            if y_pred is not None:
                plots['confusion_matrix'] = plot_confusion_matrix(self.y, y_pred, classes)
            
            if y_proba is not None:
                roc_plot = plot_roc_curve(self.y, y_proba, classes)
                if roc_plot:
                    plots['roc_curve'] = roc_plot
        
        # Regression-specific plots
        elif self.is_regression:
            y_pred = self.get_predictions()
            if y_pred is not None:
                plots['residuals'] = plot_residuals(self.y, y_pred)
        
        # PCA analysis (for all types)
        try:
            pca_variance, _pca_scatter = plot_pca_analysis(self.X, self.y)
            if pca_variance:
                plots['pca_variance'] = pca_variance
        except Exception as e:
            print(f"PCA plot failed: {e}")
        
        return plots
    
    def generate_report(self, mode: str = 'expert') -> str:
        """
        Generate HTML explainability report.
        
        Args:
            mode: 'beginner' for simplified report, 'expert' for full details
            
        Returns:
            HTML string with complete report
        """
        from xai_core.report_builder import ReportBuilder
        return ReportBuilder(self).build(mode)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export explainer results to dictionary.
        
        Returns:
            Dict with all explainability results
        """
        return {
            'model_type': self.model_type,
            'problem_type': self.problem_type,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
            'feature_names': self.feature_names,
            'metrics': self.get_metrics(),
            'feature_importance': self.get_feature_importance().to_dict('records'),
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type='{self.model_type}', "
            f"problem_type='{self.problem_type}', "
            f"n_features={self.n_features}, "
            f"n_samples={self.n_samples})"
        )
