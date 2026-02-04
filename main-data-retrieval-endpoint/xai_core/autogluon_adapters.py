"""
AutoGluon Adapters - Sklearn-compatible wrappers for AutoGluon predictor types.

Provides adapters that expose standard sklearn-like methods:
- predict(X) -> np.ndarray
- predict_proba(X) -> np.ndarray (for classifiers)
- classes_ attribute (for classifiers)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class AutoGluonAdapter(ABC):
    """
    Base adapter to make AutoGluon predictors sklearn-compatible.
    
    Allows AutoGluon predictors to work with SHAP and other sklearn-based tools.
    """
    
    def __init__(self, predictor: Any):
        """
        Initialize adapter with an AutoGluon predictor.
        
        Args:
            predictor: Any AutoGluon predictor (TabularPredictor, MultiModalPredictor, etc.)
        """
        self.predictor = predictor
        self._classes: Optional[np.ndarray] = None
        self._problem_type: Optional[str] = None
        self._feature_names: Optional[List[str]] = None
    
    @property
    def problem_type(self) -> str:
        """Get problem type: 'classification', 'regression', or 'forecasting'."""
        if self._problem_type is None:
            self._problem_type = self._detect_problem_type()
        return self._problem_type
    
    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Sklearn-compatible classes attribute for classifiers."""
        return self._classes
    
    @classes_.setter
    def classes_(self, value: np.ndarray):
        self._classes = value
    
    @abstractmethod
    def _detect_problem_type(self) -> str:
        """Detect if classification, regression, or forecasting."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible predict method."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible predict_proba method (classifiers only)."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoGluonAdapter":
        """No-op for pre-trained models. Required for sklearn compatibility."""
        return self
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters. Required for sklearn compatibility."""
        return {"predictor": self.predictor}
    
    def set_params(self, **params) -> "AutoGluonAdapter":
        """Set parameters. Required for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TabularAdapter(AutoGluonAdapter):
    """
    Adapter for AutoGluon TabularPredictor.
    
    Supports:
    - Binary classification
    - Multiclass classification
    - Regression
    - Quantile regression
    """
    
    def _detect_problem_type(self) -> str:
        """Detect problem type from TabularPredictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        elif ag_type == 'quantile':
            return 'regression'  # Treat quantile as regression for explainability
        return 'regression'
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        preds = self.predictor.predict(X)
        return preds.values if hasattr(preds, 'values') else np.array(preds)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for classification."""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba only available for classifiers")
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        proba_df = self.predictor.predict_proba(X)
        
        # Store class labels
        if isinstance(proba_df, pd.DataFrame):
            self._classes = np.array(proba_df.columns)
            return proba_df.values
        
        return proba_df
    
    def feature_importance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Get native AutoGluon feature importance.
        
        Uses permutation-based importance internally.
        """
        return self.predictor.feature_importance(X, **kwargs)
    
    def get_model_names(self) -> List[str]:
        """Get names of all models in the ensemble."""
        return self.predictor.get_model_names()
    
    def get_best_model(self) -> str:
        """Get name of the best performing model."""
        return self.predictor.get_model_best()
    
    def leaderboard(self, X: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """Get model leaderboard with performance metrics."""
        if X is not None:
            return self.predictor.leaderboard(X, **kwargs)
        return self.predictor.leaderboard(**kwargs)


class MultiModalAdapter(AutoGluonAdapter):
    """
    Adapter for AutoGluon MultiModalPredictor.
    
    Handles:
    - Text data
    - Image data
    - Tabular data
    - Combinations of the above
    """
    
    def _detect_problem_type(self) -> str:
        """Detect problem type from MultiModalPredictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        return 'regression'
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        preds = self.predictor.predict(X)
        return preds.values if hasattr(preds, 'values') else np.array(preds)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for classification."""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba only available for classifiers")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        proba = self.predictor.predict_proba(X)
        
        if isinstance(proba, pd.DataFrame):
            self._classes = np.array(proba.columns)
            return proba.values
        
        return proba
    
    def extract_embedding(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract embeddings from the multimodal model.
        
        Useful for visualization and understanding learned representations.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        embeddings = self.predictor.extract_embedding(X)
        
        if isinstance(embeddings, pd.DataFrame):
            return embeddings.values
        return embeddings


class TimeSeriesAdapter(AutoGluonAdapter):
    """
    Adapter for AutoGluon TimeSeriesPredictor.
    
    Note: Time series forecasting is fundamentally different from 
    classification/regression, so SHAP-based explainability is not applicable.
    This adapter provides forecast generation and model info instead.
    """
    
    def _detect_problem_type(self) -> str:
        """Time series is always forecasting."""
        return 'forecasting'
    
    def predict(self, X: Any) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Returns DataFrame with forecast values and potentially quantiles.
        """
        return self.predictor.predict(X)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Not applicable for time series forecasting."""
        raise NotImplementedError(
            "TimeSeriesPredictor doesn't support predict_proba. "
            "Use predict() for forecasts or get_forecast_with_intervals() for confidence intervals."
        )
    
    def get_forecast_with_intervals(
        self, 
        X: Any, 
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> pd.DataFrame:
        """
        Get forecasts with confidence intervals.
        
        Args:
            X: Input data (TimeSeriesDataFrame)
            quantiles: Quantile levels for prediction intervals
            
        Returns:
            DataFrame with point forecasts and quantile predictions
        """
        return self.predictor.predict(X, quantiles=quantiles)
    
    def leaderboard(self, X: Any = None, **kwargs) -> pd.DataFrame:
        """Get model leaderboard."""
        if X is not None:
            return self.predictor.leaderboard(X, **kwargs)
        return self.predictor.leaderboard(**kwargs)
    
    def get_model_names(self) -> List[str]:
        """Get names of all models."""
        return self.predictor.get_model_names()


class TextAdapter(AutoGluonAdapter):
    """
    Adapter for AutoGluon TextPredictor.
    
    Handles text classification and regression tasks.
    """
    
    def _detect_problem_type(self) -> str:
        """Detect problem type from TextPredictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        return 'regression'
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        preds = self.predictor.predict(X)
        return preds.values if hasattr(preds, 'values') else np.array(preds)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for classification."""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba only available for classifiers")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        proba = self.predictor.predict_proba(X)
        
        if isinstance(proba, pd.DataFrame):
            self._classes = np.array(proba.columns)
            return proba.values
        
        return proba


class ImageAdapter(AutoGluonAdapter):
    """
    Adapter for AutoGluon ImagePredictor.
    
    Handles image classification tasks.
    Note: SHAP support is limited for image models.
    """
    
    def _detect_problem_type(self) -> str:
        """ImagePredictor is always classification."""
        return 'classification'
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        preds = self.predictor.predict(X)
        return preds.values if hasattr(preds, 'values') else np.array(preds)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        proba = self.predictor.predict_proba(X)
        
        if isinstance(proba, pd.DataFrame):
            self._classes = np.array(proba.columns)
            return proba.values
        
        return proba


def create_adapter(predictor: Any) -> AutoGluonAdapter:
    """
    Factory function to create appropriate adapter for any AutoGluon predictor.
    
    Automatically detects the predictor type and returns the matching adapter.
    
    Args:
        predictor: Any AutoGluon predictor instance
        
    Returns:
        AutoGluonAdapter subclass instance
        
    Raises:
        ValueError: If predictor type is not recognized
        
    Example:
        >>> from autogluon.tabular import TabularPredictor
        >>> predictor = TabularPredictor.load("my_model")
        >>> adapter = create_adapter(predictor)
        >>> adapter.predict(X_test)  # sklearn-compatible
    """
    class_name = type(predictor).__name__
    module_name = type(predictor).__module__
    
    # Map predictor types to adapters
    adapter_map = {
        'TabularPredictor': TabularAdapter,
        'MultiModalPredictor': MultiModalAdapter,
        'TimeSeriesPredictor': TimeSeriesAdapter,
        'TextPredictor': TextAdapter,
        'ImagePredictor': ImageAdapter,
    }
    
    # Try to match by class name
    for predictor_name, adapter_class in adapter_map.items():
        if predictor_name in class_name:
            return adapter_class(predictor)
    
    # Check module path as fallback
    if 'autogluon.tabular' in module_name:
        return TabularAdapter(predictor)
    elif 'autogluon.multimodal' in module_name:
        return MultiModalAdapter(predictor)
    elif 'autogluon.timeseries' in module_name:
        return TimeSeriesAdapter(predictor)
    
    # Default to TabularAdapter (most common case)
    print(f"Warning: Unknown predictor type {class_name}, defaulting to TabularAdapter")
    return TabularAdapter(predictor)


def is_autogluon_predictor(model: Any) -> bool:
    """
    Check if a model is an AutoGluon predictor.
    
    Args:
        model: Any model object
        
    Returns:
        True if model is an AutoGluon predictor
    """
    class_name = type(model).__name__
    module_name = type(model).__module__
    
    autogluon_names = [
        'TabularPredictor', 'MultiModalPredictor', 'TimeSeriesPredictor',
        'TextPredictor', 'ImagePredictor', 'ObjectDetector'
    ]
    
    if any(name in class_name for name in autogluon_names):
        return True
    
    if 'autogluon' in module_name:
        return True
    
    return False
