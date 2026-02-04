"""
AutoGluon Tabular Explainer - Optimized for AutoGluon TabularPredictor.

Uses native AutoGluon feature_importance() and adapters for SHAP.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class AutoGluonTabularExplainer(BaseModelExplainer):
    """
    Explainer optimized for AutoGluon TabularPredictor.
    
    Uses:
    - Native predictor.feature_importance() for fast, reliable importance
    - SHAP via KernelExplainer (TreeExplainer not directly compatible)
    - Native predictor methods for predictions
    
    Example:
        >>> from autogluon.tabular import TabularPredictor
        >>> predictor = TabularPredictor.load("my_model")
        >>> explainer = AutoGluonTabularExplainer(predictor, X_test, y_test)
        >>> importance = explainer.get_feature_importance()
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AutoGluon Tabular explainer.
        
        Args:
            model: TabularPredictor instance
            X: Feature DataFrame
            y: Target Series
            label: Target column name (for feature_importance)
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self.predictor = model
        self.label = label or (y.name if hasattr(y, 'name') and y.name else 'target')
        
        # Build full DataFrame for AutoGluon methods
        self._full_data = X.copy()
        self._full_data[self.label] = y.values
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'autogluon_tabular'
    
    @property
    def problem_type(self) -> str:
        """Get problem type from predictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        elif ag_type in ['regression', 'quantile']:
            return 'regression'
        
        return 'regression'
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get predictions using native predictor."""
        if X is None:
            X = self.X
        
        predictions = self.predictor.predict(X)
        
        if hasattr(predictions, 'values'):
            return predictions.values
        return np.array(predictions)
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not self.is_classification:
            return None
        
        if X is None:
            X = self.X
        
        try:
            proba = self.predictor.predict_proba(X)
            if isinstance(proba, pd.DataFrame):
                return proba.values
            return proba
        except Exception:
            return None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using native AutoGluon method.
        
        AutoGluon's feature_importance() uses permutation-based
        importance which is model-agnostic and reliable.
        
        Returns:
            DataFrame with ['feature', 'importance'] columns
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        try:
            print("Computing native AutoGluon feature importance...")
            
            # Use native feature_importance method
            importance = self.predictor.feature_importance(
                data=self._full_data,
                subsample_size=min(self.max_samples, len(self._full_data)),
                num_shuffle_sets=5,
                silent=True
            )
            
            if isinstance(importance, pd.DataFrame):
                # Rename columns if needed
                if 'importance' in importance.columns:
                    importance = importance.reset_index()
                    importance.columns = ['feature', 'importance'] + list(importance.columns[2:])
                else:
                    importance = importance.reset_index()
                    importance.columns = ['feature'] + list(importance.columns[1:])
                    if len(importance.columns) > 1:
                        importance['importance'] = importance.iloc[:, 1]
                
                self._feature_importance = importance[['feature', 'importance']].sort_values(
                    'importance', ascending=False
                ).reset_index(drop=True)
                
                return self._feature_importance
                
        except Exception as e:
            print(f"AutoGluon feature_importance failed: {e}")
        
        # Fallback to permutation importance
        return self._get_permutation_importance()
    
    def _get_permutation_importance(self) -> pd.DataFrame:
        """Compute permutation importance as fallback."""
        try:
            from sklearn.inspection import permutation_importance
            
            X_sample, y_sample = self.sample_data(min(500, self.max_samples))
            
            # Create wrapper for sklearn compatibility
            class PredictorWrapper:
                def __init__(self, predictor):
                    self.predictor = predictor
                
                def predict(self, X):
                    return self.predictor.predict(X)
            
            wrapper = PredictorWrapper(self.predictor)
            
            result = permutation_importance(
                wrapper, X_sample, y_sample,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            self._feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            return self._feature_importance
            
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': [1.0 / self.n_features] * self.n_features
            })
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Get SHAP values using KernelExplainer.
        
        Note: AutoGluon's ensemble models don't support TreeExplainer directly,
        so we use KernelExplainer which works with any model.
        
        Args:
            X_sample: Samples to explain
            
        Returns:
            np.ndarray of SHAP values
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        try:
            if X_sample is None:
                X_sample, _ = self.sample_data(min(100, self.max_samples))
            
            print("Creating SHAP KernelExplainer for AutoGluon...")
            
            # Create prediction wrapper
            def predict_fn(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=self.feature_names)
                result = self.predictor.predict(X)
                return result.values if hasattr(result, 'values') else np.array(result)
            
            # Sample background data
            background = self.X.sample(n=min(50, len(self.X)), random_state=42)
            
            # Create KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Limit samples for performance
            X_limited = X_sample.head(min(50, len(X_sample)))
            
            # Get SHAP values
            shap_values = explainer.shap_values(X_limited, nsamples=100)
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        if self._metrics is not None:
            return self._metrics
        
        metrics = {
            'model_type': self.model_type,
            'problem_type': self.problem_type,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
        }
        
        # Get predictions
        try:
            y_pred = self.get_predictions()
        except Exception as e:
            print(f"Prediction failed: {e}")
            self._metrics = metrics
            return metrics
        
        if self.is_classification:
            metrics.update(self._get_classification_metrics(y_pred))
        else:
            metrics.update(self._get_regression_metrics(y_pred))
        
        # Add AutoGluon-specific info
        metrics.update(self._get_autogluon_info())
        
        self._metrics = metrics
        return metrics
    
    def _get_classification_metrics(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {}
        
        try:
            metrics['accuracy'] = round(accuracy_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['precision'] = round(
                precision_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        try:
            metrics['recall'] = round(
                recall_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        try:
            metrics['f1'] = round(
                f1_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        # ROC AUC
        try:
            y_proba = self.get_prediction_probabilities()
            if y_proba is not None:
                if len(self.classes) == 2:
                    metrics['roc_auc'] = round(roc_auc_score(self.y, y_proba[:, 1]), 4)
                else:
                    metrics['roc_auc'] = round(
                        roc_auc_score(self.y, y_proba, multi_class='ovr', average='weighted'), 4
                    )
        except Exception:
            pass
        
        return metrics
    
    def _get_regression_metrics(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score
        )
        
        metrics = {}
        
        try:
            metrics['mae'] = round(mean_absolute_error(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['rmse'] = round(np.sqrt(mean_squared_error(self.y, y_pred)), 4)
        except Exception:
            pass
        
        try:
            metrics['r2'] = round(r2_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        return metrics
    
    def _get_autogluon_info(self) -> Dict[str, Any]:
        """Get AutoGluon-specific model information."""
        info = {}
        
        try:
            info['best_model'] = self.predictor.get_model_best()
        except Exception:
            pass
        
        try:
            info['model_names'] = self.predictor.get_model_names()
        except Exception:
            pass
        
        return info
    
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """Get AutoGluon model leaderboard."""
        try:
            return self.predictor.leaderboard(silent=True)
        except Exception:
            return None
