"""
Linear Model Explainer - Optimized for linear models.

Supports:
- LogisticRegression
- LinearRegression
- Ridge, Lasso, ElasticNet
- SGDClassifier, SGDRegressor
- And other sklearn linear models

Uses coefficient-based importance and LinearExplainer for SHAP.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class LinearModelExplainer(BaseModelExplainer):
    """
    Explainer optimized for linear models.
    
    Uses coefficient magnitudes for feature importance and
    shap.LinearExplainer for exact SHAP values.
    
    Supported models:
    - sklearn.linear_model.LogisticRegression
    - sklearn.linear_model.LinearRegression
    - sklearn.linear_model.Ridge, RidgeClassifier
    - sklearn.linear_model.Lasso, LassoCV
    - sklearn.linear_model.ElasticNet, ElasticNetCV
    - sklearn.linear_model.SGDClassifier, SGDRegressor
    - sklearn.linear_model.BayesianRidge
    - sklearn.linear_model.Perceptron
    - sklearn.linear_model.PassiveAggressiveClassifier
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression().fit(X_train, y_train)
        >>> explainer = LinearModelExplainer(model, X_test, y_test)
        >>> importance = explainer.get_feature_importance()
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        **kwargs
    ):
        """
        Initialize linear model explainer.
        
        Args:
            model: Trained linear model
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self._shap_explainer = None
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'linear'
    
    @property
    def problem_type(self) -> str:
        """Detect if classification or regression."""
        # Check for predict_proba (classifier indicator)
        if hasattr(self.model, 'predict_proba'):
            return 'classification'
        
        # Check sklearn's _estimator_type
        if hasattr(self.model, '_estimator_type'):
            est_type = self.model._estimator_type
            if est_type == 'classifier':
                return 'classification'
            elif est_type == 'regressor':
                return 'regression'
        
        # Check class name
        class_name = type(self.model).__name__.lower()
        if 'classifier' in class_name or 'logistic' in class_name:
            return 'classification'
        
        return 'regression'
    
    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Get model coefficients."""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            # Handle multi-class (2D array)
            if len(coef.shape) > 1:
                # Average absolute coefficients across classes
                return np.abs(coef).mean(axis=0)
            return coef
        return None
    
    @property
    def intercept(self) -> Optional[float]:
        """Get model intercept."""
        if hasattr(self.model, 'intercept_'):
            intercept = self.model.intercept_
            if isinstance(intercept, np.ndarray):
                return float(intercept.mean())
            return float(intercept)
        return None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from coefficient magnitudes.
        
        For linear models, the absolute value of coefficients
        indicates feature importance (assuming standardized features).
        
        Returns:
            DataFrame with ['feature', 'importance', 'coefficient'] columns
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        coef = self.coefficients
        
        if coef is not None:
            # Use absolute coefficient values as importance
            abs_coef = np.abs(coef).flatten()
            
            self._feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': abs_coef,
                'coefficient': coef.flatten()
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            return self._feature_importance
        
        # Fallback to permutation importance
        return self._get_permutation_importance()
    
    def _get_permutation_importance(self) -> pd.DataFrame:
        """Compute permutation importance as fallback."""
        try:
            from sklearn.inspection import permutation_importance
            
            X_sample, y_sample = self.sample_data(min(500, self.max_samples))
            
            result = permutation_importance(
                self.model, X_sample, y_sample,
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
        Get SHAP values using LinearExplainer.
        
        LinearExplainer provides exact SHAP values for linear models
        and is very fast.
        
        Args:
            X_sample: Samples to explain (defaults to sampled X)
            
        Returns:
            np.ndarray of SHAP values or None if unavailable
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        try:
            if X_sample is None:
                X_sample, _ = self.sample_data(min(300, self.max_samples))
            
            # Ensure numeric data
            X_numeric = self._ensure_numeric(X_sample)
            
            # Create LinearExplainer (cached)
            if self._shap_explainer is None:
                print("Creating LinearExplainer...")
                # Use sampled data as background
                background = self._ensure_numeric(
                    self.X.sample(n=min(100, len(self.X)), random_state=42)
                )
                self._shap_explainer = shap.LinearExplainer(self.model, background)
            
            # Get SHAP values
            shap_values = self._shap_explainer.shap_values(X_numeric)
            
            return shap_values
            
        except Exception as e:
            print(f"LinearExplainer failed: {e}")
            # Try KernelExplainer as fallback
            return self._get_kernel_shap_values(X_sample)
    
    def _ensure_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric."""
        from sklearn.preprocessing import LabelEncoder
        
        X_numeric = X.copy()
        
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
                try:
                    le = LabelEncoder()
                    X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                except Exception:
                    X_numeric = X_numeric.drop(columns=[col])
        
        return X_numeric
    
    def _get_kernel_shap_values(self, X_sample: pd.DataFrame) -> Optional[np.ndarray]:
        """Fallback to KernelExplainer if LinearExplainer fails."""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            print("Falling back to KernelExplainer...")
            
            X_numeric = self._ensure_numeric(X_sample)
            background = self._ensure_numeric(
                self.X.sample(n=min(50, len(self.X)), random_state=42)
            )
            
            if self.is_classification and hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            explainer = shap.KernelExplainer(predict_fn, background)
            
            X_limited = X_numeric.head(min(50, len(X_numeric)))
            shap_values = explainer.shap_values(X_limited)
            
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            print(f"KernelExplainer also failed: {e}")
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
        
        # Add intercept info
        if self.intercept is not None:
            metrics['intercept'] = round(self.intercept, 4)
        
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
        
        try:
            metrics['mse'] = round(mean_squared_error(self.y, y_pred), 4)
        except Exception:
            pass
        
        return metrics
    
    def get_coefficient_summary(self) -> pd.DataFrame:
        """
        Get detailed coefficient summary.
        
        Returns DataFrame with feature names, coefficients, 
        absolute importance, and sign.
        """
        coef = self.coefficients
        
        if coef is None:
            return pd.DataFrame()
        
        coef_flat = coef.flatten()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coef_flat,
            'abs_coefficient': np.abs(coef_flat),
            'sign': np.where(coef_flat >= 0, '+', '-'),
            'impact': np.where(coef_flat >= 0, 'positive', 'negative')
        }).sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
