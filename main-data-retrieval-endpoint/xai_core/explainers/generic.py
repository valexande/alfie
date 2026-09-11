"""
Generic Model Explainer - Fallback for any sklearn-compatible model.

Works with any model that has predict() and optionally predict_proba().
Uses KernelExplainer for SHAP (slow but universal).
"""

from typing import Dict, Any, Optional
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


class GenericExplainer(BaseModelExplainer):
    """
    Generic explainer for any sklearn-compatible model.
    
    This is the fallback explainer used when no specific explainer
    is available for a model type. It works with any model that
    implements the sklearn interface (predict, optionally predict_proba).
    
    Uses:
    - Permutation importance (model-agnostic)
    - SHAP KernelExplainer (slow but works with any model)
    
    Supported models:
    - Any sklearn model
    - Any model with predict() method
    - SVM, KNN, Naive Bayes, etc.
    - Custom models with sklearn interface
    
    Example:
        >>> from sklearn.svm import SVC
        >>> model = SVC(probability=True).fit(X_train, y_train)
        >>> explainer = GenericExplainer(model, X_test, y_test)
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
        Initialize generic explainer.
        
        Args:
            model: Any sklearn-compatible model
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self._shap_explainer = None
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'generic'
    
    @property
    def problem_type(self) -> str:
        """Detect problem type from model."""
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
        if 'classifier' in class_name or 'classification' in class_name:
            return 'classification'
        if 'regressor' in class_name or 'regression' in class_name:
            return 'regression'
        
        # Fallback: check target variable
        if self.y.dtype == 'object' or len(np.unique(self.y)) < 10:
            return 'classification'
        
        return 'regression'
    
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
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get model predictions."""
        if X is None:
            X = self.X
        
        X_numeric = self._ensure_numeric(X)
        
        # Handle feature alignment if model has feature_names_in_
        if hasattr(self.model, 'feature_names_in_'):
            expected = list(self.model.feature_names_in_)
            if set(expected) != set(X_numeric.columns):
                # Try to align features
                available = [f for f in expected if f in X_numeric.columns]
                if len(available) == len(expected):
                    X_numeric = X_numeric[expected]
        
        predictions = self.model.predict(X_numeric)
        
        if hasattr(predictions, 'values'):
            return predictions.values
        return np.array(predictions)
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities for classification."""
        if not self.is_classification:
            return None
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        if X is None:
            X = self.X
        
        X_numeric = self._ensure_numeric(X)
        
        try:
            proba = self.model.predict_proba(X_numeric)
            return proba
        except Exception:
            return None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using permutation importance.
        
        Permutation importance is model-agnostic and works with any
        model that has a predict() method.
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        # Check for native feature importance first
        if hasattr(self.model, 'feature_importances_'):
            try:
                self._feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False).reset_index(drop=True)
                return self._feature_importance
            except Exception:
                pass
        
        # Check for coefficients (linear models)
        if hasattr(self.model, 'coef_'):
            try:
                coef = self.model.coef_
                if len(coef.shape) > 1:
                    coef = np.abs(coef).mean(axis=0)
                
                self._feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(coef).flatten()
                }).sort_values('importance', ascending=False).reset_index(drop=True)
                return self._feature_importance
            except Exception:
                pass
        
        # Fallback to permutation importance
        return self._get_permutation_importance()
    
    def _get_permutation_importance(self) -> pd.DataFrame:
        """Compute permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            X_sample, y_sample = self.sample_data(min(500, self.max_samples))
            X_numeric = self._ensure_numeric(X_sample)
            
            result = permutation_importance(
                self.model, X_numeric, y_sample,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            self._feature_importance = pd.DataFrame({
                'feature': list(X_numeric.columns),
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            return self._feature_importance
            
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            return None
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Get SHAP values using KernelExplainer.
        
        KernelExplainer is the most universal SHAP explainer but
        is computationally expensive. Use with limited samples.
        
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
            
            X_numeric = self._ensure_numeric(X_sample)
            
            print("Creating SHAP KernelExplainer (this may take a while)...")
            
            # Get background data
            background = self._ensure_numeric(
                self.X.sample(n=min(50, len(self.X)), random_state=42)
            )
            
            # Create prediction function
            if self.is_classification and hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)
            else:
                predict_fn = lambda x: self.model.predict(x)
            
            # Create KernelExplainer
            self._shap_explainer = shap.KernelExplainer(predict_fn, background)
            
            # Limit samples for performance
            X_limited = X_numeric.head(min(50, len(X_numeric)))
            
            # Get SHAP values
            shap_values = self._shap_explainer.shap_values(X_limited)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]
            
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
            'model_class': type(self.model).__name__,
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
