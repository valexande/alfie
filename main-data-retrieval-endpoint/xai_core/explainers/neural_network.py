"""
Neural Network Explainer - For PyTorch and TensorFlow/Keras models.

Uses DeepExplainer or GradientExplainer for SHAP values.
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

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class NeuralNetworkExplainer(BaseModelExplainer):
    """
    Explainer for neural network models (PyTorch, TensorFlow/Keras).
    
    Uses:
    - SHAP DeepExplainer for deep learning models
    - SHAP GradientExplainer as fallback
    - Permutation importance as final fallback
    
    Example:
        >>> # PyTorch
        >>> explainer = NeuralNetworkExplainer(pytorch_model, X, y, framework='pytorch')
        >>> 
        >>> # TensorFlow/Keras
        >>> explainer = NeuralNetworkExplainer(keras_model, X, y, framework='tensorflow')
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        framework: str = 'pytorch',
        **kwargs
    ):
        """
        Initialize neural network explainer.
        
        Args:
            model: Trained neural network model
            X: Feature DataFrame
            y: Target Series
            framework: 'pytorch' or 'tensorflow'
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self.framework = framework.lower()
        self._shap_explainer = None
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return self.framework
    
    @property
    def problem_type(self) -> str:
        """Detect problem type from output shape/type."""
        # Try to infer from model output
        try:
            X_sample = self.X.head(1)
            X_tensor = self._to_tensor(X_sample)
            
            if self.framework == 'pytorch':
                self.model.eval()
                with torch.no_grad():
                    output = self.model(X_tensor)
                    if output.shape[-1] > 1:
                        return 'classification'
            else:  # tensorflow
                output = self.model.predict(X_tensor, verbose=0)
                if len(output.shape) > 1 and output.shape[-1] > 1:
                    return 'classification'
        except Exception:
            pass
        
        # Fallback: check if target is categorical
        if self.y.dtype == 'object' or len(np.unique(self.y)) < 10:
            return 'classification'
        
        return 'regression'
    
    def _to_tensor(self, X: pd.DataFrame) -> Any:
        """Convert DataFrame to tensor for the framework."""
        X_numeric = self._ensure_numeric(X)
        
        if self.framework == 'pytorch':
            return torch.FloatTensor(X_numeric.values)
        else:  # tensorflow
            return X_numeric.values.astype(np.float32)
    
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
        
        return X_numeric.astype(np.float32)
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get model predictions."""
        if X is None:
            X = self.X
        
        X_tensor = self._to_tensor(X)
        
        try:
            if self.framework == 'pytorch':
                self.model.eval()
                with torch.no_grad():
                    output = self.model(X_tensor)
                    if self.is_classification:
                        return torch.argmax(output, dim=-1).numpy()
                    return output.numpy().flatten()
            else:  # tensorflow
                output = self.model.predict(X_tensor, verbose=0)
                if self.is_classification and len(output.shape) > 1:
                    return np.argmax(output, axis=-1)
                return output.flatten()
        except Exception as e:
            print(f"Prediction failed: {e}")
            return np.array([])
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities for classification."""
        if not self.is_classification:
            return None
        
        if X is None:
            X = self.X
        
        X_tensor = self._to_tensor(X)
        
        try:
            if self.framework == 'pytorch':
                self.model.eval()
                with torch.no_grad():
                    output = self.model(X_tensor)
                    proba = torch.softmax(output, dim=-1).numpy()
                    return proba
            else:  # tensorflow
                output = self.model.predict(X_tensor, verbose=0)
                # Apply softmax if needed
                if output.min() < 0 or output.max() > 1:
                    output = tf.nn.softmax(output).numpy()
                return output
        except Exception:
            return None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using permutation importance.
        
        Neural networks don't have native feature importance,
        so we use permutation-based importance.
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        try:
            from sklearn.inspection import permutation_importance
            
            X_sample, y_sample = self.sample_data(min(300, self.max_samples))
            
            # Create wrapper for sklearn
            class ModelWrapper:
                def __init__(self, explainer):
                    self.explainer = explainer
                
                def predict(self, X):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=self.explainer.feature_names)
                    return self.explainer.get_predictions(X)
            
            wrapper = ModelWrapper(self)
            
            result = permutation_importance(
                wrapper, 
                self._ensure_numeric(X_sample).values, 
                y_sample.values,
                n_repeats=5,
                random_state=42,
                n_jobs=1  # Neural nets often have issues with multiprocessing
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
        Get SHAP values using DeepExplainer or GradientExplainer.
        
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
            X_tensor = self._to_tensor(X_sample)
            
            # Get background data
            background = self._ensure_numeric(
                self.X.sample(n=min(50, len(self.X)), random_state=42)
            )
            background_tensor = self._to_tensor(background)
            
            if self.framework == 'pytorch':
                return self._get_pytorch_shap(X_tensor, background_tensor)
            else:
                return self._get_tensorflow_shap(X_tensor, background_tensor)
                
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return self._get_kernel_shap_fallback(X_sample)
    
    def _get_pytorch_shap(self, X_tensor, background_tensor) -> Optional[np.ndarray]:
        """Get SHAP values for PyTorch model."""
        try:
            print("Creating DeepExplainer for PyTorch model...")
            self.model.eval()
            explainer = shap.DeepExplainer(self.model, background_tensor)
            shap_values = explainer.shap_values(X_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            return shap_values
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            return None
    
    def _get_tensorflow_shap(self, X_tensor, background_tensor) -> Optional[np.ndarray]:
        """Get SHAP values for TensorFlow model."""
        try:
            print("Creating GradientExplainer for TensorFlow model...")
            explainer = shap.GradientExplainer(self.model, background_tensor)
            shap_values = explainer.shap_values(X_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            return shap_values
        except Exception as e:
            print(f"GradientExplainer failed: {e}")
            return None
    
    def _get_kernel_shap_fallback(self, X_sample: pd.DataFrame) -> Optional[np.ndarray]:
        """Fallback to KernelExplainer."""
        try:
            print("Falling back to KernelExplainer...")
            
            X_numeric = self._ensure_numeric(X_sample)
            background = self._ensure_numeric(
                self.X.sample(n=min(30, len(self.X)), random_state=42)
            )
            
            def predict_fn(X):
                X_df = pd.DataFrame(X, columns=self.feature_names)
                return self.get_predictions(X_df)
            
            explainer = shap.KernelExplainer(predict_fn, background.values)
            
            X_limited = X_numeric.head(min(30, len(X_numeric)))
            shap_values = explainer.shap_values(X_limited.values)
            
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
            'framework': self.framework,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
        }
        
        # Get predictions
        try:
            y_pred = self.get_predictions()
            if len(y_pred) == 0:
                self._metrics = metrics
                return metrics
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
        from sklearn.metrics import accuracy_score, f1_score
        
        metrics = {}
        
        try:
            metrics['accuracy'] = round(accuracy_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['f1'] = round(
                f1_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        return metrics
    
    def _get_regression_metrics(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_absolute_error, r2_score
        
        metrics = {}
        
        try:
            metrics['mae'] = round(mean_absolute_error(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['r2'] = round(r2_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        return metrics
