"""
Tree-Based Model Explainer - Optimized for tree ensemble models.

Supports:
- XGBoost (XGBClassifier, XGBRegressor)
- LightGBM (LGBMClassifier, LGBMRegressor)
- CatBoost (CatBoostClassifier, CatBoostRegressor)
- Sklearn (RandomForest, GradientBoosting, ExtraTrees, DecisionTree, etc.)

Uses TreeExplainer for fast, exact SHAP values.
"""

from typing import Dict, Any, Optional, List
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


class TreeBasedExplainer(BaseModelExplainer):
    """
    Explainer optimized for tree-based models.
    
    Uses native feature_importances_ and shap.TreeExplainer for
    fast, exact explanations.
    
    Supported models:
    - XGBoost: XGBClassifier, XGBRegressor, XGBRanker
    - LightGBM: LGBMClassifier, LGBMRegressor, LGBMRanker
    - CatBoost: CatBoostClassifier, CatBoostRegressor
    - Sklearn: RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor,
               ExtraTreesClassifier/Regressor, DecisionTreeClassifier/Regressor,
               AdaBoostClassifier/Regressor, BaggingClassifier/Regressor,
               HistGradientBoostingClassifier/Regressor
    
    Example:
        >>> from xgboost import XGBClassifier
        >>> model = XGBClassifier().fit(X_train, y_train)
        >>> explainer = TreeBasedExplainer(model, X_test, y_test)
        >>> importance = explainer.get_feature_importance()
        >>> shap_values = explainer.get_shap_values(X_test.head(100))
    """
    
    SUPPORTED_SUBTYPES = ['xgboost', 'lightgbm', 'catboost', 'sklearn_tree']
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_subtype: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize tree-based explainer.
        
        Args:
            model: Trained tree-based model
            X: Feature DataFrame
            y: Target Series
            model_subtype: Optional explicit subtype ('xgboost', 'lightgbm', etc.)
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self._model_subtype = model_subtype or self._detect_subtype()
        self._shap_explainer = None
        self._patch_sklearn_compat()
    
    def _patch_sklearn_compat(self):
        """
        Patch sklearn models trained on older versions for forward compatibility.
        Adds missing attributes (e.g. monotonic_cst added in sklearn 1.4) so that
        predict() works when the model was serialised with an older sklearn.
        """
        def _patch_one(est):
            # monotonic_cst was added in sklearn 1.4 to DecisionTreeClassifier/Regressor
            if not hasattr(est, 'monotonic_cst'):
                est.monotonic_cst = None

        _patch_one(self.model)
        # Patch all trees in ensembles (RandomForest, ExtraTrees, GradientBoosting …)
        for attr in ('estimators_', 'estimators'):
            estimators = getattr(self.model, attr, None)
            if estimators is None:
                continue
            for item in estimators:
                # GradientBoosting stores a 2-D array of estimators
                if hasattr(item, '__iter__') and not hasattr(item, 'predict'):
                    for sub in item:
                        _patch_one(sub)
                else:
                    _patch_one(item)

    def _detect_subtype(self) -> str:
        """Detect specific tree model subtype."""
        class_name = type(self.model).__name__
        module = type(self.model).__module__
        
        if 'xgboost' in module.lower() or 'XGB' in class_name:
            return 'xgboost'
        elif 'lightgbm' in module.lower() or 'LGB' in class_name:
            return 'lightgbm'
        elif 'catboost' in module.lower() or 'CatBoost' in class_name:
            return 'catboost'
        else:
            return 'sklearn_tree'
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return self._model_subtype
    
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
        if 'classifier' in class_name or 'classification' in class_name:
            return 'classification'
        
        return 'regression'
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using native method.
        
        Tree models provide fast, reliable feature importance via
        the feature_importances_ attribute.
        
        Returns:
            DataFrame with ['feature', 'importance'] columns, sorted descending
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        # Try native feature_importances_ (all tree models have this)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            self._feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
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
            return None
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Get SHAP values using TreeExplainer (fast, exact).
        
        TreeExplainer is the optimal choice for tree-based models,
        providing exact SHAP values in polynomial time.
        
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
            
            # Create TreeExplainer (cached)
            if self._shap_explainer is None:
                print(f"Creating TreeExplainer for {self._model_subtype}...")
                self._shap_explainer = shap.TreeExplainer(self.model)
            
            # Get SHAP values
            shap_values = self._shap_explainer.shap_values(X_sample)
            
            # Handle multi-class output (list of arrays)
            if isinstance(shap_values, list):
                # For binary classification, take positive class
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    # For multiclass, use first class or average
                    shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            print(f"TreeExplainer failed: {e}")
            # Try KernelExplainer as fallback
            return self._get_kernel_shap_values(X_sample)
    
    def _get_kernel_shap_values(self, X_sample: pd.DataFrame) -> Optional[np.ndarray]:
        """Fallback to KernelExplainer if TreeExplainer fails."""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            print("Falling back to KernelExplainer (slower)...")
            
            # Create background data
            background = self.X.sample(n=min(50, len(self.X)), random_state=42)
            
            # Create prediction function
            if self.is_classification and hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            # Create KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Get SHAP values (limited samples due to slowness)
            X_limited = X_sample.head(min(50, len(X_sample)))
            shap_values = explainer.shap_values(X_limited)
            
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            print(f"KernelExplainer also failed: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Returns appropriate metrics for classification or regression.
        """
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
        
        # ROC AUC (requires probabilities)
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get additional model-specific information."""
        info = {
            'model_class': type(self.model).__name__,
            'model_subtype': self._model_subtype,
        }
        
        # XGBoost specific
        if self._model_subtype == 'xgboost':
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info['max_depth'] = self.model.max_depth
            if hasattr(self.model, 'learning_rate'):
                info['learning_rate'] = self.model.learning_rate
        
        # LightGBM specific
        elif self._model_subtype == 'lightgbm':
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'num_leaves'):
                info['num_leaves'] = self.model.num_leaves
            if hasattr(self.model, 'learning_rate'):
                info['learning_rate'] = self.model.learning_rate
        
        # CatBoost specific
        elif self._model_subtype == 'catboost':
            if hasattr(self.model, 'tree_count_'):
                info['n_trees'] = self.model.tree_count_
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                info['depth'] = params.get('depth')
                info['learning_rate'] = params.get('learning_rate')
        
        # Sklearn specific
        elif self._model_subtype == 'sklearn_tree':
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info['max_depth'] = self.model.max_depth
            if hasattr(self.model, 'min_samples_split'):
                info['min_samples_split'] = self.model.min_samples_split
        
        return info
