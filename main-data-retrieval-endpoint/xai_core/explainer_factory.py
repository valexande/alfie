"""
Explainer Factory - Creates appropriate explainer based on model type.

Supports:
- Auto-detection of model type
- Manual registration of custom explainers
- Graceful fallback to generic explainer
"""

from typing import Any, Dict, Type, Optional
import pandas as pd
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')


class ExplainerFactory:
    """
    Factory for creating model-specific explainers.
    
    Features:
    - Auto-detects model type and creates optimal explainer
    - Allows registration of custom explainers
    - Falls back to GenericExplainer for unknown models
    
    Example:
        >>> from xai_core.explainer_factory import ExplainerFactory
        >>> explainer = ExplainerFactory.create(model, X, y)
        >>> report = explainer.generate_report(mode='expert')
        
        # Register custom explainer
        >>> ExplainerFactory.register('my_model', MyCustomExplainer)
    """
    
    # Registry for custom explainers
    _registry: Dict[str, Type[BaseModelExplainer]] = {}
    
    # Built-in explainer mappings
    _builtin_mappings: Dict[str, str] = {
        # AutoGluon
        'autogluon_tabular': 'autogluon_tabular',
        'autogluon_multimodal': 'autogluon_multimodal',
        'autogluon_timeseries': 'autogluon_timeseries',

        # Raw-text sklearn pipelines (vectorizer + classifier)
        'sklearn_text': 'sklearn_text',
        
        # Tree-based (use TreeBasedExplainer)
        'xgboost': 'tree_based',
        'lightgbm': 'tree_based',
        'catboost': 'tree_based',
        'sklearn_tree': 'tree_based',
        
        # Linear (use LinearModelExplainer)
        'linear': 'linear',
        
        # Neural networks
        'pytorch': 'neural_network',
        'tensorflow': 'neural_network',

        # PyTorch image classifiers (loaded as VisionModelInfo)
        'pytorch_vision': 'pytorch_vision',
        
        # SVM, KNN, etc. (use GenericExplainer)
        'svm': 'generic',
        'knn': 'generic',
        'naive_bayes': 'generic',
        'generic': 'generic',
    }
    
    @classmethod
    def register(cls, model_type: str, explainer_class: Type[BaseModelExplainer]):
        """
        Register a custom explainer for a model type.
        
        Args:
            model_type: Identifier for the model type
            explainer_class: Explainer class to use
            
        Example:
            >>> ExplainerFactory.register('my_custom_model', MyCustomExplainer)
        """
        cls._registry[model_type] = explainer_class
        print(f"Registered explainer for '{model_type}': {explainer_class.__name__}")
    
    @classmethod
    def unregister(cls, model_type: str):
        """Remove a registered explainer."""
        if model_type in cls._registry:
            del cls._registry[model_type]
    
    @classmethod
    def create(
        cls, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: Optional[str] = None,
        **kwargs
    ) -> BaseModelExplainer:
        """
        Create appropriate explainer for the given model.
        
        Args:
            model: Trained model to explain
            X: Feature DataFrame
            y: Target Series
            model_type: Optional explicit model type (auto-detected if not provided)
            **kwargs: Additional configuration for the explainer
            
        Returns:
            BaseModelExplainer instance
            
        Example:
            >>> explainer = ExplainerFactory.create(rf_model, X_test, y_test)
            >>> explainer = ExplainerFactory.create(model, X, y, model_type='xgboost')
        """
        # Auto-detect model type if not provided
        if model_type is None:
            model_type = cls.detect_model_type(model)
        
        print(f"Creating explainer for model type: {model_type}")
        
        # Check custom registry first (allows overrides)
        if model_type in cls._registry:
            return cls._registry[model_type](model, X, y, **kwargs)
        
        # Get built-in explainer category
        explainer_category = cls._builtin_mappings.get(model_type, 'generic')
        
        # Import and create the appropriate explainer
        return cls._create_builtin_explainer(
            explainer_category, model, X, y, model_type, **kwargs
        )
    
    @classmethod
    def _create_builtin_explainer(
        cls,
        category: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        **kwargs
    ) -> BaseModelExplainer:
        """Create a built-in explainer by category."""
        
        if category == 'autogluon_tabular':
            from xai_core.explainers.autogluon_tabular import AutoGluonTabularExplainer
            return AutoGluonTabularExplainer(model, X, y, **kwargs)
        
        elif category == 'autogluon_multimodal':
            from xai_core.explainers.autogluon_multimodal import AutoGluonMultiModalExplainer
            return AutoGluonMultiModalExplainer(model, X, y, **kwargs)
        
        elif category == 'autogluon_timeseries':
            from xai_core.explainers.autogluon_timeseries import AutoGluonTimeSeriesExplainer
            return AutoGluonTimeSeriesExplainer(model, X, y, **kwargs)

        elif category == 'sklearn_text':
            from xai_core.explainers.sklearn_text import SklearnTextExplainer
            return SklearnTextExplainer(model, X, y, **kwargs)
        
        elif category == 'tree_based':
            from xai_core.explainers.tree_based import TreeBasedExplainer
            return TreeBasedExplainer(model, X, y, model_subtype=model_type, **kwargs)
        
        elif category == 'linear':
            from xai_core.explainers.linear import LinearModelExplainer
            return LinearModelExplainer(model, X, y, **kwargs)
        
        elif category == 'neural_network':
            from xai_core.explainers.neural_network import NeuralNetworkExplainer
            return NeuralNetworkExplainer(model, X, y, framework=model_type, **kwargs)

        elif category == 'pytorch_vision':
            from xai_core.explainers.vision_classifier import VisionClassifierExplainer
            return VisionClassifierExplainer(model, X, y, **kwargs)
        
        else:  # generic fallback
            from xai_core.explainers.generic import GenericExplainer
            return GenericExplainer(model, X, y, **kwargs)
    
    @classmethod
    def detect_model_type(cls, model: Any) -> str:
        """
        Detect model type from model instance.
        
        Args:
            model: Model instance to identify
            
        Returns:
            str: Model type identifier
        """
        class_name = type(model).__name__
        module = type(model).__module__
        
        print(f"Detecting model type: class={class_name}, module={module}")

        # A fitted sklearn Pipeline/ColumnTransformer that contains a text
        # vectorizer must receive raw strings, not category codes.
        if cls.is_sklearn_text_pipeline(model):
            return 'sklearn_text'
        
        # =====================================================================
        # AutoGluon Predictors
        # =====================================================================
        if 'TabularPredictor' in class_name or 'autogluon.tabular' in module:
            return 'autogluon_tabular'
        
        if 'MultiModalPredictor' in class_name or 'autogluon.multimodal' in module:
            return 'autogluon_multimodal'
        
        if 'TimeSeriesPredictor' in class_name or 'autogluon.timeseries' in module:
            return 'autogluon_timeseries'
        
        # =====================================================================
        # XGBoost
        # =====================================================================
        if 'xgboost' in module.lower() or 'XGB' in class_name:
            return 'xgboost'
        
        # =====================================================================
        # LightGBM
        # =====================================================================
        if 'lightgbm' in module.lower() or 'LGB' in class_name:
            return 'lightgbm'
        
        # =====================================================================
        # CatBoost
        # =====================================================================
        if 'catboost' in module.lower() or 'CatBoost' in class_name:
            return 'catboost'
        
        # =====================================================================
        # Sklearn Tree-Based Models
        # =====================================================================
        sklearn_tree_names = [
            'RandomForest', 'GradientBoosting', 'AdaBoost',
            'ExtraTrees', 'DecisionTree', 'BaggingClassifier',
            'BaggingRegressor', 'HistGradientBoosting'
        ]
        if any(name in class_name for name in sklearn_tree_names):
            return 'sklearn_tree'
        
        # =====================================================================
        # Sklearn Linear Models
        # =====================================================================
        linear_names = [
            'LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso',
            'ElasticNet', 'SGDClassifier', 'SGDRegressor', 'Perceptron',
            'PassiveAggressive', 'BayesianRidge', 'ARDRegression',
            'HuberRegressor', 'TheilSen', 'RANSACRegressor'
        ]
        if any(name in class_name for name in linear_names):
            return 'linear'
        
        # =====================================================================
        # SVM Models
        # =====================================================================
        if any(name in class_name for name in ['SVC', 'SVR', 'NuSVC', 'NuSVR', 'LinearSVC', 'LinearSVR']):
            return 'svm'
        
        # =====================================================================
        # KNN Models
        # =====================================================================
        if 'KNeighbors' in class_name or 'KNN' in class_name.upper():
            return 'knn'
        
        # =====================================================================
        # Naive Bayes
        # =====================================================================
        if 'NaiveBayes' in class_name or 'GaussianNB' in class_name or 'MultinomialNB' in class_name:
            return 'naive_bayes'
        
        # =====================================================================
        # PyTorch Models
        # =====================================================================
        if 'torch' in module.lower():
            return 'pytorch'
        if any(name in class_name for name in ['Module', 'Sequential']) and 'torch' in str(type(model).__bases__):
            return 'pytorch'
        
        # =====================================================================
        # TensorFlow / Keras Models
        # =====================================================================
        if 'keras' in module.lower() or 'tensorflow' in module.lower():
            return 'tensorflow'
        if any(name in class_name for name in ['Sequential', 'Model', 'Functional']):
            # Check if it's actually a Keras model
            try:
                import tensorflow as tf
                if isinstance(model, tf.keras.Model):
                    return 'tensorflow'
            except ImportError:
                pass
        
        # =====================================================================
        # Sklearn Ensemble (non-tree)
        # =====================================================================
        if any(name in class_name for name in ['VotingClassifier', 'VotingRegressor', 'StackingClassifier', 'StackingRegressor']):
            return 'generic'  # These need special handling
        
        # =====================================================================
        # Fallback: Check for common sklearn patterns
        # =====================================================================
        if 'sklearn' in module:
            # Check for tree-like properties
            if hasattr(model, 'feature_importances_'):
                return 'sklearn_tree'
            if hasattr(model, 'coef_'):
                return 'linear'
        
        # =====================================================================
        # Default fallback
        # =====================================================================
        print(f"Unknown model type: {class_name} from {module}, using generic explainer")
        return 'generic'

    @staticmethod
    def is_sklearn_text_pipeline(model: Any) -> bool:
        """Return True when an estimator graph contains a text vectorizer."""
        vectorizer_names = {
            'TfidfVectorizer', 'CountVectorizer', 'HashingVectorizer'
        }
        seen = set()
        stack = [model]

        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            if type(current).__name__ in vectorizer_names:
                return True

            named_steps = getattr(current, 'named_steps', None)
            if named_steps:
                stack.extend(named_steps.values())

            transformers = getattr(current, 'transformers', None)
            if transformers:
                stack.extend(item[1] for item in transformers if len(item) >= 2)
            transformers_ = getattr(current, 'transformers_', None)
            if transformers_:
                stack.extend(item[1] for item in transformers_ if len(item) >= 2)

        return False
    
    @classmethod
    def get_supported_types(cls) -> Dict[str, str]:
        """Get all supported model types and their explainer categories."""
        return {
            **cls._builtin_mappings,
            **{k: 'custom' for k in cls._registry.keys()}
        }
    
    @classmethod
    def is_supported(cls, model: Any) -> bool:
        """Check if a model type is explicitly supported (not just generic fallback)."""
        model_type = cls.detect_model_type(model)
        return model_type != 'generic' or model_type in cls._registry


# Convenience function
def create_explainer(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    **kwargs
) -> BaseModelExplainer:
    """
    Convenience function to create an explainer.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        **kwargs: Additional configuration
        
    Returns:
        Appropriate BaseModelExplainer instance
    """
    return ExplainerFactory.create(model, X, y, **kwargs)
