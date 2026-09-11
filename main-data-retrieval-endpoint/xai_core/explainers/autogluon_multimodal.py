"""
AutoGluon MultiModal Explainer - For MultiModalPredictor.

Handles text, image, and tabular multimodal data.
Focus on embedding extraction and visualization.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')

# Optional imports
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from autogluon.multimodal import MultiModalPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class AutoGluonMultiModalExplainer(BaseModelExplainer):
    """
    Explainer for AutoGluon MultiModalPredictor.
    
    MultiModal models handle:
    - Text data
    - Image data
    - Tabular data
    - Combinations of the above
    
    Focus on:
    - Embedding extraction and visualization
    - PCA and t-SNE projections
    - Basic prediction metrics
    
    Note: SHAP is limited for multimodal models due to complexity.
    
    Example:
        >>> from autogluon.multimodal import MultiModalPredictor
        >>> predictor = MultiModalPredictor.load("my_model")
        >>> explainer = AutoGluonMultiModalExplainer(predictor, X_test, y_test)
        >>> embeddings = explainer.extract_embeddings()
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        **kwargs
    ):
        """
        Initialize MultiModal explainer.
        
        Args:
            model: MultiModalPredictor instance
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional configuration
        """
        super().__init__(model, X, y, **kwargs)
        self.predictor = model
        self._embeddings = None
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'autogluon_multimodal'
    
    @property
    def problem_type(self) -> str:
        """Get problem type from predictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        
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
    
    def extract_embeddings(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Extract embeddings from the multimodal model.
        
        Embeddings capture learned representations of the input data.
        
        Args:
            X: Data to extract embeddings from
            
        Returns:
            np.ndarray of embeddings
        """
        if X is None:
            X, _ = self.sample_data(min(500, self.max_samples))
        
        try:
            print("Extracting embeddings from MultiModal model...")
            embeddings = self.predictor.extract_embedding(X)
            
            if isinstance(embeddings, pd.DataFrame):
                embeddings = embeddings.values
            
            self._embeddings = embeddings
            return embeddings
            
        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return None
    
    def get_feature_importance(self):
        """
        Feature importance is not applicable to AutoGluon MultiModal models.

        MultiModal predictors learn from unstructured inputs (images, text, …)
        using deep neural networks; there are no tabular feature weights to rank.
        Returning None causes the report to skip the section rather than show
        fabricated equal-weight bars.
        """
        print("Feature importance is not applicable for MultiModal models — skipping.")
        return None
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        SHAP values are not well-supported for MultiModal models.
        
        Returns None with a warning.
        """
        print("Warning: SHAP analysis is limited for MultiModal models.")
        print("Consider using embedding visualization instead.")
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
        
        # Add embedding info
        if self._embeddings is not None:
            metrics['embedding_dim'] = self._embeddings.shape[1]
        
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
    
    def get_embedding_pca(self, n_components: int = 2) -> Optional[np.ndarray]:
        """
        Get PCA projection of embeddings.
        
        Args:
            n_components: Number of PCA components
            
        Returns:
            np.ndarray of shape (n_samples, n_components)
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        embeddings = self._embeddings
        if embeddings is None:
            embeddings = self.extract_embeddings()
        
        if embeddings is None:
            return None
        
        try:
            pca = PCA(n_components=min(n_components, embeddings.shape[1]))
            return pca.fit_transform(embeddings)
        except Exception as e:
            print(f"PCA failed: {e}")
            return None
    
    def get_embedding_tsne(self, n_components: int = 2) -> Optional[np.ndarray]:
        """
        Get t-SNE projection of embeddings.
        
        Args:
            n_components: Number of t-SNE components
            
        Returns:
            np.ndarray of shape (n_samples, n_components)
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        embeddings = self._embeddings
        if embeddings is None:
            embeddings = self.extract_embeddings()
        
        if embeddings is None:
            return None
        
        try:
            tsne = TSNE(n_components=n_components, random_state=42)
            return tsne.fit_transform(embeddings)
        except Exception as e:
            print(f"t-SNE failed: {e}")
            return None
    
    def generate_plots(self) -> Dict[str, str]:
        """Generate plots including embedding visualizations."""
        from xai_core.visualizations import (
            plot_confusion_matrix,
            plot_embeddings_pca,
            plot_embeddings_tsne
        )
        
        plots = {}
        
        # Classification plots
        if self.is_classification:
            try:
                y_pred = self.get_predictions()
                plots['confusion_matrix'] = plot_confusion_matrix(
                    self.y, y_pred, self.classes
                )
            except Exception:
                pass
        
        # Embedding visualizations
        embeddings = self.extract_embeddings()
        if embeddings is not None:
            y_sample, _ = self.sample_data(min(500, self.max_samples))
            y_aligned = self.y.iloc[:len(embeddings)] if len(self.y) > len(embeddings) else self.y
            
            try:
                pca_plot = plot_embeddings_pca(embeddings, y_aligned)
                if pca_plot:
                    plots['embeddings_pca'] = pca_plot
            except Exception:
                pass
            
            try:
                tsne_plot = plot_embeddings_tsne(embeddings, y_aligned)
                if tsne_plot:
                    plots['embeddings_tsne'] = tsne_plot
            except Exception:
                pass
        
        return plots
