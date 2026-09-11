"""
Visualization functions for model explainability.

Provides plotting functions for:
- Feature importance
- SHAP values
- Classification metrics (ROC, confusion matrix)
- Regression metrics (residuals)
- Dimensionality reduction (PCA, t-SNE)
- Time series forecasts
"""

from xai_core.visualizations.importance_plots import (
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_waterfall
)

from xai_core.visualizations.performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
    plot_predictions_vs_actual
)

from xai_core.visualizations.embedding_plots import (
    plot_pca_analysis,
    plot_embeddings_pca,
    plot_embeddings_tsne
)

from xai_core.visualizations.forecast_plots import (
    plot_forecast
)

__all__ = [
    # Importance
    'plot_feature_importance',
    'plot_shap_summary',
    'plot_shap_waterfall',
    # Performance
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_residuals',
    'plot_predictions_vs_actual',
    # Embeddings
    'plot_pca_analysis',
    'plot_embeddings_pca',
    'plot_embeddings_tsne',
    # Forecast
    'plot_forecast',
]
