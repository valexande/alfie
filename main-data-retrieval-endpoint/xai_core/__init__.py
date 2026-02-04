"""
XAI Core - Universal Model Explainability

This package provides:
- Strategy Pattern based explainers for all model types
- AutoGluon adapters for sklearn compatibility
- Universal model loading for any AutoGluon or sklearn model
- Explainer factory with auto-detection
- Visualization and report generation
"""

from xai_core.autogluon_adapters import (
    AutoGluonAdapter,
    TabularAdapter,
    MultiModalAdapter,
    TimeSeriesAdapter,
    TextAdapter,
    ImageAdapter,
    create_adapter,
)
from xai_core.model_loader import load_model, ModelInfo
from xai_core.explainer_service import ExplainerService  # Legacy support
from xai_core.base_explainer import BaseModelExplainer
from xai_core.explainer_factory import ExplainerFactory, create_explainer
from xai_core.report_builder import ReportBuilder

# Import explainers
from xai_core.explainers import (
    TreeBasedExplainer,
    LinearModelExplainer,
    GenericExplainer,
)

__all__ = [
    # New Architecture
    "BaseModelExplainer",
    "ExplainerFactory",
    "create_explainer",
    "ReportBuilder",
    # Explainers
    "TreeBasedExplainer",
    "LinearModelExplainer",
    "GenericExplainer",
    # Legacy Adapters
    "AutoGluonAdapter",
    "TabularAdapter", 
    "MultiModalAdapter",
    "TimeSeriesAdapter",
    "TextAdapter",
    "ImageAdapter",
    "create_adapter",
    # Model loading
    "load_model",
    "ModelInfo",
    # Legacy Explainer
    "ExplainerService",
]
