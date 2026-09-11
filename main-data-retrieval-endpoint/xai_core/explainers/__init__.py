"""
Model-specific explainers package.

Each explainer is optimized for a specific model type or family.
"""

from xai_core.explainers.tree_based import TreeBasedExplainer
from xai_core.explainers.linear import LinearModelExplainer
from xai_core.explainers.generic import GenericExplainer

# AutoGluon explainers (may not be available)
try:
    from xai_core.explainers.autogluon_tabular import AutoGluonTabularExplainer
except ImportError:
    AutoGluonTabularExplainer = None

try:
    from xai_core.explainers.autogluon_multimodal import AutoGluonMultiModalExplainer
except ImportError:
    AutoGluonMultiModalExplainer = None

try:
    from xai_core.explainers.autogluon_timeseries import AutoGluonTimeSeriesExplainer
except ImportError:
    AutoGluonTimeSeriesExplainer = None

# Neural network explainers (may not be available)
try:
    from xai_core.explainers.neural_network import NeuralNetworkExplainer
except ImportError:
    NeuralNetworkExplainer = None

__all__ = [
    'TreeBasedExplainer',
    'LinearModelExplainer',
    'GenericExplainer',
    'AutoGluonTabularExplainer',
    'AutoGluonMultiModalExplainer',
    'AutoGluonTimeSeriesExplainer',
    'NeuralNetworkExplainer',
]
