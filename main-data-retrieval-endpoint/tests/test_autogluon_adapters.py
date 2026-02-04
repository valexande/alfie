"""Tests for AutoGluon adapters."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from xai_core.autogluon_adapters import (
    AutoGluonAdapter,
    TabularAdapter,
    MultiModalAdapter,
    TimeSeriesAdapter,
    TextAdapter,
    ImageAdapter,
    create_adapter,
    is_autogluon_predictor,
)


class TestTabularAdapter:
    """Tests for TabularAdapter."""
    
    def test_classification_detection(self):
        """Test classification problem type detection."""
        mock_predictor = Mock()
        mock_predictor.problem_type = 'binary'
        
        adapter = TabularAdapter(mock_predictor)
        assert adapter.problem_type == 'classification'
    
    def test_regression_detection(self):
        """Test regression problem type detection."""
        mock_predictor = Mock()
        mock_predictor.problem_type = 'regression'
        
        adapter = TabularAdapter(mock_predictor)
        assert adapter.problem_type == 'regression'
    
    def test_predict(self):
        """Test predict method."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = pd.Series([0, 1, 0, 1])
        
        adapter = TabularAdapter(mock_predictor)
        X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        
        result = adapter.predict(X)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        mock_predictor.predict.assert_called_once()
    
    def test_predict_proba_classification(self):
        """Test predict_proba for classification."""
        mock_predictor = Mock()
        mock_predictor.problem_type = 'binary'
        mock_predictor.predict_proba.return_value = pd.DataFrame({
            0: [0.8, 0.3, 0.6, 0.2],
            1: [0.2, 0.7, 0.4, 0.8]
        })
        
        adapter = TabularAdapter(mock_predictor)
        X = pd.DataFrame({'a': [1, 2, 3, 4]})
        
        result = adapter.predict_proba(X)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        assert adapter.classes_ is not None
    
    def test_predict_proba_regression_fails(self):
        """Test that predict_proba raises error for regression."""
        mock_predictor = Mock()
        mock_predictor.problem_type = 'regression'
        
        adapter = TabularAdapter(mock_predictor)
        X = pd.DataFrame({'a': [1, 2, 3, 4]})
        
        with pytest.raises(ValueError, match="predict_proba only available"):
            adapter.predict_proba(X)
    
    def test_fit_noop(self):
        """Test that fit is a no-op (returns self)."""
        mock_predictor = Mock()
        adapter = TabularAdapter(mock_predictor)
        
        X = pd.DataFrame({'a': [1, 2]})
        y = pd.Series([0, 1])
        
        result = adapter.fit(X, y)
        
        assert result is adapter


class TestTimeSeriesAdapter:
    """Tests for TimeSeriesAdapter."""
    
    def test_problem_type_is_forecasting(self):
        """Test that problem type is always forecasting."""
        mock_predictor = Mock()
        adapter = TimeSeriesAdapter(mock_predictor)
        
        assert adapter.problem_type == 'forecasting'
    
    def test_predict_proba_not_implemented(self):
        """Test that predict_proba raises NotImplementedError."""
        mock_predictor = Mock()
        adapter = TimeSeriesAdapter(mock_predictor)
        
        with pytest.raises(NotImplementedError):
            adapter.predict_proba(pd.DataFrame())


class TestCreateAdapter:
    """Tests for adapter factory function."""
    
    def test_tabular_predictor(self):
        """Test adapter creation for TabularPredictor."""
        mock_predictor = Mock()
        mock_predictor.__class__.__name__ = 'TabularPredictor'
        mock_predictor.__class__.__module__ = 'autogluon.tabular'
        
        adapter = create_adapter(mock_predictor)
        
        assert isinstance(adapter, TabularAdapter)
    
    def test_multimodal_predictor(self):
        """Test adapter creation for MultiModalPredictor."""
        mock_predictor = Mock()
        mock_predictor.__class__.__name__ = 'MultiModalPredictor'
        mock_predictor.__class__.__module__ = 'autogluon.multimodal'
        
        adapter = create_adapter(mock_predictor)
        
        assert isinstance(adapter, MultiModalAdapter)
    
    def test_timeseries_predictor(self):
        """Test adapter creation for TimeSeriesPredictor."""
        mock_predictor = Mock()
        mock_predictor.__class__.__name__ = 'TimeSeriesPredictor'
        mock_predictor.__class__.__module__ = 'autogluon.timeseries'
        
        adapter = create_adapter(mock_predictor)
        
        assert isinstance(adapter, TimeSeriesAdapter)
    
    def test_unknown_defaults_to_tabular(self):
        """Test that unknown predictor defaults to TabularAdapter."""
        mock_predictor = Mock()
        mock_predictor.__class__.__name__ = 'UnknownPredictor'
        mock_predictor.__class__.__module__ = 'some.module'
        
        adapter = create_adapter(mock_predictor)
        
        assert isinstance(adapter, TabularAdapter)


class TestIsAutogluonPredictor:
    """Tests for is_autogluon_predictor function."""
    
    def test_tabular_predictor_detected(self):
        """Test TabularPredictor is detected."""
        mock = Mock()
        mock.__class__.__name__ = 'TabularPredictor'
        mock.__class__.__module__ = 'autogluon.tabular'
        
        assert is_autogluon_predictor(mock) is True
    
    def test_sklearn_not_detected(self):
        """Test sklearn model is not detected as AutoGluon."""
        mock = Mock()
        mock.__class__.__name__ = 'RandomForestClassifier'
        mock.__class__.__module__ = 'sklearn.ensemble'
        
        assert is_autogluon_predictor(mock) is False
    
    def test_autogluon_module_detected(self):
        """Test model with autogluon in module is detected."""
        mock = Mock()
        mock.__class__.__name__ = 'SomeClass'
        mock.__class__.__module__ = 'autogluon.core.something'
        
        assert is_autogluon_predictor(mock) is True
