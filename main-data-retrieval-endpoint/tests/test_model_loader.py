"""Tests for model loader."""

import pytest
import tempfile
import pickle
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from xai_core.model_loader import (
    load_model,
    ModelInfo,
    _detect_problem_type,
    _detect_sklearn_model_type,
    _is_zip_file,
)


class TestDetectProblemType:
    """Tests for problem type detection."""
    
    def test_classifier_with_predict_proba(self):
        """Test classifier detection via predict_proba."""
        mock_model = Mock()
        mock_model.predict_proba = Mock()
        
        result = _detect_problem_type(mock_model)
        
        assert result == 'classification'
    
    def test_classifier_with_estimator_type(self):
        """Test classifier detection via _estimator_type."""
        mock_model = Mock(spec=[])  # No predict_proba
        mock_model._estimator_type = 'classifier'
        
        result = _detect_problem_type(mock_model)
        
        assert result == 'classification'
    
    def test_regressor_default(self):
        """Test regressor is default."""
        mock_model = Mock(spec=[])  # No predict_proba or _estimator_type
        
        result = _detect_problem_type(mock_model)
        
        assert result == 'regression'


class TestDetectSklearnModelType:
    """Tests for sklearn model type detection."""
    
    def test_random_forest(self):
        """Test RandomForest detection."""
        mock = Mock()
        mock.__class__.__name__ = 'RandomForestClassifier'
        
        result = _detect_sklearn_model_type(mock)
        
        assert result == 'tree_ensemble'
    
    def test_xgboost(self):
        """Test XGBoost detection."""
        mock = Mock()
        mock.__class__.__name__ = 'XGBClassifier'
        
        result = _detect_sklearn_model_type(mock)
        
        assert result == 'xgboost'
    
    def test_lightgbm(self):
        """Test LightGBM detection."""
        mock = Mock()
        mock.__class__.__name__ = 'LGBMClassifier'
        
        result = _detect_sklearn_model_type(mock)
        
        assert result == 'lightgbm'
    
    def test_linear(self):
        """Test linear model detection."""
        mock = Mock()
        mock.__class__.__name__ = 'LogisticRegression'
        
        result = _detect_sklearn_model_type(mock)
        
        assert result == 'linear'
    
    def test_unknown(self):
        """Test unknown model type."""
        mock = Mock()
        mock.__class__.__name__ = 'SomeUnknownModel'
        
        result = _detect_sklearn_model_type(mock)
        
        assert result == 'sklearn_unknown'


class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_sklearn_compatible_model_for_sklearn(self):
        """Test sklearn_compatible_model returns raw model for sklearn."""
        mock_model = Mock()
        info = ModelInfo(
            model=mock_model,
            model_type='tree_ensemble',
            problem_type='classification',
            is_autogluon=False,
            adapter=None
        )
        
        assert info.sklearn_compatible_model is mock_model
    
    def test_sklearn_compatible_model_for_autogluon(self):
        """Test sklearn_compatible_model returns adapter for AutoGluon."""
        mock_model = Mock()
        mock_adapter = Mock()
        
        info = ModelInfo(
            model=mock_model,
            model_type='tabular',
            problem_type='classification',
            is_autogluon=True,
            adapter=mock_adapter
        )
        
        assert info.sklearn_compatible_model is mock_adapter
    
    def test_errors_default_empty_list(self):
        """Test errors defaults to empty list."""
        info = ModelInfo(
            model=Mock(),
            model_type='test',
            problem_type='classification',
            is_autogluon=False
        )
        
        assert info.errors == []


class TestLoadModel:
    """Tests for load_model function."""
    
    def test_load_pickle_model(self):
        """Test loading a pickled sklearn model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and save a simple model
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            # Load the model
            model_info = load_model(temp_path)
            
            assert model_info is not None
            assert model_info.model_type == 'tree_ensemble'
            assert model_info.problem_type == 'classification'
            assert model_info.is_autogluon is False
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_path_raises_error(self):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            load_model("/nonexistent/path/model.pkl")


class TestIsZipFile:
    """Tests for ZIP file detection."""
    
    def test_zip_file_detected(self):
        """Test ZIP file is detected."""
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with zipfile.ZipFile(f.name, 'w') as zf:
                zf.writestr('test.txt', 'content')
            temp_path = f.name
        
        try:
            result = _is_zip_file(Path(temp_path))
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_non_zip_not_detected(self):
        """Test non-ZIP file is not detected."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            f.write(b'not a zip file')
            temp_path = f.name
        
        try:
            result = _is_zip_file(Path(temp_path))
            assert result is False
        finally:
            os.unlink(temp_path)
