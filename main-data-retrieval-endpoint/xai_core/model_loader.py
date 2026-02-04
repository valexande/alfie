"""
Universal Model Loader - Loads any AutoGluon or sklearn-compatible model.

Supports:
- AutoGluon predictors (TabularPredictor, MultiModalPredictor, TimeSeriesPredictor, etc.)
- Pickled sklearn models
- Joblib-serialized models
- ZIP files containing AutoGluon models
"""

from pathlib import Path
from typing import Any, Tuple, List, Optional, Union
from dataclasses import dataclass
import tempfile
import zipfile
import shutil
import os
import warnings
import sys

import joblib
import pickle
import numpy as np
import pandas as pd

from xai_core.autogluon_adapters import (
    AutoGluonAdapter,
    create_adapter,
    is_autogluon_predictor,
)

warnings.filterwarnings('ignore')

# Windows compatibility fix for PosixPath in AutoGluon models
if sys.platform == 'win32':
    from pathlib import WindowsPath
    import pathlib
    
    class CompatiblePosixPath(WindowsPath):
        """Windows-compatible PosixPath for loading Linux-trained AutoGluon models."""
        def __new__(cls, *args, **kwargs):
            if args:
                path_str = str(args[0]).replace('/', '\\')
                if path_str.startswith('\\') and len(path_str) > 1 and path_str[1] != '\\':
                    path_str = path_str[1:]
                args = (path_str,) + args[1:]
            return WindowsPath.__new__(WindowsPath, *args, **kwargs)
    
    pathlib.PosixPath = CompatiblePosixPath

# Optional AutoGluon imports
AUTOGLUON_AVAILABLE = False
TabularPredictor = None
MultiModalPredictor = None
TimeSeriesPredictor = None

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    pass

try:
    from autogluon.multimodal import MultiModalPredictor
except ImportError:
    pass

try:
    from autogluon.timeseries import TimeSeriesPredictor
except ImportError:
    pass


@dataclass
class ModelInfo:
    """
    Container for loaded model information.
    
    Attributes:
        model: The raw model object (AutoGluon predictor or sklearn model)
        model_type: Type of model ('tabular', 'multimodal', 'timeseries', 'sklearn', etc.)
        problem_type: Problem type ('classification', 'regression', 'forecasting')
        is_autogluon: Whether the model is an AutoGluon predictor
        adapter: Sklearn-compatible adapter (only for AutoGluon models)
        errors: Any non-fatal errors encountered during loading
        model_version: Version of AutoGluon used to train the model (if available)
        current_version: Current installed AutoGluon version
        version_compatible: Whether versions are compatible for predictions
    """
    model: Any
    model_type: str
    problem_type: str
    is_autogluon: bool
    adapter: Optional[AutoGluonAdapter] = None
    errors: List[str] = None
    model_version: Optional[str] = None
    current_version: Optional[str] = None
    version_compatible: bool = True
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def sklearn_compatible_model(self) -> Any:
        """
        Get sklearn-compatible model for use with explainerdashboard.
        
        Returns adapter for AutoGluon models, raw model for sklearn.
        """
        if self.is_autogluon and self.adapter:
            return self.adapter
        return self.model
    
    @property
    def has_version_mismatch(self) -> bool:
        """Check if there's a version mismatch that might cause issues."""
        if not self.is_autogluon:
            return False
        if self.model_version and self.current_version:
            # Major version mismatch is problematic
            model_major = self.model_version.split('.')[0]
            current_major = self.current_version.split('.')[0]
            if model_major != current_major:
                return True
            # Minor version mismatch might cause issues
            model_minor = self.model_version.split('.')[1] if '.' in self.model_version else '0'
            current_minor = self.current_version.split('.')[1] if '.' in self.current_version else '0'
            return model_minor != current_minor
        return not self.version_compatible


def load_model(model_path: Union[str, Path, bytes]) -> ModelInfo:
    """
    Universal model loader for any AutoGluon or sklearn model.
    
    Automatically detects model type and creates appropriate adapter.
    
    Args:
        model_path: Path to model file/directory, or bytes from uploaded file
        
    Returns:
        ModelInfo with model, type info, and sklearn-compatible adapter
        
    Raises:
        ValueError: If model cannot be loaded
        
    Example:
        >>> model_info = load_model("./my_autogluon_model")
        >>> model = model_info.sklearn_compatible_model
        >>> predictions = model.predict(X_test)
    """
    errors = []
    
    # Handle bytes input (from file upload)
    if isinstance(model_path, bytes):
        return _load_from_bytes(model_path, errors)
    
    path = Path(model_path).resolve()
    
    if not path.exists():
        raise ValueError(f"Model path does not exist: {path}")
    
    # Handle ZIP files
    if path.suffix == '.zip' or _is_zip_file(path):
        path = _extract_zip(path)
    
    # Try AutoGluon directory loading
    if path.is_dir() and AUTOGLUON_AVAILABLE:
        model_info = _try_autogluon_load(path, errors)
        if model_info:
            return model_info
    
    # Try pickle/joblib file loading
    if path.is_file():
        model_info = _try_pickle_load(path, errors)
        if model_info:
            return model_info
    
    # All loading attempts failed
    raise ValueError(f"Could not load model from {model_path}. Errors:\n" + "\n".join(errors))


def load_model_from_bytes(
    model_bytes: bytes, 
    filename: str = "model.pkl"
) -> ModelInfo:
    """
    Load model from bytes (e.g., from file upload).
    
    Args:
        model_bytes: Raw bytes of the model file
        filename: Original filename (used to determine file type)
        
    Returns:
        ModelInfo with loaded model
    """
    errors = []
    
    # Create temp file/directory
    temp_dir = tempfile.mkdtemp(prefix='xai_model_')
    temp_path = Path(temp_dir) / filename
    
    try:
        # Write bytes to temp file
        with open(temp_path, 'wb') as f:
            f.write(model_bytes)
        
        # Check if it's a ZIP file
        if filename.endswith('.zip') or _is_zip_file(temp_path):
            extract_dir = _extract_zip(temp_path)
            
            # Try AutoGluon load from extracted directory
            if AUTOGLUON_AVAILABLE:
                model_info = _try_autogluon_load(extract_dir, errors)
                if model_info:
                    return model_info
        
        # Try pickle/joblib load
        model_info = _try_pickle_load(temp_path, errors)
        if model_info:
            return model_info
        
        raise ValueError(f"Could not load model from bytes. Errors:\n" + "\n".join(errors))
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _try_autogluon_load(path: Path, errors: List[str]) -> Optional[ModelInfo]:
    """Try loading as various AutoGluon predictor types."""
    
    # Check for predictor.pkl to confirm it's an AutoGluon directory
    if not (path / 'predictor.pkl').exists():
        # Search subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / 'predictor.pkl').exists():
                path = subdir
                break
        else:
            errors.append(f"No predictor.pkl found in {path}")
            return None
    
    # Read model version from metadata.json or version.txt
    model_version = _get_model_version(path)
    current_version = _get_current_autogluon_version()
    
    # Predictor classes to try (in order of likelihood)
    predictor_classes = []
    
    if TabularPredictor:
        predictor_classes.append(('tabular', TabularPredictor))
    if MultiModalPredictor:
        predictor_classes.append(('multimodal', MultiModalPredictor))
    if TimeSeriesPredictor:
        predictor_classes.append(('timeseries', TimeSeriesPredictor))
    
    for model_type, predictor_class in predictor_classes:
        try:
            print(f"Attempting to load as {predictor_class.__name__}...")
            # Use path relative to parent to avoid AutoGluon absolute path issues
            import os
            original_cwd = os.getcwd()
            os.chdir(path.parent)
            try:
                predictor = predictor_class.load(
                    path.name,  # Use relative path (just folder name)
                    require_py_version_match=False,
                    require_version_match=False
                )
            finally:
                os.chdir(original_cwd)
            
            # Create sklearn-compatible adapter
            adapter = create_adapter(predictor)
            
            # Test if predictor can make predictions (version compatibility check)
            version_compatible = _test_predictor_compatibility(predictor, path)
            
            print(f"Successfully loaded as {model_type} predictor")
            if not version_compatible:
                print(f"WARNING: Model version ({model_version}) differs from installed ({current_version})")
            
            return ModelInfo(
                model=predictor,
                model_type=model_type,
                problem_type=adapter.problem_type,
                is_autogluon=True,
                adapter=adapter,
                errors=errors,
                model_version=model_version,
                current_version=current_version,
                version_compatible=version_compatible
            )
            
        except Exception as e:
            errors.append(f"{predictor_class.__name__}: {str(e)}")
    
    return None


def _get_model_version(path: Path) -> Optional[str]:
    """Extract AutoGluon version from model metadata."""
    # Try version.txt first
    version_file = path / 'version.txt'
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except:
            pass
    
    # Try metadata.json
    metadata_file = path / 'metadata.json'
    if metadata_file.exists():
        try:
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            return metadata.get('version')
        except:
            pass
    
    return None


def _get_current_autogluon_version() -> Optional[str]:
    """Get currently installed AutoGluon version."""
    try:
        import autogluon.tabular as agt
        return agt.__version__
    except:
        return None


def _test_predictor_compatibility(predictor, path: Path) -> bool:
    """Test if predictor can make predictions without version errors."""
    try:
        # Try to get feature metadata
        if hasattr(predictor, 'feature_metadata'):
            _ = predictor.feature_metadata
        
        # Try a minimal prediction test using saved validation data if available
        val_data_path = path / 'utils' / 'data' / 'X_val.pkl'
        if val_data_path.exists():
            try:
                X_test = pd.read_pickle(val_data_path)
                # Test with just 1 row
                _ = predictor.predict(X_test.head(1))
                return True
            except AttributeError as e:
                if 'passthrough' in str(e) or 'AsTypeFeatureGenerator' in str(e):
                    return False
                raise
        
        return True  # Assume compatible if we can't test
        
    except AttributeError as e:
        if 'passthrough' in str(e) or 'AsTypeFeatureGenerator' in str(e):
            return False
        return True
    except Exception:
        return True  # Assume compatible for other errors


def _try_pickle_load(path: Path, errors: List[str]) -> Optional[ModelInfo]:
    """Try loading as pickle/joblib file."""
    
    # Try joblib first (better for sklearn models)
    try:
        print(f"Attempting to load with joblib...")
        model = joblib.load(path)
        
        # Check if it's actually an AutoGluon predictor
        if is_autogluon_predictor(model):
            adapter = create_adapter(model)
            return ModelInfo(
                model=model,
                model_type=_detect_autogluon_type(model),
                problem_type=adapter.problem_type,
                is_autogluon=True,
                adapter=adapter,
                errors=errors
            )
        
        return ModelInfo(
            model=model,
            model_type=_detect_sklearn_model_type(model),
            problem_type=_detect_problem_type(model),
            is_autogluon=False,
            adapter=None,
            errors=errors
        )
        
    except Exception as e:
        errors.append(f"joblib: {str(e)}")
    
    # Try pickle with various encodings
    for encoding in [None, 'latin1']:
        try:
            print(f"Attempting to load with pickle (encoding={encoding})...")
            with open(path, 'rb') as f:
                if encoding:
                    model = pickle.load(f, encoding=encoding)
                else:
                    model = pickle.load(f)
            
            # Check if it's an AutoGluon predictor
            if is_autogluon_predictor(model):
                adapter = create_adapter(model)
                return ModelInfo(
                    model=model,
                    model_type=_detect_autogluon_type(model),
                    problem_type=adapter.problem_type,
                    is_autogluon=True,
                    adapter=adapter,
                    errors=errors
                )
            
            return ModelInfo(
                model=model,
                model_type=_detect_sklearn_model_type(model),
                problem_type=_detect_problem_type(model),
                is_autogluon=False,
                adapter=None,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"pickle({encoding}): {str(e)}")
    
    return None


def _extract_zip(zip_path: Path) -> Path:
    """Extract ZIP file to temp directory and return path to predictor."""
    temp_dir = tempfile.mkdtemp(prefix='autogluon_model_')
    
    print(f"Extracting ZIP to {temp_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    
    temp_path = Path(temp_dir)
    
    # Find predictor directory (contains predictor.pkl)
    for root, dirs, files in os.walk(temp_dir):
        if 'predictor.pkl' in files:
            return Path(root)
    
    return temp_path


def _is_zip_file(path: Path) -> bool:
    """Check if file is a ZIP file by reading magic bytes."""
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'
    except:
        return False


def _detect_problem_type(model: Any) -> str:
    """Detect if model is classifier or regressor."""
    # Check for predict_proba (classifier indicator)
    if hasattr(model, 'predict_proba'):
        return 'classification'
    
    # Check sklearn's _estimator_type
    if hasattr(model, '_estimator_type'):
        est_type = model._estimator_type
        if est_type == 'classifier':
            return 'classification'
        elif est_type == 'regressor':
            return 'regression'
    
    # Check class name
    class_name = type(model).__name__.lower()
    if 'classifier' in class_name or 'classification' in class_name:
        return 'classification'
    
    return 'regression'


def _detect_sklearn_model_type(model: Any) -> str:
    """Detect sklearn model type from class name."""
    class_name = type(model).__name__.lower()
    
    if any(x in class_name for x in ['forest', 'tree', 'gbm', 'gradient']):
        return 'tree_ensemble'
    if any(x in class_name for x in ['xgb', 'xgboost']):
        return 'xgboost'
    if any(x in class_name for x in ['lgb', 'lightgbm']):
        return 'lightgbm'
    if 'catboost' in class_name:
        return 'catboost'
    if any(x in class_name for x in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
        return 'linear'
    if any(x in class_name for x in ['svm', 'svc', 'svr']):
        return 'svm'
    if any(x in class_name for x in ['neural', 'mlp', 'nn']):
        return 'neural_network'
    if any(x in class_name for x in ['kneighbors', 'knn']):
        return 'knn'
    if any(x in class_name for x in ['naive', 'bayes']):
        return 'naive_bayes'
    
    return 'sklearn_unknown'


def _detect_autogluon_type(model: Any) -> str:
    """Detect AutoGluon predictor type."""
    class_name = type(model).__name__
    
    if 'Tabular' in class_name:
        return 'tabular'
    if 'MultiModal' in class_name:
        return 'multimodal'
    if 'TimeSeries' in class_name:
        return 'timeseries'
    if 'Text' in class_name:
        return 'text'
    if 'Image' in class_name:
        return 'image'
    
    return 'autogluon_unknown'
