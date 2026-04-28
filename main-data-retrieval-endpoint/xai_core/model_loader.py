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
import json
import numpy as np
import pandas as pd

from xai_core.autogluon_adapters import (
    AutoGluonAdapter,
    create_adapter,
    is_autogluon_predictor,
)

warnings.filterwarnings('ignore')

# Optional PyTorch imports
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models as tv_models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Default preprocessing constants (ImageNet)
# ---------------------------------------------------------------------------
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD  = [0.229, 0.224, 0.225]
_DEFAULT_SIZE = 224   # square crop side


def _resolve_layer(model: "nn.Module", dotted_path: str) -> Optional["nn.Module"]:
    """
    Walk a dotted attribute path (e.g. ``"layer4.1"`` or ``"blocks.10"``)
    and return the corresponding sub-module, or None if not found.
    """
    obj = model
    for part in dotted_path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj if isinstance(obj, nn.Module) else None


def _auto_gradcam_layer(model: "nn.Module") -> Optional["nn.Module"]:
    """
    Heuristically pick the best GradCAM target layer for any architecture.

    Priority order:
    1. ``layer4`` last block          → ResNet / ResNeXt / Wide-ResNet
    2. ``features[-1]``               → VGG / AlexNet / MobileNetV2 / EfficientNet (old)
    3. ``blocks[-1]`` / ``stages[-1]``→ EfficientNetV2 / ConvNeXt / RegNet
    4. ``encoder.layers[-1]``         → ViT (uses LayerNorm as proxy)
    5. Last ``Conv2d`` in the tree    → generic fallback
    6. Last ``LayerNorm``             → transformer-only fallback
    """
    # ResNet family
    for attr in ("layer4", "layer3"):
        blk = getattr(model, attr, None)
        if blk is not None and isinstance(blk, nn.Module):
            children = list(blk.children())
            return children[-1] if children else blk

    # Sequential feature extractor (VGG, MobileNetV2, EfficientNet-old)
    features = getattr(model, "features", None)
    if features is not None:
        children = list(features.children())
        if children:
            return children[-1]

    # EfficientNetV2 / ConvNeXt / RegNet — look for blocks or stages
    for attr in ("blocks", "stages"):
        seq = getattr(model, attr, None)
        if seq is not None:
            children = list(seq.children())
            if children:
                last = children[-1]
                sub = list(last.children())
                return sub[-1] if sub else last

    # Vision Transformer — encoder.layers last block
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if layers is not None:
            children = list(layers.children())
            if children:
                return children[-1]

    # Generic: last Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return last_conv

    # Transformer-only fallback: last LayerNorm
    last_ln = None
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            last_ln = m
    return last_ln


@dataclass
class VisionModelInfo:
    """
    Container for a loaded PyTorch image-classification model.

    Attributes:
        model:            nn.Module in eval mode on CPU.
        labels:           {int_id: class_name_str}
        input_size:       Square crop side in pixels (default 224).
        mean:             Normalisation mean per channel (default ImageNet).
        std:              Normalisation std per channel (default ImageNet).
        gradcam_layer:    Dotted path to GradCAM target layer, or None for auto.
        architecture:     Human-readable name logged from metadata (optional).
        model_type:       Always ``"pytorch_vision"``.
        problem_type:     Always ``"classification"``.
        num_classes:      Number of output classes.
        errors:           Non-fatal loading errors.
    """
    model: Any
    labels: dict
    input_size: int = _DEFAULT_SIZE
    mean: List[float] = None
    std: List[float] = None
    gradcam_layer: Optional[str] = None   # dotted path or None → auto
    architecture: Optional[str] = None
    model_type: str = "pytorch_vision"
    problem_type: str = "classification"
    num_classes: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.mean is None:
            self.mean = _DEFAULT_MEAN
        if self.std is None:
            self.std = _DEFAULT_STD
        if self.num_classes == 0 and self.labels:
            self.num_classes = len(self.labels)

    def _build_transform(self) -> "transforms.Compose":
        """Build the torchvision preprocessing pipeline from stored config."""
        size = self.input_size
        return transforms.Compose([
            transforms.Resize(int(size * 256 / 224)),   # keep aspect ratio
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def get_gradcam_target(self) -> Optional["nn.Module"]:
        """
        Resolve the GradCAM target layer.

        Uses ``gradcam_layer`` dotted path if provided, otherwise auto-detects.
        """
        if self.gradcam_layer:
            layer = _resolve_layer(self.model, self.gradcam_layer)
            if layer is not None:
                return layer
            print(f"WARNING: gradcam_layer '{self.gradcam_layer}' not found, falling back to auto-detect")
        return _auto_gradcam_layer(self.model)

    def predict_image(self, image: "Image.Image") -> dict:
        """
        Run inference on a PIL image using the stored preprocessing config.

        Returns:
            {
              "predicted_class": str,
              "confidence": float,
              "probabilities": {class_name: float}
            }
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        transform = self._build_transform()
        tensor = transform(image.convert("RGB")).unsqueeze(0)  # (1, C, H, W)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        pred_idx = int(np.argmax(probs))
        pred_class = self.labels.get(pred_idx, self.labels.get(str(pred_idx), str(pred_idx)))

        return {
            "predicted_class": pred_class,
            "confidence": round(probs[pred_idx], 4),
            "probabilities": {
                self.labels.get(i, self.labels.get(str(i), str(i))): round(p, 4)
                for i, p in enumerate(probs)
            },
        }

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
        model_type: Type of model ('tabular', 'multimodal', 'timeseries', 'sklearn',
                    'pytorch_vision', etc.)
        problem_type: Problem type ('classification', 'regression', 'forecasting')
        is_autogluon: Whether the model is an AutoGluon predictor
        adapter: Sklearn-compatible adapter (only for AutoGluon models)
        errors: Any non-fatal errors encountered during loading
        model_version: Version of AutoGluon used to train the model (if available)
        current_version: Current installed AutoGluon version
        version_compatible: Whether versions are compatible for predictions
        vision_info: Populated for pytorch_vision models; holds the full VisionModelInfo
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
    vision_info: Optional[VisionModelInfo] = None

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


def load_vision_model_from_bytes(
    model_bytes: bytes,
    labels_bytes: bytes,
    filename: str = "model.pt",
) -> "VisionModelInfo":
    """
    Load *any* PyTorch image-classification model from a ``.pt`` file
    plus a ``labels.json`` sidecar.

    **labels.json format**

    Minimal (flat id→label mapping)::

        {"0": "cat", "1": "dog"}

    With AutoGluon-style wrapper::

        {"id2label": {"0": "cat", "1": "dog"}, "label2id": {...}}

    With optional ``model_config`` block (all fields optional)::

        {
            "id2label": {"0": "cardboard", "1": "glass", "2": "metal"},
            "model_config": {
                "architecture": "resnet50",
                "input_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225],
                "gradcam_layer": "layer4.2"
            }
        }

    ``architecture`` must be a name recognised by
    ``torchvision.models.get_model`` (e.g. ``"resnet18"``, ``"efficientnet_b0"``,
    ``"vit_b_16"``, ``"convnext_tiny"``).  If omitted the loader probes the
    state-dict key patterns to pick the most likely backbone.

    **Loading strategy**

    1. Try ``torch.load(...)`` as a full ``nn.Module``
       (model saved with ``torch.save(model, path)``).
    2. Try ``torch.load(...)`` as a state-dict and rebuild the backbone:
       a. Use ``architecture`` from ``model_config`` if present.
       b. Otherwise probe state-dict keys (layer4 → ResNet, blocks →
          EfficientNet/ConvNeXt, encoder.layers → ViT, features → VGG/MobileNet).
       c. Last resort: ResNet-18 with ``strict=False``.

    Args:
        model_bytes:  Raw bytes of the ``.pt`` file.
        labels_bytes: Raw bytes of the ``labels.json`` file.
        filename:     Original filename (logging only).

    Returns:
        :class:`VisionModelInfo` ready for inference and GradCAM.

    Raises:
        RuntimeError: If PyTorch is not installed.
        ValueError:   If the model cannot be loaded.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. "
            "Install it with: pip install torch torchvision pillow"
        )

    errors: List[str] = []

    # ------------------------------------------------------------------ labels
    try:
        raw = json.loads(labels_bytes.decode("utf-8"))
        if "id2label" in raw:
            id2label = {int(k): v for k, v in raw["id2label"].items()}
        else:
            id2label = {int(k): v for k, v in raw.items()
                        if k not in ("label2id", "model_config")}
        model_config: dict = raw.get("model_config", {})
    except Exception as e:
        raise ValueError(f"Could not parse labels.json: {e}")

    num_classes = len(id2label)

    # ---- preprocessing config (from model_config or ImageNet defaults) ----
    input_size    = int(model_config.get("input_size", _DEFAULT_SIZE))
    mean          = list(model_config.get("mean", _DEFAULT_MEAN))
    std           = list(model_config.get("std",  _DEFAULT_STD))
    gradcam_layer = model_config.get("gradcam_layer")        # dotted path or None
    arch_name     = model_config.get("architecture", "").strip().lower()

    # ------------------------------------------------------------------ model
    temp_dir  = tempfile.mkdtemp(prefix="xai_pt_model_")
    temp_path = Path(temp_dir) / filename

    try:
        with open(temp_path, "wb") as f:
            f.write(model_bytes)

        # ---- attempt 1: full saved model ----
        try:
            obj = torch.load(temp_path, map_location="cpu", weights_only=False)
            if isinstance(obj, nn.Module):
                obj.eval()
                print(f"Loaded full PyTorch model ({type(obj).__name__})")
                return VisionModelInfo(
                    model=obj,
                    labels=id2label,
                    input_size=input_size,
                    mean=mean,
                    std=std,
                    gradcam_layer=gradcam_layer,
                    architecture=arch_name or type(obj).__name__,
                    num_classes=num_classes,
                    errors=errors,
                )
        except Exception as e:
            errors.append(f"full-model load: {e}")

        # ---- attempt 2: state-dict → rebuild backbone ----
        try:
            state_dict = torch.load(temp_path, map_location="cpu", weights_only=True)
        except Exception as e:
            errors.append(f"state-dict load (weights_only=True): {e}")
            try:
                state_dict = torch.load(temp_path, map_location="cpu", weights_only=False)
                if not isinstance(state_dict, dict):
                    raise ValueError("Not a state-dict")
            except Exception as e2:
                errors.append(f"state-dict load (weights_only=False): {e2}")
                state_dict = None

        if isinstance(state_dict, dict):
            backbone = _build_backbone(arch_name, state_dict, num_classes, errors)
            if backbone is not None:
                backbone.load_state_dict(state_dict, strict=False)
                backbone.eval()
                print(f"Loaded state-dict into {type(backbone).__name__}")
                return VisionModelInfo(
                    model=backbone,
                    labels=id2label,
                    input_size=input_size,
                    mean=mean,
                    std=std,
                    gradcam_layer=gradcam_layer,
                    architecture=arch_name or type(backbone).__name__,
                    num_classes=num_classes,
                    errors=errors,
                )

        raise ValueError(
            f"Could not load PyTorch model from '{filename}'.\n"
            + "\n".join(errors)
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_backbone(
    arch_name: str,
    state_dict: dict,
    num_classes: int,
    errors: List[str],
) -> Optional["nn.Module"]:
    """
    Construct a torchvision backbone, replace its classification head with
    ``num_classes`` outputs, and return it (without loading weights yet).

    Resolution order:
    1. ``arch_name`` via ``torchvision.models.get_model`` (registry lookup).
    2. Key-pattern heuristics on ``state_dict``.
    3. ResNet-18 fallback (``strict=False`` will silence shape mismatches).
    """
    # ---- 1. explicit architecture name ----
    if arch_name:
        try:
            model = tv_models.get_model(arch_name, weights=None, num_classes=num_classes)
            print(f"Built backbone via registry: {arch_name}")
            return model
        except Exception as e:
            errors.append(f"registry build ({arch_name}): {e}")

    # ---- 2. heuristics on state-dict key patterns ----
    keys = set(state_dict.keys())

    def has(pattern: str) -> bool:
        return any(pattern in k for k in keys)

    try:
        # ResNet / ResNeXt / Wide-ResNet: layer1..layer4 + fc
        if has("layer4") and has(".fc."):
            # Distinguish depth by layer4 block count
            l4_keys = [k for k in keys if k.startswith("layer4.")]
            max_block = max(
                int(k.split(".")[1]) for k in l4_keys if k.split(".")[1].isdigit()
            )
            if max_block >= 2:
                base = tv_models.resnet50(weights=None)
            else:
                base = tv_models.resnet18(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            print(f"Heuristic: {type(base).__name__}")
            return base

        # EfficientNet-B* (old-style): features.N.block…
        if has("features.") and has("_expand_conv") or (has("features.") and has("_depthwise_conv")):
            base = tv_models.efficientnet_b0(weights=None)
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
            print("Heuristic: efficientnet_b0")
            return base

        # ConvNeXt: features.N.N.block
        if has("features.") and has(".block.") and has(".grn."):
            base = tv_models.convnext_tiny(weights=None)
            base.classifier[2] = nn.Linear(base.classifier[2].in_features, num_classes)
            print("Heuristic: convnext_tiny")
            return base

        # VGG / MobileNetV2: features.N + classifier
        if has("features.") and has("classifier."):
            base = tv_models.mobilenet_v2(weights=None)
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
            print("Heuristic: mobilenet_v2")
            return base

        # Vision Transformer: encoder.layers.encoder_layer_N
        if has("encoder.layers.encoder_layer_"):
            base = tv_models.vit_b_16(weights=None)
            base.heads.head = nn.Linear(base.heads.head.in_features, num_classes)
            print("Heuristic: vit_b_16")
            return base

        # DenseNet: features.denseblock
        if has("features.denseblock"):
            base = tv_models.densenet121(weights=None)
            base.classifier = nn.Linear(base.classifier.in_features, num_classes)
            print("Heuristic: densenet121")
            return base

        # MobileNetV3: features + classifier.3
        if has("features.") and has("classifier.3"):
            base = tv_models.mobilenet_v3_small(weights=None)
            base.classifier[3] = nn.Linear(base.classifier[3].in_features, num_classes)
            print("Heuristic: mobilenet_v3_small")
            return base

    except Exception as e:
        errors.append(f"heuristic backbone build: {e}")

    # ---- 3. fallback: ResNet-18 with strict=False ----
    print("WARNING: Could not identify backbone — falling back to ResNet-18 (strict=False)")
    errors.append("backbone unidentified; using resnet18 fallback with strict=False")
    base = tv_models.resnet18(weights=None)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base


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

    # Try vision-model bundle (model.pt + labels.json) — checked before AutoGluon
    if path.is_dir() and TORCH_AVAILABLE:
        model_info = _try_vision_bundle_load(path, errors)
        if model_info:
            return model_info

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

            # Try vision-model bundle first (model.pt + labels.json)
            if TORCH_AVAILABLE:
                model_info = _try_vision_bundle_load(extract_dir, errors)
                if model_info:
                    return model_info

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


def _try_vision_bundle_load(path: Path, errors: List[str]) -> Optional[ModelInfo]:
    """
    Detect a PyTorch vision-model bundle inside *path*.

    A bundle is recognised when the directory tree contains both:
      - ``model.pt`` or ``model.pth``  (the saved model / state-dict)
      - ``labels.json``                (the id→class mapping)

    Both files may live at any depth inside *path*.
    """
    if not TORCH_AVAILABLE:
        return None

    model_pt: Optional[Path] = None
    labels_json: Optional[Path] = None

    for root, _dirs, files in os.walk(path):
        files_lower = {f.lower(): f for f in files}
        if model_pt is None:
            for candidate in ('model.pt', 'model.pth'):
                if candidate in files_lower:
                    model_pt = Path(root) / files_lower[candidate]
                    break
        if labels_json is None and 'labels.json' in files_lower:
            labels_json = Path(root) / files_lower['labels.json']
        if model_pt is not None and labels_json is not None:
            break

    if model_pt is None or labels_json is None:
        return None

    try:
        vision_info = load_vision_model_from_bytes(
            model_pt.read_bytes(),
            labels_json.read_bytes(),
            model_pt.name,
        )
        print(f"Loaded PyTorch vision bundle: {model_pt.name}, "
              f"{len(vision_info.labels)} classes")
        return ModelInfo(
            model=vision_info.model,
            model_type='pytorch_vision',
            problem_type='classification',
            is_autogluon=False,
            adapter=None,
            vision_info=vision_info,
            errors=errors,
        )
    except Exception as exc:
        errors.append(f"vision bundle load: {exc}")
        return None


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
