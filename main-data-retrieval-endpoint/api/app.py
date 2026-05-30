"""
FastAPI Application for XAI Explainability API.

Provides endpoints for:
- Universal model explainability (AutoGluon + sklearn)
- Data interpretability and analysis
- UC2-specific driver analysis
"""

from contextlib import asynccontextmanager
from typing import Optional
import io
import sys
import os
import zipfile
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    UserLevel,
    HealthResponse,
    ErrorResponse,
    ModelInfoResponse
)
from api.config import settings
from xai_core.model_loader import load_model_from_bytes, AUTOGLUON_AVAILABLE
from xai_core.explainer_service import ExplainerService  # Legacy support
from xai_core.explainer_factory import ExplainerFactory, create_explainer
from xai_core.utils import detect_target_column
from xai_core.data_interpretability_service import DataInterpretabilityService


def _load_vision_dataset(data_bytes: bytes, vision_info) -> tuple:
    """
    Extract a labelled-image dataset ZIP and build (X, y) for VisionClassifierExplainer.

    The ZIP is expected to have the structure produced by the training pipeline:

        {split}/metadata.csv          — columns: filename, label
        {split}/{filename}            — image files

    Split preference: ``test/`` → ``train/`` → any other split → top-level.

    Parameters
    ----------
    data_bytes  : raw bytes of the dataset ZIP file
    vision_info : VisionModelInfo with the ``labels`` id→name mapping

    Returns
    -------
    (X, y, tmp_dir)
        X       — pd.DataFrame with column ``image_path`` (absolute paths)
        y       — pd.Series of integer class-ids matching vision_info.labels
        tmp_dir — str path of temp extraction dir (caller should clean up)
    """
    tmp_dir = tempfile.mkdtemp(prefix="xai_vision_dataset_")
    try:
        # Write ZIP to temp file and extract
        zip_path = Path(tmp_dir) / "dataset.zip"
        zip_path.write_bytes(data_bytes)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        zip_path.unlink(missing_ok=True)

        extract_root = Path(tmp_dir)

        # Find the best metadata.csv
        candidates = []
        for meta in extract_root.rglob("metadata.csv"):
            parts = meta.relative_to(extract_root).parts
            split = parts[0].lower() if len(parts) > 1 else "root"
            priority = {"test": 0, "val": 1, "validation": 1, "train": 2, "drift": 3}.get(split, 4)
            candidates.append((priority, meta))

        if not candidates:
            raise ValueError(
                "No metadata.csv found in dataset ZIP. "
                "Expected structure: {split}/metadata.csv with columns filename,label"
            )

        best_meta = sorted(candidates, key=lambda x: x[0])[0][1]
        split_dir = best_meta.parent          # e.g. /tmp/xai_.../test
        print(f"Using dataset split: {split_dir.name} ({best_meta})")

        df = pd.read_csv(best_meta)
        if "filename" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"metadata.csv must have 'filename' and 'label' columns. "
                f"Found: {list(df.columns)}"
            )

        # Build label → id mapping from vision_info
        label2id = {name: cid for cid, name in vision_info.labels.items()}

        rows = []
        skipped = 0
        for _, row in df.iterrows():
            label_name = str(row["label"]).strip()
            if label_name not in label2id:
                skipped += 1
                continue
            img_path = split_dir / str(row["filename"])
            if not img_path.exists():
                skipped += 1
                continue
            rows.append({"image_path": str(img_path), "label_id": label2id[label_name]})

        if not rows:
            raise ValueError(
                f"No valid image-label pairs found in {best_meta}. "
                f"({skipped} rows skipped due to missing files or unknown labels)"
            )

        if skipped:
            print(f"  - Skipped {skipped} rows (missing files or unknown labels)")

        dataset_df = pd.DataFrame(rows)
        X = dataset_df[["image_path"]]
        y = dataset_df["label_id"].reset_index(drop=True)
        X = X.reset_index(drop=True)

        print(f"  - Vision dataset: {len(X)} images, {y.nunique()} classes")
        return X, y, tmp_dir

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _align_autogluon_features(predictor, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align X to the exact feature set the AutoGluon predictor was trained on.

    Handles three scenarios:
    1. X already has all expected columns → just reorder/select.
    2. X has raw categorical columns that were OHE'd before training
       (e.g. CSV has ``gender`` but model expects ``gender_Female``,
       ``gender_Male``) → apply pd.get_dummies then align.
    3. X has garbage columns (file paths, leaky text labels, split flags, …)
       → drop silently.

    This runs before DataInterpretabilityService so the data analysis also
    only sees the real model features, not metadata columns.
    """
    try:
        expected = list(predictor.feature_metadata.type_map_raw.keys())
    except Exception as e:
        print(f"  Could not read predictor.feature_metadata: {e}")
        return X

    if not expected:
        return X

    # Case 1: X already has all expected columns
    if all(c in X.columns for c in expected):
        dropped = [c for c in X.columns if c not in expected]
        if dropped:
            print(
                f"  AutoGluon feature alignment: dropping {len(dropped)} non-model "
                f"column(s): {dropped[:8]}{'...' if len(dropped) > 8 else ''}"
            )
        return X[expected]

    # Case 2 / 3: mix of raw categoricals and garbage columns
    raw_ohe_cols = [
        c for c in X.columns
        if c not in expected and any(exp.startswith(f"{c}_") for exp in expected)
    ]
    garbage_cols = [
        c for c in X.columns
        if c not in expected and c not in raw_ohe_cols
    ]

    if garbage_cols:
        print(
            f"  AutoGluon feature alignment: dropping {len(garbage_cols)} non-model "
            f"column(s): {garbage_cols[:8]}{'...' if len(garbage_cols) > 8 else ''}"
        )
    X = X.drop(columns=garbage_cols, errors='ignore')

    if raw_ohe_cols:
        print(f"  AutoGluon feature alignment: OHE-encoding {raw_ohe_cols}")
        X = pd.get_dummies(X, columns=raw_ohe_cols)

    # Add expected OHE columns missing due to unseen categories
    for col in expected:
        if col not in X.columns:
            X[col] = 0

    available = [c for c in expected if c in X.columns]
    return X[available]


# Check available features
FEATURES = {
    "shap": False,
    "autogluon_tabular": False,
    "autogluon_multimodal": False,
    "autogluon_timeseries": False,
    "autogluon_eda": True,  # Always available via fallback
    "data_interpretability": True,  # Always available - standalone data analysis
}

try:
    import shap
    FEATURES["shap"] = True
except ImportError:
    pass

try:
    from autogluon.tabular import TabularPredictor
    FEATURES["autogluon_tabular"] = True
except ImportError:
    pass

try:
    from autogluon.multimodal import MultiModalPredictor
    FEATURES["autogluon_multimodal"] = True
except ImportError:
    pass

try:
    from autogluon.timeseries import TimeSeriesPredictor
    FEATURES["autogluon_timeseries"] = True
except ImportError:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("=" * 60)
    print("XAI Explainability API Starting...")
    print("=" * 60)
    print(f"Features available:")
    for feature, available in FEATURES.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")
    print("=" * 60)
    yield
    print("XAI API shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
Universal Model Explainability API

Supports:
- **AutoGluon models**: TabularPredictor, MultiModalPredictor, TimeSeriesPredictor
- **sklearn models**: RandomForest, XGBoost, LightGBM, CatBoost, Linear models, etc.

Features:
- **Feature Importance**: Native AutoGluon feature_importance() or sklearn permutation importance
- **SHAP Explanations**: KernelExplainer-based SHAP values
- **Beginner/Expert Reports**: Different detail levels for different audiences
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPError",
            detail=str(exc.detail),
            suggestions=[]
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
            suggestions=["Check server logs for details"]
        ).model_dump()
    )


# ============================================================================
# Health Check
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health():
    """
    Check API health and available features.

    Returns information about:
    - Service status
    - Available explainability features
    - AutoGluon predictor support
    """
    return HealthResponse(
        status="healthy",
        service="xai-explainability",
        version=settings.app_version,
        features=FEATURES
    )



@app.post(
    "/explain-model",
    response_class=HTMLResponse,
    tags=["Explainability"],
    summary="Generate explainability report for any model",
    responses={
        200: {"description": "HTML explainability report"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def explain_model(
    model_file: UploadFile = File(
        ...,
        description="Model file (pickle, joblib, or AutoGluon ZIP)"
    ),
    data_file: UploadFile = File(
        ...,
        description="CSV file with test data"
    ),
    user_level: UserLevel = Form(
        default=UserLevel.expert,
        description="Report detail level"
    ),
    target_col: Optional[str] = Form(
        default=None,
        description="Target column name (auto-detected if not provided)"
    ),
    max_shap_samples: int = Form(
        default=300,
        ge=10,
        le=5000,
        description="Max samples for SHAP/importance computation"
    )
):
    """
    Generate an explainability report for any supported model.

    **Supported Models:**
    - AutoGluon: TabularPredictor, MultiModalPredictor, TimeSeriesPredictor
    - sklearn: RandomForest, GradientBoosting, XGBoost, LightGBM, etc.

    **Report Modes:**
    - `beginner`: Simplified report with key insights
    - `expert`: Full report with SHAP values, metrics, and visualizations

    **Note:** For AutoGluon models, categorical data is handled automatically
    by the native EDA module. No manual encoding required.

    **Returns:** HTML report with visualizations
    """
    try:
        print(f"Received explain-model request:")
        print(f"  - Model file: {model_file.filename}")
        print(f"  - Data file: {data_file.filename}")
        print(f"  - User level: {user_level}")

        # Read uploaded files
        model_bytes = await model_file.read()
        data_bytes = await data_file.read()

        # Load model
        print("Loading model...")
        model_info = load_model_from_bytes(model_bytes, model_file.filename)
        print(f"  - Model type: {model_info.model_type}")
        print(f"  - Problem type: {model_info.problem_type}")
        print(f"  - Is AutoGluon: {model_info.is_autogluon}")

        # ── Vision model branch (pytorch_vision) ──────────────────────────────
        if model_info.model_type == 'pytorch_vision':
            if model_info.vision_info is None:
                raise HTTPException(
                    status_code=500,
                    detail="Vision model loaded but VisionModelInfo is missing."
                )
            vision_tmp = None
            try:
                print("Loading vision dataset from ZIP...")
                X, y, vision_tmp = _load_vision_dataset(
                    data_bytes, model_info.vision_info
                )
                from xai_core.report_builder import ReportBuilder
                explainer = ExplainerFactory.create(
                    model=model_info.vision_info,
                    X=X,
                    y=y,
                    model_type='pytorch_vision',
                )
                print(f"Generating {user_level.value} vision report...")
                html_report = ReportBuilder(explainer).build(mode=user_level.value)
                print("Vision report generated successfully!")
                return HTMLResponse(content=html_report)
            finally:
                if vision_tmp:
                    shutil.rmtree(vision_tmp, ignore_errors=True)
        # ── End vision branch ──────────────────────────────────────────────────

        # Load data
        print("Loading data...")
        df = pd.read_csv(io.BytesIO(data_bytes))
        print(f"  - Data shape: {df.shape}")
        
        # Detect target column
        target = target_col or detect_target_column(df)
        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target}' not found in data. Available columns: {list(df.columns)}"
            )
        
        print(f"  - Target column: {target}")
        
        # Prepare features and target
        X = df.drop(columns=[target])
        y = df[target]

        # ── Feature alignment + encoding (sklearn only — AutoGluon handles its own pipeline) ──
        model_obj = model_info.model
        if not model_info.is_autogluon:
            expected_features = None
            if hasattr(model_obj, 'feature_names_in_'):
                expected_features = list(model_obj.feature_names_in_)
            elif hasattr(model_obj, 'feature_name_'):      # LightGBM
                expected_features = list(model_obj.feature_name_())
            elif hasattr(model_obj, 'feature_names_'):     # XGBoost Booster
                expected_features = list(model_obj.feature_names_)

            if expected_features:
                # Build rename map: model feature → CSV column (fuzzy match for renamed cols)
                rename_map = {}
                csv_cols_lower = {c.lower().replace(' ', '_'): c for c in X.columns}
                for feat in expected_features:
                    if feat not in X.columns:
                        norm = feat.lower().replace(' ', '_')
                        if norm in csv_cols_lower:
                            rename_map[csv_cols_lower[norm]] = feat
                        else:
                            candidates = [c for c in X.columns
                                          if c.lower().startswith(norm) or norm.startswith(c.lower())]
                            if len(candidates) == 1:
                                rename_map[candidates[0]] = feat

                if rename_map:
                    print(f"  - Column renames: {rename_map}")
                    X = X.rename(columns=rename_map)

                available = [f for f in expected_features if f in X.columns]
                extra     = [f for f in X.columns if f not in expected_features]
                missing   = [f for f in expected_features if f not in X.columns]
                if extra or missing:
                    print(f"  - Feature alignment: dropping {extra}, still missing {missing}")
                X = X[available]

            # Encode string/object columns so sklearn predict() works
            obj_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == 'category']
            if obj_cols:
                print(f"  - Auto-encoding {len(obj_cols)} categorical columns: {obj_cols}")
                X = X.copy()
                for col in obj_cols:
                    X[col] = X[col].astype('category').cat.codes  # -1 for NaN, 0+ for categories

        # For AutoGluon models, narrow X to only the features the predictor was
        # trained on before explaining the model.
        if model_info.is_autogluon:
            X = _align_autogluon_features(model_info.model, X)
            print(f"  - Feature-aligned X shape: {X.shape}")

        # Use new ExplainerFactory for optimized model-specific explainability
        print("Creating explainer using ExplainerFactory...")
        print(f"  - Detected model type: {ExplainerFactory.detect_model_type(model_info.model)}")
        
        try:
            # Create explainer using the factory (auto-detects optimal explainer)
            explainer = ExplainerFactory.create(
                model=model_info.model,
                X=X,
                y=y,
                max_samples=max_shap_samples
            )
            print(f"  - Using explainer: {explainer.__class__.__name__}")

            # Generate model-only report. Data analysis is available separately
            # through the /analyze-data endpoint.
            print(f"Generating {user_level.value} report...")
            from xai_core.report_builder import ReportBuilder
            html_report = ReportBuilder(explainer).build(mode=user_level.value)

        except Exception as factory_error:
            # Fallback to legacy ExplainerService if new architecture fails
            print(f"ExplainerFactory failed: {factory_error}, falling back to legacy service")
            predictor = model_info.model if model_info.is_autogluon else None
            service = ExplainerService(
                model_info=model_info,
                X=X,
                y=y,
                predictor=predictor,
                max_samples=max_shap_samples,
                label=target
            )
            html_report = service.generate_html_report(mode=user_level.value)
        
        print("Report generated successfully!")
        return HTMLResponse(content=html_report)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in explain_model: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/explain-model/eda",
    response_class=HTMLResponse,
    tags=["Explainability"],
    summary="Generate dataset health and EDA report"
)
async def explain_model_eda(
    data_file: UploadFile = File(..., description="CSV file with data to analyze"),
    model_file: Optional[UploadFile] = File(None, description="Optional model file for target detection"),
    target_col: Optional[str] = Form(None, description="Target column name (overrides auto-detection)")
):
    """
    Generate a detailed Exploratory Data Analysis (EDA) report.
    
    **Features:**
    - Statistical summary of all features
    - Missing value analysis
    - Feature correlation heatmaps
    - Target distribution analysis (if target is known)
    
    **Returns:** HTML report with interactive-ready visualizations
    """
    try:
        # Load data
        data_bytes = await data_file.read()
        df = pd.read_csv(io.BytesIO(data_bytes))
        
        model_info = None
        target = target_col
        
        # Try to detect target from model if provided
        if model_file:
            model_bytes = await model_file.read()
            model_info = load_model_from_bytes(model_bytes, model_file.filename)
            
            # If target not explicitly provided, try to guess or use model info?
            # AutoGluon models often store label info, but here we just rely on the data
            pass
            
        if not target:
            target = detect_target_column(df)
            
        X = df.drop(columns=[target]) if target and target in df.columns else df
        y = df[target] if target and target in df.columns else None
        
        # Create service (dummy model info if none provided)
        if not model_info:
            # Create a dummy model info for the service init
            from xai_core.model_loader import ModelInfo
            model_info = ModelInfo(
                model=None, model_type="unknown", 
                problem_type="unknown", is_autogluon=False
            )
            
        service = ExplainerService(model_info=model_info, X=X, y=y)
        return HTMLResponse(content=service.generate_eda_report())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/explain-model/info",
    response_model=ModelInfoResponse,
    tags=["Explainability"],
    summary="Get model information without generating report"
)
async def get_model_info(
    model_file: UploadFile = File(..., description="Model file"),
    data_file: UploadFile = File(..., description="CSV file with data")
):
    """
    Get model metadata and basic metrics without generating full report.
    
    Useful for quick model inspection before generating detailed report.
    """
    try:
        model_bytes = await model_file.read()
        data_bytes = await data_file.read()
        
        model_info = load_model_from_bytes(model_bytes, model_file.filename)
        df = pd.read_csv(io.BytesIO(data_bytes))
        
        target = detect_target_column(df)
        X = df.drop(columns=[target]) if target in df.columns else df
        y = df[target] if target in df.columns else None
        
        service = ExplainerService(model_info=model_info, X=X, y=y)
        metrics = service.get_metrics()
        
        return ModelInfoResponse(
            model_type=model_info.model_type,
            problem_type=model_info.problem_type,
            is_autogluon=model_info.is_autogluon,
            n_features=len(X.columns),
            n_samples=len(X),
            ensemble_models=metrics.get('ensemble_models'),
            best_model=metrics.get('best_model'),
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post(
    "/analyze-data",
    response_class=HTMLResponse,
    tags=["Data Analysis"],
    summary="Generate comprehensive data analysis report",
    responses={
        200: {"description": "HTML data analysis report"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def analyze_data(
    csv_file: UploadFile = File(
        ..., 
        description="CSV file to analyze"
    ),
    user_level: UserLevel = Form(
        default=UserLevel.expert,
        description="Report detail level ('beginner' or 'expert')"
    )
):
    """
    Generate a comprehensive data analysis report for any CSV dataset.
    
    **Features:**
    - Dataset overview (shape, memory usage)
    - Column type detection (numeric, categorical, datetime, boolean)
    - Missing value analysis
    - Outlier detection (Z-score method)
    - Correlation analysis with heatmap
    - Distribution plots for numeric columns
    - Frequency analysis for categorical columns
    
    **Report Modes:**
    - `beginner`: Simplified report with plain language explanations
    - `expert`: Full statistical details with all visualizations
    
    **Returns:** HTML report with embedded visualizations
    """
    try:
        print(f"Received analyze-data request:")
        print(f"  - File: {csv_file.filename}")
        print(f"  - User level: {user_level}")
        
        # Read uploaded file
        data_bytes = await csv_file.read()
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(io.BytesIO(data_bytes))
        print(f"  - Data shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Create service and generate report
        print("Creating data interpretability service...")
        service = DataInterpretabilityService(df)
        
        print(f"Generating {user_level.value} report...")
        html_report = service.generate_report(user_level=user_level.value)
        
        print("Report generated successfully!")
        return HTMLResponse(content=html_report)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/analyze-data/json",
    tags=["Data Analysis"],
    summary="Get data analysis results as JSON",
    responses={
        200: {"description": "JSON data analysis results"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def analyze_data_json(
    csv_file: UploadFile = File(
        ..., 
        description="CSV file to analyze"
    )
):
    """
    Get data analysis results as JSON instead of HTML report.
    
    Useful for programmatic access to analysis results.
    
    **Returns:** JSON with data info, column analysis, and statistics
    """
    try:
        # Read uploaded file
        data_bytes = await csv_file.read()
        df = pd.read_csv(io.BytesIO(data_bytes))
        
        # Create service
        service = DataInterpretabilityService(df)
        
        # Get results
        data_info = service.get_data_info()
        column_info = service.get_column_info()
        analysis = service.get_analysis_results()
        
        # Convert analysis results to JSON-serializable format
        result = {
            "data_info": data_info,
            "column_info": column_info,
            "outliers": analysis.get('outliers', {}),
            "numeric_columns": [col for col, info in column_info.items() if info['type'] == 'numeric'],
            "categorical_columns": [col for col, info in column_info.items() if info['type'] == 'categorical'],
        }
        
        # Add correlation matrix if available
        if 'correlation_matrix' in analysis:
            result['correlation_matrix'] = analysis['correlation_matrix'].to_dict()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Include UC2 Router
# ============================================================================

from api.routers.uc2 import router as uc2_router
app.include_router(uc2_router)


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint - redirects to API documentation."""
    return {
        "message": "XAI Explainability API",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc"
    }
