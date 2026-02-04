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
            
            # Generate report using new architecture
            print(f"Generating {user_level.value} report...")
            html_report = explainer.generate_report(mode=user_level.value)
            
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
    - K-Means clustering visualization
    - Scatter matrix for pairwise relationships
    
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
