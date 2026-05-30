"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Dict, List, Any
from enum import Enum


class UserLevel(str, Enum):
    """User expertise level for report customization."""
    beginner = "beginner"
    expert = "expert"


# ============================================================================
# Request Schemas
# ============================================================================

class ExplainModelForm(BaseModel):
    """Form data for /explain-model endpoint."""
    user_level: UserLevel = Field(
        default=UserLevel.expert,
        description="User expertise level ('beginner' or 'expert')"
    )
    target_col: Optional[str] = Field(
        default=None,
        description="Target column name (auto-detected if not provided)"
    )
    max_shap_samples: int = Field(
        default=300,
        ge=10,
        le=5000,
        description="Maximum samples for SHAP computation"
    )
    
    @field_validator('target_col')
    @classmethod
    def validate_target_col(cls, v):
        if v and len(v) > 100:
            raise ValueError('target_col name too long (max 100 chars)')
        return v


class ExplainUC2DataForm(BaseModel):
    """Form data for /explain-uc2-data endpoint."""
    user_level: UserLevel = Field(
        default=UserLevel.expert,
        description="User expertise level"
    )


class ExplainUC2ModelForm(BaseModel):
    """Form data for /explain-uc2-model endpoint."""
    user_level: UserLevel = Field(
        default=UserLevel.expert,
        description="User expertise level"
    )


# ============================================================================
# Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: Literal["healthy", "unhealthy"]
    service: str
    version: str = "2.0.0"
    features: Dict[str, bool]
    
    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "service": "xai-explainability",
            "version": "2.0.0",
            "features": {
                "explainerdashboard": True,
                "autogluon_tabular": True,
                "autogluon_multimodal": True,
                "autogluon_timeseries": True
            }
        }
    }}


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(description="Error type/name")
    detail: str = Field(description="Error details/message")
    suggestions: List[str] = Field(
        default=[],
        description="Suggestions for resolving the error"
    )
    
    model_config = {"json_schema_extra": {
        "example": {
            "error": "ModelLoadError",
            "detail": "Could not load model from provided file",
            "suggestions": [
                "Ensure the model file is a valid pickle or AutoGluon predictor",
                "Check that all required dependencies are installed"
            ]
        }
    }}


class ModelInfoResponse(BaseModel):
    """Response with model metadata."""
    model_type: str = Field(description="Type of model (tabular, multimodal, etc.)")
    problem_type: str = Field(description="Problem type (classification, regression, forecasting)")
    is_autogluon: bool = Field(description="Whether model is an AutoGluon predictor")
    n_features: int = Field(description="Number of features")
    n_samples: int = Field(description="Number of samples in data")
    ensemble_models: Optional[List[str]] = Field(
        default=None,
        description="Models in ensemble (AutoGluon only)"
    )
    best_model: Optional[str] = Field(
        default=None,
        description="Best performing model (AutoGluon only)"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model performance metrics"
    )


class FeatureImportanceResponse(BaseModel):
    """Response with feature importance data."""
    method: str = Field(description="Method used (native, permutation, shap)")
    features: List[Dict[str, Any]] = Field(
        description="List of features with importance scores"
    )
    
    model_config = {"json_schema_extra": {
        "example": {
            "method": "permutation",
            "features": [
                {"feature": "age", "importance": 0.25},
                {"feature": "income", "importance": 0.18},
                {"feature": "education", "importance": 0.12}
            ]
        }
    }}


# ============================================================================
# UC2 Specific Schemas
# ============================================================================

class DriverAnalysisResponse(BaseModel):
    """Response for UC2 driver analysis."""
    n_samples: int
    n_anomalies: int
    correlation_matrix: Dict[str, Dict[str, float]]
    alerts_by_category: Optional[Dict[str, Any]] = None


class FairnessMetrics(BaseModel):
    """Fairness metrics per demographic group."""
    group: str
    accuracy: float
    precision: float
    recall: float
    f1: Optional[float] = None
