from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ReportType(str, Enum):
    """Type of XAI report"""
    MODEL_EXPLANATION = "model_explanation"
    DATA_EXPLANATION = "data_explanation"


class ExpertiseLevel(str, Enum):
    """Expertise level of the report"""
    BEGINNER = "beginner"
    EXPERT = "expert"


class XAIReportMetadata(BaseModel):
    """Metadata for XAI report stored in MongoDB"""
    user_id: str = Field(..., description="User who owns the report")
    dataset_id: str = Field(..., description="Dataset ID associated with the report")
    model_id: str = Field(..., description="AI Model ID associated with the report")
    report_type: ReportType = Field(..., description="Type of explanation report")
    level: ExpertiseLevel = Field(..., description="Expertise level of the report")
    file_path: str = Field(..., description="Path to HTML file in MinIO")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="MD5 hash of the file")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class XAIReportCreate(BaseModel):
    """Schema for creating an XAI report"""
    user_id: str
    dataset_id: str
    model_id: str
    report_type: ReportType
    level: ExpertiseLevel
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class XAIReportResponse(BaseModel):
    """Response schema for XAI report"""
    user_id: str
    dataset_id: str
    model_id: str
    report_type: ReportType
    level: ExpertiseLevel
    file_path: str
    file_size: int
    file_hash: Optional[str]
    custom_metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True

