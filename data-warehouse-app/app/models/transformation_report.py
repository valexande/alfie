from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class TransformationReport(BaseModel):
    user_id: str = Field(...)
    dataset_id: str = Field(...)
    version: str = Field(..., description="Target version (e.g., v2) for mitigated dataset")
    report: Any = Field(..., description="Arbitrary transformation report structure")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TransformationReportCreate(BaseModel):
    user_id: str
    dataset_id: str
    version: str
    report: Any


class TransformationReportResponse(BaseModel):
    user_id: str
    dataset_id: str
    version: str
    report: Any
    created_at: datetime
    updated_at: datetime

