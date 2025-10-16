from datetime import datetime
from typing import Optional, Dict, Any, List, Annotated
from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer
from bson import ObjectId


def validate_object_id(v):
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return ObjectId(v)
    raise ValueError("Invalid ObjectId")


PyObjectId = Annotated[
    ObjectId,
    BeforeValidator(validate_object_id),
    PlainSerializer(lambda x: str(x), return_type=str)
]


class DatasetFile(BaseModel):
    """Represents a single file within a dataset folder"""
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path in MinIO storage")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    file_hash: str = Field(..., description="MD5 hash of the file")
    content_type: Optional[str] = Field(None, description="MIME type of the file")


class DatasetMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    dataset_id: str = Field(..., description="Unique dataset identifier")
    user_id: str = Field(..., description="User who owns this dataset")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    version: str = Field(default="v1", description="Dataset version")
    
    # Single file metadata (for backward compatibility)
    file_type: str = Field(..., description="Type of file (csv, xlsx, json, folder, etc.)")
    file_size: int = Field(..., description="Total file size in bytes")
    file_path: str = Field(..., description="Path in MinIO storage (single file or folder)")
    original_filename: str = Field(..., description="Original filename when uploaded")
    
    # Multiple files support (for folder uploads)
    files: Optional[List[DatasetFile]] = Field(None, description="List of files (for folder uploads)")
    is_folder: bool = Field(default=False, description="Whether this is a folder upload")
    
    # Metadata extracted from file content (single file only)
    columns: Optional[List[str]] = Field(None, description="Column names for structured data (single file)")
    row_count: Optional[int] = Field(None, description="Number of rows for structured data (single file)")
    data_types: Optional[Dict[str, str]] = Field(None, description="Data types for each column (single file)")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # File hash for integrity checking
    file_hash: Optional[str] = Field(None, description="MD5 hash of the file")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class DatasetCreate(BaseModel):
    dataset_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    version: str = "v1"
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetResponse(BaseModel):
    dataset_id: str
    user_id: str
    name: str
    description: Optional[str]
    version: str
    file_type: str
    file_size: int
    original_filename: str
    files: Optional[List[DatasetFile]] = None
    is_folder: bool = False
    columns: Optional[List[str]]
    row_count: Optional[int]
    data_types: Optional[Dict[str, str]]
    tags: List[str]
    custom_metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    file_hash: Optional[str]
