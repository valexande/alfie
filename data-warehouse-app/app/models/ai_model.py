from datetime import datetime
from typing import Optional, Dict, Any, List, Annotated
from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer
from bson import ObjectId
from enum import Enum


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


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    ONNX = "onnx"
    KERAS = "keras"
    HUGGINGFACE = "huggingface"
    OTHER = "other"


class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    OTHER = "other"


class ModelFile(BaseModel):
    """Represents a single file within a model"""
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path in MinIO storage")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    file_hash: str = Field(..., description="MD5 hash of the file")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    is_primary: bool = Field(default=False, description="Whether this is the main model file")
    description: Optional[str] = Field(None, description="Description of this file's purpose")


class AIModelMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    model_id: str = Field(..., description="Unique model identifier")
    user_id: str = Field(..., description="User who owns this model")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    version: str = Field(default="v1", description="Model version")
    
    # Model characteristics
    framework: ModelFramework = Field(..., description="ML framework used")
    model_type: ModelType = Field(..., description="Type of ML model")
    algorithm: Optional[str] = Field(None, description="Specific algorithm used")
    
    # Model files
    files: List[ModelFile] = Field(default_factory=list, description="List of model files")
    primary_file_path: Optional[str] = Field(None, description="Path to the primary model file")
    is_model_folder: bool = Field(default=False, description="Whether this is a folder upload (multiple files)")
    model_file_count: int = Field(default=1, description="Number of model files")
    
    # Model metadata
    input_shape: Optional[List[int]] = Field(None, description="Expected input shape")
    output_shape: Optional[List[int]] = Field(None, description="Expected output shape")
    num_parameters: Optional[int] = Field(None, description="Number of model parameters")
    model_size_mb: Optional[float] = Field(None, description="Total model size in MB")
    
    # Training metadata
    training_dataset: Optional[str] = Field(None, description="Training dataset identifier")
    training_accuracy: Optional[float] = Field(None, description="Training accuracy")
    validation_accuracy: Optional[float] = Field(None, description="Validation accuracy")
    test_accuracy: Optional[float] = Field(None, description="Test accuracy")
    training_loss: Optional[float] = Field(None, description="Final training loss")
    
    # Model requirements
    python_version: Optional[str] = Field(None, description="Required Python version")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    hardware_requirements: Optional[str] = Field(None, description="Hardware requirements")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Model status
    is_active: bool = Field(default=True, description="Whether the model is active")
    is_production_ready: bool = Field(default=False, description="Whether the model is production ready")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ModelCreate(BaseModel):
    model_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    version: str = "v1"
    framework: ModelFramework
    model_type: ModelType
    algorithm: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    training_dataset: Optional[str] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    python_version: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    hardware_requirements: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    algorithm: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    training_dataset: Optional[str] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    python_version: Optional[str] = None
    dependencies: Optional[List[str]] = None
    hardware_requirements: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_production_ready: Optional[bool] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelResponse(BaseModel):
    model_id: str
    user_id: str
    name: str
    description: Optional[str]
    version: str
    framework: ModelFramework
    model_type: ModelType
    algorithm: Optional[str]
    files: List[ModelFile]
    primary_file_path: Optional[str]
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]
    num_parameters: Optional[int]
    model_size_mb: Optional[float]
    training_dataset: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    test_accuracy: Optional[float]
    training_loss: Optional[float]
    python_version: Optional[str]
    dependencies: List[str]
    hardware_requirements: Optional[str]
    tags: List[str]
    custom_metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    is_production_ready: bool


class ModelFileUpload(BaseModel):
    """Request model for uploading a single model file"""
    user_id: str
    model_id: str
    version: str = "v1"
    is_primary: bool = False
    description: Optional[str] = None


class ModelFolderUpload(BaseModel):
    """Request model for uploading a model folder"""
    user_id: str
    model_id: str
    version: str = "v1"
    preserve_structure: bool = True
