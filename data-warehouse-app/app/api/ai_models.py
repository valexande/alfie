from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
import io
import logging

from ..models.ai_model import (
    AIModelMetadata, ModelCreate, ModelUpdate, ModelResponse, 
    ModelFile, ModelFramework, ModelType
)
from ..services.ai_model_service import ai_model_service
from ..services.kafka_service import kafka_producer_service
from ..core.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-models", tags=["AI Models"])


async def _get_dataset_folder_info(user_id: str, dataset_id: str) -> tuple:
    """
    Get is_folder and file_count from dataset metadata
    
    Returns:
        Tuple of (is_folder, file_count)
    """
    if not dataset_id:
        return False, 1
    
    try:
        db = get_database()
        dataset = await db.datasets.find_one({
            "user_id": user_id,
            "dataset_id": dataset_id
        }, sort=[("created_at", -1)])  # Get latest version
        
        if dataset:
            is_folder = dataset.get("is_folder", False)
            file_count = dataset.get("custom_metadata", {}).get("file_count", 1) if is_folder else 1
            return is_folder, file_count
        
        return False, 1
    except Exception as e:
        logger.warning(f"Could not fetch dataset folder info: {e}")
        return False, 1


async def _get_next_model_version(user_id: str, model_id: str) -> str:
    """
    Auto-detect the next version number for a model.
    Queries existing versions and increments (v1, v2, v3, etc.)
    """
    try:
        db = get_database()
        
        # Find all versions of this model for this user
        cursor = db.ai_models.find({
            "user_id": user_id,
            "model_id": model_id
        }, {"version": 1})
        
        versions = await cursor.to_list(length=1000)
        
        if not versions:
            return "v1"
        
        # Extract version numbers (v1 -> 1, v2 -> 2, etc.)
        version_numbers = []
        for doc in versions:
            version_str = doc.get("version", "v1")
            if version_str.startswith("v"):
                try:
                    version_numbers.append(int(version_str[1:]))
                except ValueError:
                    pass
        
        # Get the max version and increment
        if version_numbers:
            next_version = max(version_numbers) + 1
        else:
            next_version = 1
        
        return f"v{next_version}"
        
    except Exception as e:
        logger.warning(f"Error detecting next model version, defaulting to v1: {e}")
        return "v1"


@router.post("/upload/single/{user_id}", response_model=ModelResponse)
async def upload_single_model_file(
    user_id: str,
    file: UploadFile = File(...),
    model_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    framework: ModelFramework = Form(...),
    model_type: ModelType = Form(...),
    algorithm: Optional[str] = Form(None),
    is_primary: bool = Form(False),
    tags: Optional[str] = Form(None),
    training_dataset: Optional[str] = Form(None),
    training_accuracy: Optional[float] = Form(None),
    validation_accuracy: Optional[float] = Form(None),
    test_accuracy: Optional[float] = Form(None),
    training_loss: Optional[float] = Form(None),
    python_version: Optional[str] = Form(None),
    hardware_requirements: Optional[str] = Form(None)
):
    """
    Upload a single AI model file
    
    Version is automatically incremented (v1, v2, v3, etc.)
    User ID is now in the path parameter for better REST API structure
    """
    try:
        db = get_database()
        collection = db.ai_models
        
        # Auto-detect next version
        version = await _get_next_model_version(user_id, model_id)
        logger.info(f"Auto-detected version: {version} for model {model_id}")
        
        # Upload file to MinIO
        file_path, model_file = await ai_model_service.upload_model_file(
            file=file,
            user_id=user_id,
            model_id=model_id,
            version=version,
            is_primary=is_primary
        )
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Detect framework if not provided
        if not framework:
            framework = ai_model_service._detect_model_framework(file.filename, file.content_type)
            if not framework:
                framework = ModelFramework.OTHER
        
        # Calculate model size
        model_size_mb = await ai_model_service.calculate_model_size([model_file])
        
        # Create model metadata
        model_metadata = AIModelMetadata(
            model_id=model_id,
            user_id=user_id,
            name=name,
            description=description,
            version=version,
            framework=framework,
            model_type=model_type,
            algorithm=algorithm,
            files=[model_file],
            primary_file_path=file_path if is_primary else None,
            is_model_folder=False,
            model_file_count=1,
            model_size_mb=model_size_mb,
            training_dataset=training_dataset,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            test_accuracy=test_accuracy,
            training_loss=training_loss,
            python_version=python_version,
            hardware_requirements=hardware_requirements,
            tags=tag_list
        )
        
        # Save to MongoDB
        result = await collection.insert_one(model_metadata.dict(by_alias=True))
        
        logger.info(f"AI model uploaded successfully: {model_id} by {user_id}")
        
        # Return the created model
        created_model = await collection.find_one({"_id": result.inserted_id})
        
        # Send Kafka AutoML event (non-blocking)
        try:
            # Get dataset folder info if training_dataset is provided
            is_folder, file_count = await _get_dataset_folder_info(user_id, model_metadata.training_dataset)
            
            await kafka_producer_service.send_automl_event(
                user_id=user_id,
                model_id=model_id,
                dataset_id=model_metadata.training_dataset,
                version=version,
                framework=framework.value if framework else None,
                model_type=model_type.value if model_type else None,
                algorithm=algorithm,
                model_size_mb=model_size_mb,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                test_accuracy=test_accuracy,
                is_folder=is_folder,
                file_count=file_count,
                is_model_folder=model_metadata.is_model_folder,
                model_file_count=model_metadata.model_file_count
            )
        except Exception as e:
            logger.warning(f"Failed to send AutoML Kafka event: {e}")
        
        return ModelResponse(**created_model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload AI model")


@router.post("/upload/folder/{user_id}", response_model=ModelResponse)
async def upload_model_folder(
    user_id: str,
    zip_file: UploadFile = File(...),
    model_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    framework: ModelFramework = Form(...),
    model_type: ModelType = Form(...),
    algorithm: Optional[str] = Form(None),
    preserve_structure: bool = Form(True),
    tags: Optional[str] = Form(None),
    training_dataset: Optional[str] = Form(None),
    training_accuracy: Optional[float] = Form(None),
    validation_accuracy: Optional[float] = Form(None),
    test_accuracy: Optional[float] = Form(None),
    training_loss: Optional[float] = Form(None),
    python_version: Optional[str] = Form(None),
    hardware_requirements: Optional[str] = Form(None)
):
    """
    Upload a folder of AI model files (as zip)
    
    Version is automatically incremented (v1, v2, v3, etc.)
    User ID is now in the path parameter for better REST API structure
    """
    try:
        db = get_database()
        collection = db.ai_models
        
        # Auto-detect next version
        version = await _get_next_model_version(user_id, model_id)
        logger.info(f"Auto-detected version: {version} for model {model_id}")
        
        # Upload folder to MinIO
        model_files = await ai_model_service.upload_model_folder(
            zip_file=zip_file,
            user_id=user_id,
            model_id=model_id,
            version=version,
            preserve_structure=preserve_structure
        )
        
        if not model_files:
            raise HTTPException(status_code=400, detail="No valid model files found in the uploaded folder")
        
        # Detect primary file
        primary_file = await ai_model_service.detect_primary_file(model_files)
        if primary_file:
            primary_file.is_primary = True
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Calculate model size
        model_size_mb = await ai_model_service.calculate_model_size(model_files)
        
        # Create model metadata
        model_metadata = AIModelMetadata(
            model_id=model_id,
            user_id=user_id,
            name=name,
            description=description,
            version=version,
            framework=framework,
            model_type=model_type,
            algorithm=algorithm,
            files=model_files,
            primary_file_path=primary_file.file_path if primary_file else None,
            is_model_folder=True,
            model_file_count=len(model_files),
            model_size_mb=model_size_mb,
            training_dataset=training_dataset,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            test_accuracy=test_accuracy,
            training_loss=training_loss,
            python_version=python_version,
            hardware_requirements=hardware_requirements,
            tags=tag_list
        )
        
        # Save to MongoDB
        result = await collection.insert_one(model_metadata.dict(by_alias=True))
        
        logger.info(f"AI model folder uploaded successfully: {model_id} by {user_id}")
        
        # Return the created model
        created_model = await collection.find_one({"_id": result.inserted_id})
        
        # Send Kafka AutoML event (non-blocking)
        try:
            # Get dataset folder info if training_dataset is provided
            is_folder, file_count = await _get_dataset_folder_info(user_id, model_metadata.training_dataset)
            
            await kafka_producer_service.send_automl_event(
                user_id=user_id,
                model_id=model_id,
                dataset_id=model_metadata.training_dataset,
                version=version,
                framework=framework.value if framework else None,
                model_type=model_type.value if model_type else None,
                algorithm=algorithm,
                model_size_mb=model_size_mb,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                test_accuracy=test_accuracy,
                is_folder=is_folder,
                file_count=file_count,
                is_model_folder=model_metadata.is_model_folder,
                model_file_count=model_metadata.model_file_count
            )
        except Exception as e:
            logger.warning(f"Failed to send AutoML Kafka event: {e}")
        
        return ModelResponse(**created_model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading AI model folder: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload AI model folder")


@router.get("/search/{user_id}", response_model=List[ModelResponse])
async def search_models(
    user_id: str,
    query: Optional[str] = Query(None),
    framework: Optional[ModelFramework] = Query(None),
    model_type: Optional[ModelType] = Query(None),
    tags: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Search AI models with various filters"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Build search query
        search_query = {"user_id": user_id}
        
        if query:
            # Create text search conditions
            text_conditions = [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
                {"algorithm": {"$regex": query, "$options": "i"}},
                {"model_id": {"$regex": query, "$options": "i"}}
            ]
            
            # If we already have other conditions, we need to combine them properly
            if any([framework, model_type, tags]):
                search_query["$and"] = [{"$or": text_conditions}]
            else:
                search_query["$or"] = text_conditions
        
        if framework:
            if "$and" in search_query:
                search_query["$and"].append({"framework": framework})
            else:
                search_query["framework"] = framework
                
        if model_type:
            if "$and" in search_query:
                search_query["$and"].append({"model_type": model_type})
            else:
                search_query["model_type"] = model_type
                
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            tag_condition = {"tags": {"$in": tag_list}}
            if "$and" in search_query:
                search_query["$and"].append(tag_condition)
            else:
                search_query.update(tag_condition)
        
        logger.info(f"Search query: {search_query}")
        
        # Execute search
        cursor = collection.find(search_query).skip(skip).limit(limit).sort("created_at", -1)
        models = await cursor.to_list(length=limit)
        
        logger.info(f"Found {len(models)} models matching search criteria")
        
        return [ModelResponse(**model) for model in models]
        
    except Exception as e:
        logger.error(f"Error searching AI models: {e}")
        raise HTTPException(status_code=500, detail="Failed to search AI models")


@router.get("/{user_id}/{model_id}", response_model=ModelResponse)
async def get_model(user_id: str, model_id: str, version: str = Query("v1")):
    """Get AI model metadata"""
    try:
        db = get_database()
        collection = db.ai_models
        
        model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelResponse(**model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI model")


@router.get("/{user_id}", response_model=List[ModelResponse])
async def get_user_models(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    framework: Optional[ModelFramework] = Query(None),
    model_type: Optional[ModelType] = Query(None),
    tags: Optional[str] = Query(None)
):
    """Get all AI models for a user with optional filtering"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Build filter query
        filter_query = {"user_id": user_id}
        
        if framework:
            filter_query["framework"] = framework
        if model_type:
            filter_query["model_type"] = model_type
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            filter_query["tags"] = {"$in": tag_list}
        
        # Get models
        cursor = collection.find(filter_query).skip(skip).limit(limit).sort("created_at", -1)
        models = await cursor.to_list(length=limit)
        
        return [ModelResponse(**model) for model in models]
        
    except Exception as e:
        logger.error(f"Error retrieving user AI models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user AI models")


@router.get("/{user_id}/{model_id}/download")
async def download_model_file(
    user_id: str, 
    model_id: str, 
    version: str = Query("v1"),
    filename: Optional[str] = Query(None)
):
    """Download a specific model file or all files as zip"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Get model metadata
        model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if filename:
            # Find the file in the model's files list to get the correct path
            model_files = model.get("files", [])
            target_file = None
            
            for file_info in model_files:
                if file_info.get("filename") == filename:
                    target_file = file_info
                    break
            
            if not target_file:
                raise HTTPException(status_code=404, detail=f"File '{filename}' not found in model")
            
            # Use the stored file path from metadata
            file_path = target_file.get("file_path")
            if not file_path:
                raise HTTPException(status_code=404, detail="File path not found in metadata")
            
            file_data = await ai_model_service.download_model_file(file_path)
            
            return StreamingResponse(
                io.BytesIO(file_data),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Download all files as zip
            zip_data = await ai_model_service.download_model_as_zip(user_id, model_id, version)
            
            return StreamingResponse(
                io.BytesIO(zip_data),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={model_id}_{version}.zip"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to download AI model")


@router.put("/{user_id}/{model_id}", response_model=ModelResponse)
async def update_model(
    user_id: str, 
    model_id: str, 
    version: str = Query("v1"),
    model_update: ModelUpdate = None
):
    """Update AI model metadata"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Check if model exists
        existing_model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if not existing_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prepare update data
        update_data = model_update.dict(exclude_unset=True)
        
        # Update model
        result = await collection.update_one(
            {"user_id": user_id, "model_id": model_id, "version": version},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No changes made to the model")
        
        # Return updated model
        updated_model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        logger.info(f"AI model updated: {model_id} by {user_id}")
        return ModelResponse(**updated_model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update AI model")


@router.delete("/{user_id}/{model_id}")
async def delete_model(user_id: str, model_id: str, version: str = Query("v1")):
    """Delete an AI model and all its files"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Check if model exists
        existing_model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if not existing_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Delete files from MinIO
        await ai_model_service.delete_model_files(user_id, model_id, version)
        
        # Delete from MongoDB
        result = await collection.delete_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=400, detail="Failed to delete model")
        
        logger.info(f"AI model deleted: {model_id} by {user_id}")
        return {"message": "Model deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete AI model")


@router.get("/{user_id}/{model_id}/files", response_model=List[ModelFile])
async def list_model_files(user_id: str, model_id: str, version: str = Query("v1")):
    """List all files for a specific model"""
    try:
        db = get_database()
        collection = db.ai_models
        
        # Get model metadata
        model = await collection.find_one({
            "user_id": user_id,
            "model_id": model_id,
            "version": version
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        files = model.get("files", [])
        
        # Add a helpful summary
        logger.info(f"Listed {len(files)} files for model {model_id} version {version}")
        
        return files
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing model files: {e}")
        raise HTTPException(status_code=500, detail="Failed to list model files")
