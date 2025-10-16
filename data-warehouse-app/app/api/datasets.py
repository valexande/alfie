from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from ..models.dataset import DatasetCreate, DatasetUpdate, DatasetResponse, DatasetMetadata, DatasetFile
from ..services.file_service import file_service
from ..services.metadata_service import metadata_service
from ..services.kafka_service import kafka_producer_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def _get_next_dataset_version(user_id: str, dataset_id: str) -> str:
    """
    Auto-detect the next version number for a dataset.
    Queries existing versions and increments (v1, v2, v3, etc.)
    """
    try:
        from ..core.database import get_database
        db = get_database()
        
        # Find all versions of this dataset for this user
        cursor = db.datasets.find({
            "user_id": user_id,
            "dataset_id": dataset_id
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
        logger.warning(f"Error detecting next version, defaulting to v1: {e}")
        return "v1"


@router.post("/upload/{user_id}", response_model=DatasetResponse)
async def upload_dataset(
    user_id: str,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)  # Comma-separated tags
):
    """
    Upload a dataset file and create metadata
    
    This endpoint handles the main flow:
    1. Auto-detect next version number for this dataset
    2. Upload file to MinIO with organized path structure
    3. Extract metadata from file content
    4. Store metadata in MongoDB
    
    Version is automatically incremented (v1, v2, v3, etc.)
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Parse tags if provided
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Auto-detect next version
        version = await _get_next_dataset_version(user_id, dataset_id)
        logger.info(f"Auto-detected version: {version} for dataset {dataset_id}")
        
        # Create dataset metadata object
        dataset_create = DatasetCreate(
            dataset_id=dataset_id,
            user_id=user_id,
            name=name,
            description=description,
            version=version,
            tags=tag_list
        )
        
        # Upload file to MinIO and get metadata
        file_path, file_metadata = await file_service.upload_file(
            file=file,
            user_id=user_id,
            dataset_id=dataset_id,
            version=version
        )
        
        # Add file path to metadata
        file_metadata['file_path'] = file_path
        
        # Store metadata in MongoDB
        dataset_metadata = await metadata_service.create_dataset_metadata(
            dataset_data=dataset_create,
            file_metadata=file_metadata
        )
        
        # Send Kafka event (non-blocking failure)
        try:
            await kafka_producer_service.send_dataset_uploaded_event(dataset_metadata)
        except Exception as kafka_error:
            logger.warning(f"Failed to send Kafka event for upload: {kafka_error}")

        # Convert to response model
        return DatasetResponse(
            dataset_id=dataset_metadata.dataset_id,
            user_id=dataset_metadata.user_id,
            name=dataset_metadata.name,
            description=dataset_metadata.description,
            version=dataset_metadata.version,
            file_type=dataset_metadata.file_type,
            file_size=dataset_metadata.file_size,
            original_filename=dataset_metadata.original_filename,
            columns=dataset_metadata.columns,
            row_count=dataset_metadata.row_count,
            data_types=dataset_metadata.data_types,
            tags=dataset_metadata.tags,
            custom_metadata=dataset_metadata.custom_metadata,
            created_at=dataset_metadata.created_at,
            updated_at=dataset_metadata.updated_at,
            file_hash=dataset_metadata.file_hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.post("/upload/folder/{user_id}", response_model=DatasetResponse)
async def upload_dataset_folder(
    user_id: str,
    zip_file: UploadFile = File(...),
    dataset_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    preserve_structure: bool = Form(True),
    tags: Optional[str] = Form(None)
):
    """
    Upload a folder of dataset files (as zip)
    
    For folder uploads:
    - Auto-detects next version
    - Extracts and uploads all files
    - Simplified metadata (no column/row analysis)
    - Lists all files with sizes and types
    """
    try:
        if not zip_file.filename or not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Auto-detect next version
        version = await _get_next_dataset_version(user_id, dataset_id)
        logger.info(f"Auto-detected version: {version} for dataset folder {dataset_id}")
        
        # Upload folder to MinIO
        dataset_files, total_size = await file_service.upload_dataset_folder(
            zip_file=zip_file,
            user_id=user_id,
            dataset_id=dataset_id,
            version=version,
            preserve_structure=preserve_structure
        )
        
        if not dataset_files:
            raise HTTPException(status_code=400, detail="No valid files found in ZIP")
        
        # Simplified metadata for folder
        file_types = list(set([f.file_type for f in dataset_files]))
        folder_path = f"datasets/{user_id}/{dataset_id}/{version}/"
        
        dataset_metadata = DatasetMetadata(
            dataset_id=dataset_id,
            user_id=user_id,
            name=name,
            description=description,
            version=version,
            file_type=", ".join(file_types),
            file_size=total_size,
            file_path=folder_path,
            original_filename=zip_file.filename,
            files=dataset_files,
            is_folder=True,
            columns=None,
            row_count=None,
            data_types=None,
            tags=tag_list,
            custom_metadata={"file_count": len(dataset_files), "preserve_structure": preserve_structure},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            file_hash=None
        )
        
        from ..core.database import get_database
        db = get_database()
        result = await db.datasets.insert_one(dataset_metadata.dict(by_alias=True, exclude={'id'}))
        created_dataset = await db.datasets.find_one({"_id": result.inserted_id})
        
        logger.info(f"Dataset folder uploaded: {dataset_id} ({len(dataset_files)} files)")
        
        try:
            await kafka_producer_service.send_dataset_uploaded_event(DatasetMetadata(**created_dataset))
        except Exception as kafka_error:
            logger.warning(f"Failed to send Kafka event: {kafka_error}")
        
        return DatasetResponse(**created_dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset folder: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset folder")


@router.get("/search/{user_id}", response_model=List[DatasetResponse])
async def search_datasets(
    user_id: str,
    query: Optional[str] = Query(None, description="Search in name, description, and dataset_id"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    file_type: Optional[str] = Query(None, description="File type to filter by"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Search datasets with various filters"""
    # Parse tags if provided
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    datasets = await metadata_service.search_datasets(
        user_id=user_id,
        query=query,
        tags=tag_list,
        file_type=file_type,
        skip=skip,
        limit=limit
    )
    
    return [
        DatasetResponse(
            dataset_id=dataset.dataset_id,
            user_id=dataset.user_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            file_type=dataset.file_type,
            file_size=dataset.file_size,
            original_filename=dataset.original_filename,
            columns=dataset.columns,
            row_count=dataset.row_count,
            data_types=dataset.data_types,
            tags=dataset.tags,
            custom_metadata=dataset.custom_metadata,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            file_hash=dataset.file_hash
        )
        for dataset in datasets
    ]


@router.get("/{user_id}/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(user_id: str, dataset_id: str):
    """Get latest dataset metadata by user_id and dataset_id"""
    dataset = await metadata_service.get_latest_dataset(dataset_id, user_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        dataset_id=dataset.dataset_id,
        user_id=dataset.user_id,
        name=dataset.name,
        description=dataset.description,
        version=dataset.version,
        file_type=dataset.file_type,
        file_size=dataset.file_size,
        original_filename=dataset.original_filename,
        columns=dataset.columns,
        row_count=dataset.row_count,
        data_types=dataset.data_types,
        tags=dataset.tags,
        custom_metadata=dataset.custom_metadata,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        file_hash=dataset.file_hash
    )


@router.get("/{user_id}", response_model=List[DatasetResponse])
async def get_user_datasets(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all datasets for a user with pagination"""
    datasets = await metadata_service.get_datasets_by_user(user_id, skip, limit)
    
    return [
        DatasetResponse(
            dataset_id=dataset.dataset_id,
            user_id=dataset.user_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            file_type=dataset.file_type,
            file_size=dataset.file_size,
            original_filename=dataset.original_filename,
            columns=dataset.columns,
            row_count=dataset.row_count,
            data_types=dataset.data_types,
            tags=dataset.tags,
            custom_metadata=dataset.custom_metadata,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            file_hash=dataset.file_hash
        )
        for dataset in datasets
    ]


@router.put("/{user_id}/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    user_id: str,
    dataset_id: str,
    update_data: DatasetUpdate
):
    """Update dataset metadata"""
    updated_dataset = await metadata_service.update_dataset_metadata(
        dataset_id, user_id, update_data
    )
    
    if not updated_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        dataset_id=updated_dataset.dataset_id,
        user_id=updated_dataset.user_id,
        name=updated_dataset.name,
        description=updated_dataset.description,
        version=updated_dataset.version,
        file_type=updated_dataset.file_type,
        file_size=updated_dataset.file_size,
        original_filename=updated_dataset.original_filename,
        columns=updated_dataset.columns,
        row_count=updated_dataset.row_count,
        data_types=updated_dataset.data_types,
        tags=updated_dataset.tags,
        custom_metadata=updated_dataset.custom_metadata,
        created_at=updated_dataset.created_at,
        updated_at=updated_dataset.updated_at,
        file_hash=updated_dataset.file_hash
    )


@router.delete("/{user_id}/{dataset_id}")
async def delete_dataset(user_id: str, dataset_id: str):
    """Delete all versions of a dataset and their files"""
    # Get all versions of the dataset first
    datasets = await metadata_service.get_dataset_versions(dataset_id, user_id)
    if not datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete all files from MinIO
    files_deleted = 0
    files_failed = 0
    for dataset in datasets:
        try:
            if dataset.is_folder:
                # Delete all files in folder
                deleted_count = await file_service.delete_folder_files(dataset.file_path)
                files_deleted += deleted_count
                logger.info(f"Deleted {deleted_count} files from folder: {dataset.file_path}")
            else:
                # Delete single file
                if await file_service.delete_file(dataset.file_path):
                    files_deleted += 1
                else:
                    files_failed += 1
        except Exception as e:
            logger.warning(f"Failed to delete files for {dataset.file_path}: {e}")
            files_failed += 1
    
    # Delete all metadata from MongoDB
    versions_deleted = await metadata_service.delete_all_dataset_versions(dataset_id, user_id)
    
    if versions_deleted == 0:
        raise HTTPException(status_code=500, detail="Failed to delete dataset metadata")
    
    return {
        "message": f"All {versions_deleted} version(s) of dataset deleted successfully",
        "versions_deleted": versions_deleted,
        "files_deleted": files_deleted,
        "files_failed": files_failed
    }


@router.get("/{user_id}/{dataset_id}/download")
async def download_dataset(
    user_id: str, 
    dataset_id: str,
    filename: Optional[str] = Query(None, description="Specific filename to download (for folders)")
):
    """
    Download latest dataset file or folder
    
    - For single files: downloads the file
    - For folders without filename: downloads all files as ZIP
    - For folders with filename: downloads specific file from folder
    """
    # Get latest dataset metadata
    dataset = await metadata_service.get_latest_dataset(dataset_id, user_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Handle folder downloads
    if dataset.is_folder:
        if filename:
            # Download specific file from folder
            dataset_files = dataset.files or []
            target_file = None
            
            for file_info in dataset_files:
                if file_info.filename == filename:
                    target_file = file_info
                    break
            
            if not target_file:
                raise HTTPException(status_code=404, detail=f"File '{filename}' not found in dataset")
            
            file_data = await file_service.download_file(target_file.file_path)
            
            return StreamingResponse(
                BytesIO(file_data),
                media_type='application/octet-stream',
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Download all files as ZIP
            zip_data = await file_service.download_folder_as_zip(dataset.file_path)
            
            return StreamingResponse(
                BytesIO(zip_data),
                media_type='application/zip',
                headers={"Content-Disposition": f"attachment; filename={dataset_id}_{dataset.version}.zip"}
            )
    else:
        # Single file download
        file_data = await file_service.download_file(dataset.file_path)
        
        return StreamingResponse(
            BytesIO(file_data),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={dataset.original_filename}"}
        )


@router.get("/{user_id}/{dataset_id}/version/{version}/download")
async def download_dataset_by_version(
    user_id: str, 
    dataset_id: str, 
    version: str,
    filename: Optional[str] = Query(None, description="Specific filename to download (for folders)")
):
    """
    Download a specific version of a dataset file or folder
    
    - For single files: downloads the file
    - For folders without filename: downloads all files as ZIP
    - For folders with filename: downloads specific file from folder
    """
    # Get dataset metadata for the specific version
    dataset = await metadata_service.get_dataset_by_version(dataset_id, user_id, version)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    
    # Handle folder downloads
    if dataset.is_folder:
        if filename:
            # Download specific file from folder
            dataset_files = dataset.files or []
            target_file = None
            
            for file_info in dataset_files:
                if file_info.filename == filename:
                    target_file = file_info
                    break
            
            if not target_file:
                raise HTTPException(status_code=404, detail=f"File '{filename}' not found in dataset")
            
            file_data = await file_service.download_file(target_file.file_path)
            
            return StreamingResponse(
                BytesIO(file_data),
                media_type='application/octet-stream',
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Download all files as ZIP
            zip_data = await file_service.download_folder_as_zip(dataset.file_path)
            
            return StreamingResponse(
                BytesIO(zip_data),
                media_type='application/zip',
                headers={"Content-Disposition": f"attachment; filename={dataset_id}_{version}.zip"}
            )
    else:
        # Single file download
        file_data = await file_service.download_file(dataset.file_path)
        
        return StreamingResponse(
            BytesIO(file_data),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={dataset.original_filename}"}
        )


@router.get("/{user_id}/{dataset_id}/files", response_model=List[DatasetFile])
async def list_dataset_files(
    user_id: str, 
    dataset_id: str,
    version: Optional[str] = Query(None, description="Specific version (defaults to latest)")
):
    """
    List all files in a dataset folder
    
    For single file datasets, returns a single-item list.
    For folder datasets, returns all files with metadata.
    """
    try:
        # Get dataset metadata
        if version:
            dataset = await metadata_service.get_dataset_by_version(dataset_id, user_id, version)
        else:
            dataset = await metadata_service.get_latest_dataset(dataset_id, user_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Return files list
        if dataset.is_folder and dataset.files:
            logger.info(f"Listed {len(dataset.files)} files for dataset {dataset_id} version {dataset.version}")
            return dataset.files
        else:
            # For single files, create a single-item list
            single_file = DatasetFile(
                filename=dataset.original_filename,
                file_path=dataset.file_path,
                file_size=dataset.file_size,
                file_type=dataset.file_type,
                file_hash=dataset.file_hash or "",
                content_type="application/octet-stream"
            )
            return [single_file]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing dataset files: {e}")
        raise HTTPException(status_code=500, detail="Failed to list dataset files")


@router.get("/{user_id}/{dataset_id}/versions", response_model=List[DatasetResponse])
async def get_dataset_versions(user_id: str, dataset_id: str):
    """Get all versions of a dataset"""
    datasets = await metadata_service.get_dataset_versions(dataset_id, user_id)
    
    return [
        DatasetResponse(
            dataset_id=dataset.dataset_id,
            user_id=dataset.user_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            file_type=dataset.file_type,
            file_size=dataset.file_size,
            original_filename=dataset.original_filename,
            columns=dataset.columns,
            row_count=dataset.row_count,
            data_types=dataset.data_types,
            tags=dataset.tags,
            custom_metadata=dataset.custom_metadata,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            file_hash=dataset.file_hash
        )
        for dataset in datasets
    ]


@router.get("/{user_id}/{dataset_id}/version/{version}", response_model=DatasetResponse)
async def get_dataset_by_version(user_id: str, dataset_id: str, version: str):
    """Get a specific version of a dataset"""
    dataset = await metadata_service.get_dataset_by_version(dataset_id, user_id, version)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(
        dataset_id=dataset.dataset_id,
        user_id=dataset.user_id,
        name=dataset.name,
        description=dataset.description,
        version=dataset.version,
        file_type=dataset.file_type,
        file_size=dataset.file_size,
        original_filename=dataset.original_filename,
        columns=dataset.columns,
        row_count=dataset.row_count,
        data_types=dataset.data_types,
        tags=dataset.tags,
        custom_metadata=dataset.custom_metadata,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        file_hash=dataset.file_hash
    )


@router.delete("/{user_id}/{dataset_id}/version/{version}")
async def delete_dataset_by_version(user_id: str, dataset_id: str, version: str):
    """Delete a specific version of a dataset and its file(s)"""
    dataset = await metadata_service.get_dataset_by_version(dataset_id, user_id, version)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Handle folder vs single file deletion
    files_deleted = 0
    if dataset.is_folder:
        files_deleted = await file_service.delete_folder_files(dataset.file_path)
        logger.info(f"Deleted {files_deleted} files from folder: {dataset.file_path}")
    else:
        file_deleted = await file_service.delete_file(dataset.file_path)
        files_deleted = 1 if file_deleted else 0
    
    metadata_deleted = await metadata_service.delete_dataset_by_version(dataset_id, user_id, version)
    if not metadata_deleted:
        raise HTTPException(status_code=500, detail="Failed to delete dataset metadata")
    
    return {
        "message": "Dataset deleted successfully",
        "files_deleted": files_deleted,
        "metadata_deleted": metadata_deleted
    }

