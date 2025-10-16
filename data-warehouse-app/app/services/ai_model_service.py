import hashlib
import os
import zipfile
import tempfile
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from minio.error import S3Error
from fastapi import UploadFile, HTTPException
from ..core.minio_client import minio_client
from ..models.ai_model import AIModelMetadata, ModelFile, ModelFramework, ModelType
from ..core.database import get_database
import logging

logger = logging.getLogger(__name__)


class AIModelService:
    def __init__(self):
        self.client = None
        self.bucket_name = minio_client.bucket_name
        self.db = None

    async def initialize(self):
        """Initialize the AI model service"""
        await minio_client.connect()
        self.client = minio_client.get_client()
        self.db = get_database()
        self.collection = self.db.ai_models

    def generate_model_path(self, user_id: str, model_id: str, version: str, filename: str) -> str:
        """Generate organized model path with user separation and versioning"""
        # Format: models/user1/model_name/v1/filename.pkl
        return f"models/{user_id}/{model_id}/{version}/{filename}"

    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate MD5 hash of file data"""
        return hashlib.md5(file_data).hexdigest()

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename)[1].lower().lstrip('.')

    def _detect_model_framework(self, filename: str, content_type: Optional[str]) -> Optional[ModelFramework]:
        """Detect model framework based on file extension and content"""
        extension = self._get_file_extension(filename).lower()
        
        framework_map = {
            'pkl': ModelFramework.SKLEARN,
            'joblib': ModelFramework.SKLEARN,
            'onnx': ModelFramework.ONNX,
            'h5': ModelFramework.KERAS,
            'hdf5': ModelFramework.KERAS,
            'pb': ModelFramework.TENSORFLOW,
            'pth': ModelFramework.PYTORCH,
            'pt': ModelFramework.PYTORCH,
            'bin': ModelFramework.HUGGINGFACE,
            'safetensors': ModelFramework.HUGGINGFACE,
        }
        
        return framework_map.get(extension)

    async def upload_model_file(
        self, 
        file: UploadFile, 
        user_id: str, 
        model_id: str, 
        version: str = "v1",
        is_primary: bool = False,
        description: Optional[str] = None
    ) -> Tuple[str, ModelFile]:
        """
        Upload a single model file to MinIO
        
        Returns:
            Tuple of (file_path, ModelFile object)
        """
        try:
            # Read file data
            file_data = await file.read()
            file_size = len(file_data)
            
            # Generate file path
            file_path = self.generate_model_path(user_id, model_id, version, file.filename)
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_data)
            
            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=file_path,
                data=BytesIO(file_data),
                length=file_size,
                content_type=file.content_type or 'application/octet-stream'
            )
            
            logger.info(f"Model file uploaded successfully: {file_path}")
            
            # Create ModelFile object
            model_file = ModelFile(
                filename=file.filename,
                file_path=file_path,
                file_size=file_size,
                file_type=self._get_file_extension(file.filename),
                file_hash=file_hash,
                content_type=file.content_type,
                is_primary=is_primary,
                description=description
            )
            
            return file_path, model_file
            
        except S3Error as e:
            logger.error(f"MinIO error during model file upload: {e}")
            raise HTTPException(status_code=500, detail=f"Model file upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during model file upload: {e}")
            raise HTTPException(status_code=500, detail=f"Model file upload failed: {str(e)}")

    async def upload_model_folder(
        self, 
        zip_file: UploadFile, 
        user_id: str, 
        model_id: str, 
        version: str = "v1",
        preserve_structure: bool = True
    ) -> List[ModelFile]:
        """
        Upload a folder of model files (as zip) to MinIO
        
        Returns:
            List of ModelFile objects
        """
        try:
            # Read zip file data
            zip_data = await zip_file.read()
            
            model_files = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                zip_path = os.path.join(temp_dir, "model_files.zip")
                with open(zip_path, 'wb') as f:
                    f.write(zip_data)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Walk through extracted files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file == "model_files.zip":
                            continue
                            
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, temp_dir)
                        
                        # Read file data
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        file_size = len(file_data)
                        
                        # Determine if this should be the primary file
                        is_primary = self._is_primary_model_file(file)
                        
                        # Generate MinIO path
                        if preserve_structure:
                            minio_path = self.generate_model_path(user_id, model_id, version, relative_path)
                        else:
                            minio_path = self.generate_model_path(user_id, model_id, version, file)
                        
                        # Calculate file hash
                        file_hash = self.calculate_file_hash(file_data)
                        
                        # Upload to MinIO
                        self.client.put_object(
                            bucket_name=self.bucket_name,
                            object_name=minio_path,
                            data=BytesIO(file_data),
                            length=file_size,
                            content_type='application/octet-stream'
                        )
                        
                        # Create ModelFile object
                        model_file = ModelFile(
                            filename=file,
                            file_path=minio_path,
                            file_size=file_size,
                            file_type=self._get_file_extension(file),
                            file_hash=file_hash,
                            content_type='application/octet-stream',
                            is_primary=is_primary,
                            description=f"Uploaded from folder structure"
                        )
                        
                        model_files.append(model_file)
                        
                        logger.info(f"Model file uploaded from folder: {minio_path}")
            
            return model_files
            
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")
        except S3Error as e:
            logger.error(f"MinIO error during model folder upload: {e}")
            raise HTTPException(status_code=500, detail=f"Model folder upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during model folder upload: {e}")
            raise HTTPException(status_code=500, detail=f"Model folder upload failed: {str(e)}")

    def _is_primary_model_file(self, filename: str) -> bool:
        """Determine if a file should be considered the primary model file"""
        extension = self._get_file_extension(filename).lower()
        
        # Common primary model file extensions
        primary_extensions = ['pkl', 'joblib', 'onnx', 'h5', 'hdf5', 'pb', 'pth', 'pt']
        
        return extension in primary_extensions

    async def download_model_file(self, file_path: str) -> bytes:
        """Download model file from MinIO"""
        try:
            response = self.client.get_object(self.bucket_name, file_path)
            return response.read()
        except S3Error as e:
            logger.error(f"MinIO error during model file download: {e}")
            raise HTTPException(status_code=404, detail="Model file not found")
        except Exception as e:
            logger.error(f"Unexpected error during model file download: {e}")
            raise HTTPException(status_code=500, detail="Model file download failed")

    async def download_model_as_zip(self, user_id: str, model_id: str, version: str) -> bytes:
        """Download all model files as a zip archive"""
        try:
            # List all files for the model
            prefix = f"models/{user_id}/{model_id}/{version}/"
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            
            if not objects:
                raise HTTPException(status_code=404, detail="No model files found")
            
            # Create zip file in memory
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for obj in objects:
                    # Download file
                    file_data = self.client.get_object(self.bucket_name, obj.object_name)
                    
                    # Add to zip (remove the prefix from the path)
                    zip_path = obj.object_name.replace(prefix, "")
                    zip_file.writestr(zip_path, file_data.read())
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
            
        except S3Error as e:
            logger.error(f"MinIO error during model zip download: {e}")
            raise HTTPException(status_code=404, detail="Model files not found")
        except Exception as e:
            logger.error(f"Unexpected error during model zip download: {e}")
            raise HTTPException(status_code=500, detail="Model zip download failed")

    async def delete_model_files(self, user_id: str, model_id: str, version: str) -> bool:
        """Delete all model files from MinIO"""
        try:
            prefix = f"models/{user_id}/{model_id}/{version}/"
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            
            for obj in objects:
                self.client.remove_object(self.bucket_name, obj.object_name)
                logger.info(f"Deleted model file: {obj.object_name}")
            
            return True
            
        except S3Error as e:
            logger.error(f"MinIO error during model file deletion: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model file deletion: {e}")
            return False

    async def list_model_files(self, user_id: str, model_id: str, version: str) -> List[str]:
        """List all files for a specific model version"""
        try:
            prefix = f"models/{user_id}/{model_id}/{version}/"
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"MinIO error listing model files: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing model files: {e}")
            return []

    async def calculate_model_size(self, model_files: List[ModelFile]) -> float:
        """Calculate total model size in MB"""
        total_size = sum(file.file_size for file in model_files)
        return total_size / (1024 * 1024)  # Convert to MB

    async def detect_primary_file(self, model_files: List[ModelFile]) -> Optional[ModelFile]:
        """Detect the primary model file from a list of files"""
        # First, check if any file is explicitly marked as primary
        primary_files = [f for f in model_files if f.is_primary]
        if primary_files:
            return primary_files[0]
        
        # If no explicit primary, find the most likely candidate
        for file in model_files:
            if self._is_primary_model_file(file.filename):
                return file
        
        # If no primary detected, return the largest file
        if model_files:
            return max(model_files, key=lambda f: f.file_size)
        
        return None


# Global AI model service instance
ai_model_service = AIModelService()
