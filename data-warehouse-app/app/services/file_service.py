import hashlib
import os
import pandas as pd
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from minio.error import S3Error
from fastapi import UploadFile, HTTPException
from ..core.minio_client import minio_client
from ..models.dataset import DatasetMetadata, DatasetFile
import logging
import tempfile
import zipfile

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self):
        self.client = None
        self.bucket_name = minio_client.bucket_name

    async def initialize(self):
        """Initialize the file service"""
        await minio_client.connect()
        self.client = minio_client.get_client()

    def generate_file_path(self, user_id: str, dataset_id: str, version: str, filename: str) -> str:
        """Generate organized file path with user separation and versioning"""
        # Format: datasets/user1/dataset_name/v1/filename.csv
        return f"datasets/{user_id}/{dataset_id}/{version}/{filename}"

    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate MD5 hash of file data"""
        return hashlib.md5(file_data).hexdigest()

    async def upload_file(
        self, 
        file: UploadFile, 
        user_id: str, 
        dataset_id: str, 
        version: str = "v1"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Upload file to MinIO and return file path and metadata
        
        Returns:
            Tuple of (file_path, metadata_dict)
        """
        try:
            # Read file data
            file_data = await file.read()
            file_size = len(file_data)
            
            # Generate file path
            file_path = self.generate_file_path(user_id, dataset_id, version, file.filename)
            
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
            
            logger.info(f"File uploaded successfully: {file_path}")
            
            # Extract metadata based on file type
            metadata = await self._extract_file_metadata(file_data, file.filename, file.content_type)
            
            # Add basic file information
            metadata.update({
                'file_size': file_size,
                'file_hash': file_hash,
                'file_type': self._get_file_extension(file.filename),
                'original_filename': file.filename
            })
            
            return file_path, metadata
            
        except S3Error as e:
            logger.error(f"MinIO error during file upload: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    async def download_file(self, file_path: str) -> bytes:
        """Download file from MinIO"""
        try:
            response = self.client.get_object(self.bucket_name, file_path)
            return response.read()
        except S3Error as e:
            logger.error(f"MinIO error during file download: {e}")
            raise HTTPException(status_code=404, detail="File not found")
        except Exception as e:
            logger.error(f"Unexpected error during file download: {e}")
            raise HTTPException(status_code=500, detail="File download failed")

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, file_path)
            logger.info(f"File deleted successfully: {file_path}")
            return True
        except S3Error as e:
            logger.error(f"MinIO error during file deletion: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during file deletion: {e}")
            return False

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket_name, file_path)
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    async def list_user_files(self, user_id: str, dataset_id: Optional[str] = None) -> List[str]:
        """List all files for a user, optionally filtered by dataset"""
        try:
            prefix = f"datasets/{user_id}/"
            if dataset_id:
                prefix += f"{dataset_id}/"
            
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"MinIO error listing files: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}")
            return []

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename)[1].lower().lstrip('.')

    async def _extract_file_metadata(self, file_data: bytes, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
        """Extract metadata from file content based on file type"""
        metadata = {}
        file_extension = self._get_file_extension(filename)
        
        try:
            if file_extension in ['csv', 'xlsx', 'xls']:
                # For structured data files
                if file_extension == 'csv':
                    df = pd.read_csv(BytesIO(file_data))
                else:
                    df = pd.read_excel(BytesIO(file_data))
                
                metadata.update({
                    'columns': df.columns.tolist(),
                    'row_count': len(df),
                    'data_types': df.dtypes.astype(str).to_dict()
                })
                
            elif file_extension in ['json']:
                # For JSON files, we could parse and extract schema
                pass
                
            elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                # For image files, we could extract dimensions, etc.
                pass
                
            elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
                # For video files, we could extract duration, resolution, etc.
                pass
                
            elif file_extension in ['mp3', 'wav', 'flac']:
                # For audio files, we could extract duration, bitrate, etc.
                pass
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {filename}: {e}")
            
        return metadata
    
    async def upload_dataset_folder(
        self,
        zip_file: UploadFile,
        user_id: str,
        dataset_id: str,
        version: str = "v1",
        preserve_structure: bool = True
    ) -> Tuple[List[DatasetFile], int]:
        """
        Upload a folder of dataset files (as zip) to MinIO
        
        Returns:
            Tuple of (List of DatasetFile objects, total size in bytes)
        """
        try:
            # Read zip file data
            zip_data = await zip_file.read()
            
            dataset_files = []
            total_size = 0
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                zip_path = os.path.join(temp_dir, "dataset_files.zip")
                with open(zip_path, 'wb') as f:
                    f.write(zip_data)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Walk through extracted files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file == "dataset_files.zip":
                            continue
                        
                        # Skip hidden files and directories
                        if file.startswith('.') or '__MACOSX' in root:
                            continue
                        
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, temp_dir)
                        
                        # Read file data
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        file_size = len(file_data)
                        total_size += file_size
                        
                        # Calculate file hash
                        file_hash = self.calculate_file_hash(file_data)
                        
                        # Generate MinIO path
                        if preserve_structure:
                            minio_path = self.generate_file_path(user_id, dataset_id, version, relative_path)
                        else:
                            minio_path = self.generate_file_path(user_id, dataset_id, version, file)
                        
                        # Upload to MinIO
                        self.client.put_object(
                            bucket_name=self.bucket_name,
                            object_name=minio_path,
                            data=BytesIO(file_data),
                            length=file_size,
                            content_type='application/octet-stream'
                        )
                        
                        # Create DatasetFile object
                        dataset_file = DatasetFile(
                            filename=file,
                            file_path=minio_path,
                            file_size=file_size,
                            file_type=self._get_file_extension(file),
                            file_hash=file_hash,
                            content_type='application/octet-stream'
                        )
                        dataset_files.append(dataset_file)
                        
                        logger.info(f"Uploaded dataset file: {minio_path}")
            
            logger.info(f"Uploaded {len(dataset_files)} dataset files, total size: {total_size} bytes")
            return dataset_files, total_size
            
        except zipfile.BadZipFile:
            logger.error("Invalid zip file")
            raise HTTPException(status_code=400, detail="Invalid zip file")
        except S3Error as e:
            logger.error(f"MinIO error during folder upload: {e}")
            raise HTTPException(status_code=500, detail=f"Folder upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during folder upload: {e}")
            raise HTTPException(status_code=500, detail=f"Folder upload failed: {str(e)}")
    
    async def delete_folder_files(self, folder_path: str) -> int:
        """
        Delete all files in a folder from MinIO
        
        Returns:
            Number of files deleted
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=folder_path, recursive=True)
            deleted_count = 0
            
            for obj in objects:
                try:
                    self.client.remove_object(self.bucket_name, obj.object_name)
                    logger.info(f"Deleted file: {obj.object_name}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {obj.object_name}: {e}")
            
            return deleted_count
            
        except S3Error as e:
            logger.error(f"MinIO error during folder deletion: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during folder deletion: {e}")
            return 0
    
    async def download_folder_as_zip(self, folder_path: str) -> bytes:
        """
        Download all files in a folder as a zip archive
        
        Args:
            folder_path: Path to folder in MinIO (e.g., "datasets/user1/dataset1/v1/")
            
        Returns:
            ZIP archive as bytes
        """
        try:
            # List all files in the folder
            objects = self.client.list_objects(self.bucket_name, prefix=folder_path, recursive=True)
            
            # Create zip file in memory
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                has_files = False
                for obj in objects:
                    # Download file
                    file_data = self.client.get_object(self.bucket_name, obj.object_name)
                    
                    # Add to zip (remove the prefix from the path)
                    zip_path = obj.object_name.replace(folder_path, "")
                    if zip_path:  # Skip empty paths
                        zip_file.writestr(zip_path, file_data.read())
                        has_files = True
                
                if not has_files:
                    raise HTTPException(status_code=404, detail="No files found in folder")
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
            
        except S3Error as e:
            logger.error(f"MinIO error during folder zip download: {e}")
            raise HTTPException(status_code=404, detail="Folder not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during folder zip download: {e}")
            raise HTTPException(status_code=500, detail="Folder zip download failed")


# Global file service instance
file_service = FileService()
