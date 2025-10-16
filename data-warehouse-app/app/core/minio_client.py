from minio import Minio
from minio.error import S3Error
from .config import settings
import logging

logger = logging.getLogger(__name__)


class MinIOClient:
    def __init__(self):
        self.client = None
        self.bucket_name = settings.minio_bucket_name

    async def connect(self):
        """Initialize MinIO client and create bucket if it doesn't exist"""
        try:
            self.client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure
            )
            
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"MinIO bucket '{self.bucket_name}' already exists")
                
            logger.info("Connected to MinIO successfully")
            
        except S3Error as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MinIO: {e}")
            raise

    def get_client(self):
        """Get MinIO client instance"""
        if not self.client:
            raise Exception("MinIO client not initialized. Call connect() first.")
        return self.client


# Global MinIO client instance
minio_client = MinIOClient()
