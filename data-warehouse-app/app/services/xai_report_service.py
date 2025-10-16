import hashlib
from typing import Optional, List
from datetime import datetime
from fastapi import UploadFile, HTTPException
from ..core.database import get_database
from ..core.minio_client import minio_client
from ..models.xai_report import (
    XAIReportMetadata,
    XAIReportCreate,
    XAIReportResponse,
    ReportType,
    ExpertiseLevel
)
import logging

logger = logging.getLogger(__name__)


class XAIReportService:
    def __init__(self):
        self.client = None
        self.bucket_name = minio_client.bucket_name
        self.db = None

    async def initialize(self):
        """Initialize the XAI report service"""
        await minio_client.connect()
        self.client = minio_client.get_client()
        self.db = get_database()
        self.collection = self.db.xai_reports

    def generate_file_path(
        self,
        user_id: str,
        dataset_id: str,
        model_id: str,
        report_type: str,
        level: str
    ) -> str:
        """Generate organized path for XAI report"""
        # Format: xai_reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}/report.html
        return f"xai_reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}/report.html"

    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate MD5 hash of file data"""
        return hashlib.md5(file_data).hexdigest()

    async def upload_report(
        self,
        file: UploadFile,
        report_data: XAIReportCreate
    ) -> XAIReportResponse:
        """Upload XAI report HTML file and store metadata"""
        try:
            # Read file data
            file_data = await file.read()
            file_size = len(file_data)

            # Validate it's an HTML file
            if not file.filename.endswith('.html'):
                raise HTTPException(status_code=400, detail="Only HTML files are supported")

            # Generate file path
            file_path = self.generate_file_path(
                report_data.user_id,
                report_data.dataset_id,
                report_data.model_id,
                report_data.report_type,
                report_data.level
            )

            # Calculate file hash
            file_hash = self.calculate_file_hash(file_data)

            # Check if report already exists
            existing_report = await self.collection.find_one({
                "user_id": report_data.user_id,
                "dataset_id": report_data.dataset_id,
                "model_id": report_data.model_id,
                "report_type": report_data.report_type,
                "level": report_data.level
            })

            # Upload to MinIO
            from io import BytesIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=file_path,
                data=BytesIO(file_data),
                length=file_size,
                content_type='text/html'
            )

            logger.info(f"XAI report uploaded successfully: {file_path}")

            # Create or update metadata
            metadata = {
                "user_id": report_data.user_id,
                "dataset_id": report_data.dataset_id,
                "model_id": report_data.model_id,
                "report_type": report_data.report_type,
                "level": report_data.level,
                "file_path": file_path,
                "file_size": file_size,
                "file_hash": file_hash,
                "custom_metadata": report_data.custom_metadata,
                "updated_at": datetime.utcnow()
            }

            if existing_report:
                # Update existing report
                metadata["created_at"] = existing_report["created_at"]
                await self.collection.update_one(
                    {"_id": existing_report["_id"]},
                    {"$set": metadata}
                )
            else:
                # Create new report
                metadata["created_at"] = datetime.utcnow()
                await self.collection.insert_one(metadata)

            return XAIReportResponse(**metadata)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading XAI report: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload XAI report: {str(e)}")

    async def get_report(
        self,
        user_id: str,
        dataset_id: str,
        model_id: str,
        report_type: ReportType,
        level: ExpertiseLevel
    ) -> Optional[XAIReportResponse]:
        """Get specific XAI report metadata"""
        try:
            report = await self.collection.find_one({
                "user_id": user_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "report_type": report_type,
                "level": level
            })

            if report:
                return XAIReportResponse(**report)
            return None

        except Exception as e:
            logger.error(f"Error retrieving XAI report: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve XAI report")

    async def get_all_reports(
        self,
        user_id: str,
        dataset_id: str,
        model_id: str
    ) -> List[XAIReportResponse]:
        """Get all XAI reports for a specific model and dataset"""
        try:
            cursor = self.collection.find({
                "user_id": user_id,
                "dataset_id": dataset_id,
                "model_id": model_id
            })

            reports = await cursor.to_list(length=None)
            return [XAIReportResponse(**report) for report in reports]

        except Exception as e:
            logger.error(f"Error retrieving XAI reports: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve XAI reports")

    async def download_report(
        self,
        file_path: str
    ) -> bytes:
        """Download XAI report HTML file from MinIO"""
        try:
            response = self.client.get_object(self.bucket_name, file_path)
            return response.read()
        except Exception as e:
            logger.error(f"Error downloading XAI report: {e}")
            raise HTTPException(status_code=404, detail="XAI report file not found")

    async def delete_report(
        self,
        user_id: str,
        dataset_id: str,
        model_id: str,
        report_type: ReportType,
        level: ExpertiseLevel
    ) -> bool:
        """Delete XAI report and its file"""
        try:
            # Get report metadata
            report = await self.get_report(user_id, dataset_id, model_id, report_type, level)
            if not report:
                return False

            # Delete file from MinIO
            try:
                self.client.remove_object(self.bucket_name, report.file_path)
                logger.info(f"Deleted XAI report file: {report.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file from MinIO: {e}")

            # Delete metadata from MongoDB
            result = await self.collection.delete_one({
                "user_id": user_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "report_type": report_type,
                "level": level
            })

            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting XAI report: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete XAI report")


# Global XAI report service instance
xai_report_service = XAIReportService()

