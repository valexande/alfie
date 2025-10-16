from typing import Optional
from datetime import datetime
from ..core.database import get_database
from ..models.transformation_report import (
    TransformationReportCreate,
    TransformationReportResponse,
)


class TransformationReportService:
    def __init__(self) -> None:
        self.collection = None

    async def initialize(self) -> None:
        db = get_database()
        self.collection = db.transformation_reports

    async def upsert_report(self, data: TransformationReportCreate) -> TransformationReportResponse:
        now = datetime.utcnow()
        update = {
            "$set": {
                "user_id": data.user_id,
                "dataset_id": data.dataset_id,
                "version": data.version,
                "report": data.report,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        }
        await self.collection.update_one(
            {"user_id": data.user_id, "dataset_id": data.dataset_id, "version": data.version},
            update,
            upsert=True,
        )
        doc = await self.collection.find_one({"user_id": data.user_id, "dataset_id": data.dataset_id, "version": data.version})
        return TransformationReportResponse(
            user_id=doc["user_id"],
            dataset_id=doc["dataset_id"],
            version=doc["version"],
            report=doc["report"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    async def get_report(self, user_id: str, dataset_id: str, version: str) -> Optional[TransformationReportResponse]:
        doc = await self.collection.find_one({"user_id": user_id, "dataset_id": dataset_id, "version": version})
        if not doc:
            return None
        return TransformationReportResponse(
            user_id=doc["user_id"],
            dataset_id=doc["dataset_id"],
            version=doc["version"],
            report=doc["report"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    async def delete_report(self, user_id: str, dataset_id: str, version: str) -> bool:
        """Delete a transformation report"""
        result = await self.collection.delete_one({"user_id": user_id, "dataset_id": dataset_id, "version": version})
        return result.deleted_count > 0


transformation_report_service = TransformationReportService()

