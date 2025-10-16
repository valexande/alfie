from typing import Optional
from datetime import datetime
from .metadata_service import metadata_service
from ..core.database import get_database
from ..models.bias_report import BiasReportCreate, BiasReportResponse


class BiasReportService:
    def __init__(self) -> None:
        self.collection = None

    async def initialize(self) -> None:
        db = get_database()
        self.collection = db.bias_reports

    async def upsert_report(self, data: BiasReportCreate) -> BiasReportResponse:
        # Ensure dataset exists
        dataset = await metadata_service.get_dataset_by_id(data.dataset_id, data.user_id)
        if not dataset:
            raise ValueError("Dataset not found for bias report")

        now = datetime.utcnow()
        update = {
            "$set": {
                "user_id": data.user_id,
                "dataset_id": data.dataset_id,
                "report": data.report,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        }
        await self.collection.update_one(
            {"user_id": data.user_id, "dataset_id": data.dataset_id}, update, upsert=True
        )

        doc = await self.collection.find_one({"user_id": data.user_id, "dataset_id": data.dataset_id})
        return BiasReportResponse(
            id=str(doc.get("_id")) if doc and doc.get("_id") else None,
            user_id=doc["user_id"],
            dataset_id=doc["dataset_id"],
            report=doc["report"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    async def get_report(self, user_id: str, dataset_id: str) -> Optional[BiasReportResponse]:
        doc = await self.collection.find_one({"user_id": user_id, "dataset_id": dataset_id})
        if not doc:
            return None
        return BiasReportResponse(
            id=str(doc.get("_id")) if doc and doc.get("_id") else None,
            user_id=doc["user_id"],
            dataset_id=doc["dataset_id"],
            report=doc["report"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    async def delete_report(self, user_id: str, dataset_id: str) -> bool:
        """Delete a bias report"""
        result = await self.collection.delete_one({"user_id": user_id, "dataset_id": dataset_id})
        return result.deleted_count > 0


bias_report_service = BiasReportService()
