from fastapi import APIRouter, HTTPException
from ..models.transformation_report import TransformationReportCreate, TransformationReportResponse
from ..services.transformation_report_service import transformation_report_service

router = APIRouter(prefix="/transformation-reports", tags=["transformation-reports"])


@router.post("/", response_model=TransformationReportResponse)
async def create_or_update_transformation_report(payload: TransformationReportCreate):
    try:
        # Placeholder: here you could also trigger saving a mitigated v2 dataset
        # Example (commented out to avoid re-triggering upload consumer):
        # from ..services.file_service import file_service
        # mitigated_bytes = run_mitigator_somewhere(...)
        # await file_service.save_new_version(user_id=payload.user_id, dataset_id=payload.dataset_id, version=payload.version, data=mitigated_bytes)
        # This would create datasets/{user_id}/{dataset_id}/{version}/... and store metadata
        report = await transformation_report_service.upsert_report(payload)
        return report
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save transformation report")


@router.get("/{user_id}/{dataset_id}/{version}", response_model=TransformationReportResponse)
async def get_transformation_report(user_id: str, dataset_id: str, version: str):
    report = await transformation_report_service.get_report(user_id, dataset_id, version)
    if not report:
        raise HTTPException(status_code=404, detail="Transformation report not found")
    return report


@router.delete("/{user_id}/{dataset_id}/{version}")
async def delete_transformation_report(user_id: str, dataset_id: str, version: str):
    """Delete a transformation report"""
    deleted = await transformation_report_service.delete_report(user_id, dataset_id, version)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transformation report not found")
    return {"message": "Transformation report deleted successfully"}
