import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer
from ..core.config import settings
from ..models.dataset import DatasetMetadata

logger = logging.getLogger(__name__)


class KafkaProducerService:
    def __init__(self) -> None:
        self.producer: Optional[AIOKafkaProducer] = None
        self.bootstrap_servers = settings.kafka_bootstrap_servers
        self.client_id = settings.kafka_client_id

    async def start(self) -> None:
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
            )
            await self.producer.start()
            logger.info(f"Kafka producer started, servers={self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        if self.producer:
            try:
                await self.producer.stop()
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")

    async def send_dataset_event(
        self,
        event_type: str,
        dataset_metadata: DatasetMetadata,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.producer:
            logger.warning("Kafka producer not initialized; skipping event")
            return

        try:
            payload: Dict[str, Any] = {
                "event_type": event_type,
                "event_id": f"{event_type}_{dataset_metadata.dataset_id}_{int(datetime.utcnow().timestamp())}",
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": {
                    "dataset_id": dataset_metadata.dataset_id,
                    "user_id": dataset_metadata.user_id,
                    "name": dataset_metadata.name,
                    "description": dataset_metadata.description,
                    "version": dataset_metadata.version,
                    "file_type": dataset_metadata.file_type,
                    "file_size": dataset_metadata.file_size,
                    "file_path": dataset_metadata.file_path,
                    "original_filename": dataset_metadata.original_filename,
                    "is_folder": dataset_metadata.is_folder,
                    "file_count": len(dataset_metadata.files) if dataset_metadata.files else 1,
                    "columns": dataset_metadata.columns,
                    "row_count": dataset_metadata.row_count,
                    "data_types": dataset_metadata.data_types,
                    "tags": dataset_metadata.tags,
                    "custom_metadata": dataset_metadata.custom_metadata,
                    "created_at": dataset_metadata.created_at.isoformat(),
                    "updated_at": dataset_metadata.updated_at.isoformat(),
                    "file_hash": dataset_metadata.file_hash,
                },
            }
            if additional_data:
                payload.update(additional_data)

            key = dataset_metadata.dataset_id
            await self.producer.send_and_wait(settings.kafka_dataset_topic, value=payload, key=key)
            logger.info(f"Kafka event sent: {event_type} dataset_id={key}")
        except Exception as e:
            logger.warning(f"Failed to send Kafka event: {e}")

    async def send_dataset_uploaded_event(self, dataset_metadata: DatasetMetadata) -> None:
        await self.send_dataset_event("dataset.uploaded", dataset_metadata)

    async def send_dataset_updated_event(self, dataset_metadata: DatasetMetadata) -> None:
        await self.send_dataset_event("dataset.updated", dataset_metadata)

    async def send_dataset_deleted_event(self, dataset_metadata: DatasetMetadata) -> None:
        await self.send_dataset_event("dataset.deleted", dataset_metadata)

    async def send_bias_event(
        self, 
        *, 
        dataset_id: str, 
        user_id: str, 
        bias_report_id: str | None, 
        has_transformation_report: bool,
        target_column_name: Optional[str] = None,
        task_type: Optional[str] = None,
        is_folder: Optional[bool] = False,
        file_count: Optional[int] = 1
    ) -> None:
        if not self.producer:
            logger.warning("Kafka producer not initialized; skipping bias event")
            return
        payload = {
            "event_type": "bias.reported",
            "dataset_id": dataset_id,
            "user_id": user_id,
            "bias_report_id": bias_report_id,
            "has_transformation_report": has_transformation_report,
            "target_column_name": target_column_name,
            "task_type": task_type,
            "is_folder": is_folder,
            "file_count": file_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            await self.producer.send_and_wait(settings.kafka_bias_topic, value=payload, key=dataset_id)
            logger.info(f"Bias event sent for dataset_id={dataset_id}")
        except Exception as e:
            logger.warning(f"Failed to send bias event: {e}")

    async def send_automl_event(
        self,
        *,
        user_id: str,
        model_id: str,
        dataset_id: Optional[str] = None,
        version: str = "v1",
        framework: Optional[str] = None,
        model_type: Optional[str] = None,
        algorithm: Optional[str] = None,
        model_size_mb: Optional[float] = None,
        training_accuracy: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
        test_accuracy: Optional[float] = None,
        is_folder: Optional[bool] = False,
        file_count: Optional[int] = 1,
        is_model_folder: Optional[bool] = False,
        model_file_count: Optional[int] = 1
    ) -> None:
        """Send AutoML event when a model is uploaded"""
        if not self.producer:
            logger.warning("Kafka producer not initialized; skipping AutoML event")
            return
        
        payload = {
            "event_type": "model.uploaded",
            "event_id": f"model_uploaded_{model_id}_{int(datetime.utcnow().timestamp())}",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "version": version,
            "framework": framework,
            "model_type": model_type,
            "algorithm": algorithm,
            "model_size_mb": model_size_mb,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "is_folder": is_folder,
            "file_count": file_count,
            "is_model_folder": is_model_folder,
            "model_file_count": model_file_count,
        }
        
        try:
            key = f"{user_id}_{model_id}"
            await self.producer.send_and_wait(settings.kafka_automl_topic, value=payload, key=key)
            logger.info(f"AutoML event sent for model_id={model_id}, user_id={user_id}")
        except Exception as e:
            logger.warning(f"Failed to send AutoML event: {e}")

    async def send_xai_event(
        self,
        *,
        user_id: str,
        dataset_id: str,
        model_id: str,
        report_type: str,
        level: str,
        xai_report_id: Optional[str] = None
    ) -> None:
        """Send XAI event when an XAI report is uploaded"""
        if not self.producer:
            logger.warning("Kafka producer not initialized; skipping XAI event")
            return
        
        payload = {
            "event_type": "xai.reported",
            "event_id": f"xai_reported_{model_id}_{int(datetime.utcnow().timestamp())}",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "report_type": report_type,
            "level": level,
            "xai_report_id": xai_report_id,
        }
        
        try:
            key = f"{user_id}_{model_id}"
            await self.producer.send_and_wait(settings.kafka_xai_topic, value=payload, key=key)
            logger.info(f"XAI event sent for model_id={model_id}, user_id={user_id}, report_type={report_type}, level={level}")
        except Exception as e:
            logger.warning(f"Failed to send XAI event: {e}")


kafka_producer_service = KafkaProducerService()
