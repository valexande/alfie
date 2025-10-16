from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "data_warehouse"
    
    # MinIO Configuration
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_name: str = "data-warehouse"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_dataset_topic: str = "dataset-events"
    kafka_client_id: str = "data-warehouse-api"
    kafka_bias_topic: str = "bias-events"
    kafka_automl_topic: str = "automl-events"
    kafka_xai_topic: str = "xai-events"
    kafka_bias_trigger_topic: str = "bias-trigger-events"
    kafka_automl_trigger_topic: str = "automl-trigger-events"
    kafka_xai_trigger_topic: str = "xai-trigger-events"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
