#!/usr/bin/env python3
"""
Kafka Consumer Example for XAI Trigger Events

This consumer listens to the xai-trigger-events topic (from Agentic Core).
When an XAI trigger event is received, this consumer should:
- Load the dataset and model from the Data Warehouse
- Generate XAI explanations (SHAP, LIME, etc.)
- Create HTML reports for different expertise levels
- Upload the reports to the Data Warehouse (which triggers xai-events)

Usage:
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  KAFKA_XAI_TRIGGER_TOPIC=xai-trigger-events \
  KAFKA_CONSUMER_GROUP=xai-consumer \
  python kafka_xai_consumer_example.py
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from aiokafka import AIOKafkaConsumer
import requests
import pandas as pd
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("kafka_xai_consumer")

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_XAI_TRIGGER_TOPIC = os.getenv("KAFKA_XAI_TRIGGER_TOPIC", "xai-trigger-events")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "xai-consumer")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def fetch_dataset_metadata(user_id: str, dataset_id: str, version: str = None) -> dict:
    """Fetch dataset metadata from the Data Warehouse API (specific version or latest)"""
    if version:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}"
    else:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def read_csv_with_encoding(file_data: bytes) -> pd.DataFrame:
    """
    Try to read CSV with multiple encodings
    
    Handles files with different encodings:
    - utf-8: Standard
    - latin-1 (ISO-8859-1): Western European
    - cp1252 (Windows-1252): Windows default
    - utf-16: Some Excel exports
    """
    from io import BytesIO
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(BytesIO(file_data), encoding=encoding)
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    
    # If all encodings fail, try with error handling
    try:
        df = pd.read_csv(BytesIO(file_data), encoding='utf-8', encoding_errors='ignore')
        logger.warning("Read CSV with 'ignore' errors - some characters may be missing")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV with all encodings: {e}")
        raise


def download_dataset_file(user_id: str, dataset_id: str, version: str = None) -> bytes:
    """Download dataset file (single file or folder as ZIP) - specific version or latest"""
    if version:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}/download"
    else:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/download"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def extract_dataset_folder(zip_bytes: bytes, extract_to: str = "temp_dataset") -> list:
    """
    Extract ZIP file containing dataset folder
    
    Returns:
        List of extracted file paths
    """
    import zipfile
    from io import BytesIO
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # List extracted files
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)
    
    return extracted_files


def fetch_model_metadata(user_id: str, model_id: str, version: str = "v1") -> dict:
    """Fetch AI model metadata from the Data Warehouse API"""
    url = f"{API_BASE}/ai-models/{user_id}/{model_id}"
    params = {"version": version}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def download_model_file(user_id: str, model_id: str, version: str = "v1", filename: str = None) -> bytes:
    """Download AI model file (single file or folder as ZIP)"""
    url = f"{API_BASE}/ai-models/{user_id}/{model_id}/download"
    params = {"version": version}
    if filename:
        params["filename"] = filename
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.content


def extract_model_folder(zip_bytes: bytes, extract_to: str = "temp_model") -> list:
    """
    Extract ZIP file containing model folder
    
    Returns:
        List of extracted file paths
    """
    import zipfile
    from io import BytesIO
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # List extracted files
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)
    
    return extracted_files


def upload_xai_report(user_id: str, dataset_id: str, dataset_version: str,
                     model_id: str, model_version: str,
                     report_type: str, level: str, html_file_path: str,
                     task_id: str | None = None) -> dict:
    """
    Upload XAI report to Data Warehouse
    This will automatically trigger an xai-events message
    """
    url = f"{API_BASE}/xai-reports/upload/{user_id}"
    
    with open(html_file_path, 'rb') as f:
        files = {'file': (os.path.basename(html_file_path), f, 'text/html')}
        data = {
            'dataset_id': dataset_id,
            'dataset_version': dataset_version,
            'model_id': model_id,
            'model_version': model_version,
            'report_type': report_type,
            'level': level
        }
        
        headers = {"X-Task-ID": task_id} if task_id else None
        r = requests.post(url, files=files, data=data, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()


async def process_xai_trigger(event: dict) -> None:
    """
    Process an XAI trigger event from Agentic Core
    
    Simplified event structure:
    {
        "task_id": "xai_task_<...>",
        "event_type": "xai-trigger",
        "timestamp": "...",
        "input": {
            "user_id": "user123",
            "dataset_id": "dataset123",
            "dataset_version": "v1",
            "model_id": "model123",
            "model_version": "v1",
            "report_type": "lime",
            "level": "beginner"
        }
    }
    """
    try:
        task_id = event.get("task_id")
        input_obj = event.get("input", {})
        user_id = input_obj.get("user_id")
        dataset_id = input_obj.get("dataset_id")
        dataset_version = input_obj.get("dataset_version", "v1")  # Default to v1 for backward compatibility
        model_id = input_obj.get("model_id")
        model_version = input_obj.get("model_version", "v1")  # Default to v1 for backward compatibility
        level = input_obj.get("level", "beginner")
        report_type = input_obj.get("report_type", "lime")
        
        # Simplified schema doesn't include these; default values
        is_folder = False
        file_count = 1
        is_model_folder = False
        model_file_count = 1
        
        if not user_id or not dataset_id or not model_id:
            logger.warning("Missing required fields in event; skipping")
            return
        
        logger.info(f"Processing XAI trigger for model {model_id} version {model_version}")
        logger.info(f"  User: {user_id}")
        logger.info(f"  Dataset: {dataset_id} version {dataset_version}")
        logger.info(f"  Level: {level}")
        logger.info(f"  Dataset type: {'FOLDER' if is_folder else 'SINGLE FILE'} ({file_count} file(s))")
        logger.info(f"  Model type: {'FOLDER' if is_model_folder else 'SINGLE FILE'} ({model_file_count} file(s))")
        
        # Step 1: Fetch and download dataset
        try:
            dataset_meta = fetch_dataset_metadata(user_id, dataset_id, dataset_version)
            logger.info(f"Dataset metadata fetched: {dataset_meta.get('name')} (version {dataset_version})")
            
            # Download dataset (single file or ZIP)
            dataset_bytes = download_dataset_file(user_id, dataset_id, dataset_version)
            logger.info(f"Dataset downloaded: {len(dataset_bytes)} bytes")
            
            # Handle dataset based on type
            dataset_extracted_files = []
            if is_folder:
                logger.info("Extracting folder dataset...")
                dataset_extracted_files = extract_dataset_folder(dataset_bytes, f"temp_xai_dataset_{dataset_id}")
                logger.info(f"Extracted {len(dataset_extracted_files)} dataset files:")
                for file_path in dataset_extracted_files:
                    logger.info(f"  - {file_path}")
            else:
                logger.info("Dataset is single file - ready for XAI analysis")
            
        except Exception as e:
            logger.error(f"Failed to fetch dataset: {e}")
            return
        
        # Step 2: Fetch and download model
        try:
            model_meta = fetch_model_metadata(user_id, model_id, model_version)
            logger.info(f"Model metadata fetched: {model_meta.get('name')} (version {model_version})")
            logger.info(f"  Framework: {model_meta.get('framework')}")
            logger.info(f"  Files: {len(model_meta.get('files', []))}")
            
            # Download model (single file or folder as ZIP)
            model_bytes = download_model_file(user_id, model_id, model_version)
            logger.info(f"Model downloaded: {len(model_bytes)} bytes")
            
            # Handle model based on type
            model_extracted_files = []
            if is_model_folder:
                logger.info("Extracting model folder...")
                model_extracted_files = extract_model_folder(model_bytes, f"temp_xai_model_{model_id}")
                logger.info(f"Extracted {len(model_extracted_files)} model files:")
                for file_path in model_extracted_files:
                    logger.info(f"  - {file_path}")
                logger.info("NOTE: All model files are available for XAI analysis")
            else:
                logger.info("Model is single file - ready for XAI analysis")
            
        except Exception as e:
            logger.error(f"Failed to fetch model: {e}")
            return
        
        # Step 3: Generate XAI explanations
        # TODO: Replace this with actual XAI generation logic
        # For now, we'll use dummy HTML files for testing
        
        logger.info("=" * 80)
        logger.info("Generating XAI explanations (using dummy HTML files for testing)")
        logger.info(f"  - Model: {model_meta.get('name')}")
        logger.info(f"  - Framework: {model_meta.get('framework')}")
        logger.info(f"  - Level: {level}")
        logger.info("=" * 80)
        
        # 4 Possible Combinations:
        # 1. Single Dataset + Single Model (most common)
        # 2. Single Dataset + Model Folder (model.pkl + label_encoders.pkl)
        # 3. Folder Dataset + Single Model (train.csv, test.csv + model.pkl)
        # 4. Folder Dataset + Model Folder (multiple data files + multiple model files)
        
        logger.info("XAI Resources Available:")
        if is_folder:
            logger.info(f"  Dataset: {len(dataset_extracted_files)} files extracted")
            # Example: Load CSV for XAI
            # csv_files = [f for f in dataset_extracted_files if f.endswith('.csv')]
            # df = pd.read_csv(csv_files[0])
        else:
            logger.info(f"  Dataset: Single file ({len(dataset_bytes)} bytes)")
            # Example: Load dataset
            # df = pd.read_csv(BytesIO(dataset_bytes))
        
        if is_model_folder:
            logger.info(f"  Model: {len(model_extracted_files)} files extracted")
            # Example: Load model and encoders
            # import pickle
            # model_pkl = [f for f in model_extracted_files if f.endswith('model.pkl')][0]
            # encoder_pkl = [f for f in model_extracted_files if 'encoder' in f.lower()][0]
            # with open(model_pkl, 'rb') as f:
            #     model = pickle.load(f)
            # with open(encoder_pkl, 'rb') as f:
            #     encoders = pickle.load(f)
        else:
            logger.info(f"  Model: Single file ({len(model_bytes)} bytes)")
            # Example: Load single model
            # import pickle
            # model = pickle.loads(model_bytes)
        
        # Use dummy HTML files for testing
        html_file_path_model = f"model-{level}.html"
        html_file_path_data = f"data-{level}.html"
        
        # Upload model explanation report
        if os.path.exists(html_file_path_model):
            try:
                logger.info(f"Uploading model explanation report: {html_file_path_model}")
                result_model = upload_xai_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    model_id=model_id,
                    model_version=model_version,
                    report_type="model_explanation",
                    level=level,
                    html_file_path=html_file_path_model,
                    task_id=task_id
                )
                logger.info(f"✅ Model explanation report uploaded successfully!")
                logger.info(f"   Report type: model_explanation")
                logger.info(f"   Level: {level}")
                logger.info(f"   Response: {json.dumps(result_model, indent=2, default=str)}")
            except Exception as e:
                logger.error(f"Failed to upload model explanation report: {e}", exc_info=True)
        else:
            logger.warning(f"Model HTML file not found: {html_file_path_model}")
            logger.info(f"Skipping model explanation - create {html_file_path_model} in the root directory to test")
        
        # Upload data explanation report
        if os.path.exists(html_file_path_data):
            try:
                logger.info(f"Uploading data explanation report: {html_file_path_data}")
                result_data = upload_xai_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    model_id=model_id,
                    model_version=model_version,
                    report_type="data_explanation",
                    level=level,
                    html_file_path=html_file_path_data,
                    task_id=task_id
                )
                logger.info(f"✅ Data explanation report uploaded successfully!")
                logger.info(f"   Report type: data_explanation")
                logger.info(f"   Level: {level}")
                logger.info(f"   Response: {json.dumps(result_data, indent=2, default=str)}")
                logger.info("   XAI events will be automatically sent by the DW")
            except Exception as e:
                logger.error(f"Failed to upload data explanation report: {e}", exc_info=True)
        else:
            logger.warning(f"Data HTML file not found: {html_file_path_data}")
            logger.info(f"Skipping data explanation - create {html_file_path_data} in the root directory to test")
        
        logger.info(f"XAI processing completed for model {model_id}")
        
    except Exception as e:
        logger.error(f"Error processing XAI trigger event: {e}", exc_info=True)


async def run_consumer() -> None:
    """Main consumer loop"""
    consumer = AIOKafkaConsumer(
        KAFKA_XAI_TRIGGER_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda m: m.decode("utf-8") if m else None,
    )

    logger.info("Starting XAI Trigger consumer...")
    logger.info(f"Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic: {KAFKA_XAI_TRIGGER_TOPIC}")
    logger.info(f"Consumer group: {KAFKA_CONSUMER_GROUP}")
    logger.info("Waiting for XAI trigger events from Agentic Core...")

    await consumer.start()
    
    try:
        async for msg in consumer:
            key = msg.key
            value = msg.value
            
            logger.info("=" * 80)
            logger.info("XAI Trigger Message received")
            logger.info(f"  Partition={msg.partition} Offset={msg.offset}")
            logger.info(f"  Key={key}")
            logger.info(f"  Event={json.dumps(value, indent=2)}")
            logger.info("=" * 80)
            
            # Process the XAI trigger event
            await process_xai_trigger(value)
    
    finally:
        await consumer.stop()
        logger.info("XAI Trigger consumer stopped")


if __name__ == "__main__":
    try:
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
