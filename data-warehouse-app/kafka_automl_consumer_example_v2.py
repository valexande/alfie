#!/usr/bin/env python3
"""
Kafka AutoML Consumer - Version 2 (Real Model with Multiple Files)

This version uploads a real model folder (multiple files) to the Data Warehouse.
Uses the xai-model/model-explanation-endpoint folder which contains:
- model.pkl
- label_encoders.pkl

This tests:
- Uploading model folder (ZIP) from AutoML consumer
- Real-world use case with multiple model files
- Complete Kafka flow with folder upload

Usage:
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  KAFKA_AUTOML_TRIGGER_TOPIC=automl-trigger-events \
  KAFKA_CONSUMER_GROUP=automl-consumer-v2 \
  python kafka_automl_consumer_example_v2.py
"""

import os
import asyncio
import json
import logging
import zipfile
from datetime import datetime
from aiokafka import AIOKafkaConsumer
import requests
import pandas as pd
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("kafka_automl_consumer_v2")

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_AUTOML_TRIGGER_TOPIC = os.getenv("KAFKA_AUTOML_TRIGGER_TOPIC", "automl-trigger-events")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "automl-consumer")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Real model folder path
MODEL_FOLDER_PATH = "xai-model/model-explanation-endpoint"


def fetch_dataset_metadata(user_id: str, dataset_id: str) -> dict:
    """Fetch dataset metadata from the Data Warehouse API"""
    url = f"{API_BASE}/datasets/{user_id}/{dataset_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def read_csv_with_encoding(file_data: bytes) -> pd.DataFrame:
    """Try to read CSV with multiple encodings"""
    from io import BytesIO
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(BytesIO(file_data), encoding=encoding)
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    
    try:
        df = pd.read_csv(BytesIO(file_data), encoding='utf-8', encoding_errors='ignore')
        logger.warning("Read CSV with 'ignore' errors - some characters may be missing")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV with all encodings: {e}")
        raise


def download_dataset_file(user_id: str, dataset_id: str) -> bytes:
    """Download dataset file (single file or folder as ZIP)"""
    url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/download"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def extract_dataset_folder(zip_bytes: bytes, extract_to: str = "temp_dataset") -> list:
    """Extract ZIP file containing dataset folder"""
    import zipfile
    from io import BytesIO
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)
    
    return extracted_files


def create_model_zip(source_folder: str, output_zip: str) -> str:
    """
    Create a ZIP file from model folder
    
    Args:
        source_folder: Path to folder containing model files
        output_zip: Output ZIP file path
        
    Returns:
        Path to created ZIP file
    """
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Model folder not found: {source_folder}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip with relative path
                arcname = os.path.relpath(file_path, source_folder)
                zipf.write(file_path, arcname)
                logger.info(f"Added to ZIP: {arcname}")
    
    logger.info(f"Created ZIP: {output_zip} from {source_folder}")
    return output_zip


def upload_model_folder_to_dw(user_id: str, model_id: str, dataset_id: str, 
                               zip_file_path: str, model_type: str, 
                               framework: str = "sklearn", accuracy: float = None) -> dict:
    """
    Upload model folder (as ZIP) to Data Warehouse
    
    This uses the folder upload endpoint for AI models.
    Version is auto-incremented by the DW.
    """
    url = f"{API_BASE}/ai-models/upload/folder/{user_id}"
    
    with open(zip_file_path, 'rb') as f:
        files = {'zip_file': (os.path.basename(zip_file_path), f, 'application/zip')}
        data = {
            'model_id': model_id,
            'name': f"AutoML Model (Multi-File) - {model_id}",
            'description': f"AutoML trained model for {model_type} with multiple files",
            'framework': framework,
            'model_type': model_type,
            'preserve_structure': 'true',
            'training_dataset': dataset_id,
            'training_accuracy': accuracy,
        }
        
        r = requests.post(url, files=files, data=data, timeout=120)
        r.raise_for_status()
        return r.json()


async def process_automl_trigger(event: dict) -> None:
    """
    Process an AutoML trigger event - V2 with real model folder
    
    This version:
    - Uses real model from xai-model/model-explanation-endpoint
    - Creates ZIP from folder
    - Uploads folder with multiple files
    - Tests folder upload endpoint for AI models
    """
    try:
        user_id = event.get("user_id")
        dataset_id = event.get("dataset_id")
        target_column = event.get("target_column_name")
        task_type = event.get("task_type")
        time_budget = event.get("time_budget", "10")
        
        # Get is_folder from trigger event (not from fetched metadata)
        is_folder = event.get("is_folder", False)
        file_count = event.get("file_count", 1)
        
        if not user_id or not dataset_id:
            logger.warning("Missing user_id or dataset_id in event; skipping")
            return
        
        logger.info(f"Processing AutoML trigger for dataset {dataset_id}")
        logger.info(f"  User: {user_id}")
        logger.info(f"  Target column: {target_column}")
        logger.info(f"  Task type: {task_type}")
        logger.info(f"  Dataset type: {'FOLDER' if is_folder else 'SINGLE FILE'}")
        if is_folder:
            logger.info(f"  File count: {file_count}")
        
        # Step 1: Fetch and process dataset
        try:
            metadata = fetch_dataset_metadata(user_id, dataset_id)
            file_bytes = download_dataset_file(user_id, dataset_id)
            logger.info(f"Dataset downloaded: {len(file_bytes)} bytes")
        except Exception as e:
            logger.error(f"Failed to fetch dataset: {e}")
            return
        
        # Step 2: Load dataset (handle single file or folder)
        df = None
        extracted_files = []
        
        try:
            if is_folder:
                logger.info("Extracting folder dataset...")
                extracted_files = extract_dataset_folder(file_bytes, f"temp_automl_{dataset_id}")
                logger.info(f"Extracted {len(extracted_files)} files")
                
                csv_files = [f for f in extracted_files if f.endswith('.csv')]
                if csv_files:
                    logger.info(f"Found {len(csv_files)} CSV file(s), loading first one")
                    with open(csv_files[0], 'rb') as f:
                        csv_bytes = f.read()
                    df = read_csv_with_encoding(csv_bytes)
                    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                else:
                    logger.error("No CSV files found")
                    return
            else:
                df = read_csv_with_encoding(file_bytes)
                logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to parse dataset: {e}")
            return
        
        # Step 3: Train model (simulation)
        logger.info("=" * 80)
        logger.info("TRAINING MODEL (V2 - Real Model Folder)")
        logger.info(f"  Target: {target_column}")
        logger.info(f"  Task: {task_type}")
        logger.info(f"  Data shape: {df.shape}")
        logger.info("=" * 80)
        
        # Generate model ID
        model_id = f"automl_{dataset_id}_{int(datetime.utcnow().timestamp())}"
        
        # Step 4: Prepare model folder for upload
        model_folder = MODEL_FOLDER_PATH
        
        if not os.path.exists(model_folder):
            logger.error(f"Model folder not found: {model_folder}")
            logger.info("Please ensure the xai-model/model-explanation-endpoint folder exists")
            return
        
        # List model files
        logger.info(f"Using real model from: {model_folder}")
        for root, dirs, files in os.walk(model_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
        
        # Step 5: Create ZIP from model folder
        zip_file_path = f"temp_model_{model_id}.zip"
        try:
            create_model_zip(model_folder, zip_file_path)
            logger.info(f"✅ Created model ZIP: {zip_file_path}")
        except Exception as e:
            logger.error(f"Failed to create model ZIP: {e}")
            return
        
        # Step 6: Upload model folder to DW
        try:
            logger.info(f"Uploading model folder to DW: {model_id}")
            result = upload_model_folder_to_dw(
                user_id=user_id,
                model_id=model_id,
                dataset_id=dataset_id,
                zip_file_path=zip_file_path,
                model_type=task_type,
                framework="sklearn",
                accuracy=0.95  # Simulated accuracy
            )
            
            logger.info("=" * 80)
            logger.info("✅ MODEL FOLDER UPLOADED TO DW!")
            logger.info(f"   Model ID: {model_id}")
            logger.info(f"   Version: {result.get('version')}")
            logger.info(f"   Files: {len(result.get('files', []))}")
            logger.info(f"   Total size: {result.get('model_size_mb', 0):.2f} MB")
            logger.info("=" * 80)
            logger.info("AutoML event will be automatically sent by the DW")
            
        except Exception as e:
            logger.error(f"Failed to upload model folder to DW: {e}", exc_info=True)
            return
        finally:
            # Cleanup ZIP file
            try:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)
                    logger.info(f"Cleaned up temp ZIP: {zip_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp ZIP: {e}")
        
        logger.info(f"AutoML V2 processing completed for dataset {dataset_id}")
        
    except Exception as e:
        logger.error(f"Error processing AutoML trigger event: {e}", exc_info=True)


async def run_consumer() -> None:
    """Main consumer loop"""
    consumer = AIOKafkaConsumer(
        KAFKA_AUTOML_TRIGGER_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda m: m.decode("utf-8") if m else None,
    )

    logger.info("=" * 80)
    logger.info("Starting AutoML Trigger consumer V2 (Real Model Folder)")
    logger.info("=" * 80)
    logger.info(f"Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic: {KAFKA_AUTOML_TRIGGER_TOPIC}")
    logger.info(f"Consumer group: {KAFKA_CONSUMER_GROUP}")
    logger.info(f"Model folder: {MODEL_FOLDER_PATH}")
    logger.info("Waiting for AutoML trigger events from Agentic Core...")

    await consumer.start()
    
    try:
        async for msg in consumer:
            key = msg.key
            value = msg.value
            
            logger.info("=" * 80)
            logger.info("AutoML Trigger Message received (V2)")
            logger.info(f"  Partition={msg.partition} Offset={msg.offset}")
            logger.info(f"  Key={key}")
            logger.info(f"  Event={json.dumps(value, indent=2)}")
            logger.info("=" * 80)
            
            # Process the AutoML trigger event
            await process_automl_trigger(value)
    
    finally:
        await consumer.stop()
        logger.info("AutoML Trigger consumer V2 stopped")


if __name__ == "__main__":
    try:
        # Verify model folder exists
        if not os.path.exists(MODEL_FOLDER_PATH):
            print(f"❌ ERROR: Model folder not found: {MODEL_FOLDER_PATH}")
            print(f"   Please ensure the folder exists with model files.")
            exit(1)
        
        # List model files
        print("\n" + "=" * 80)
        print("📦 Model Files to Upload:")
        print("=" * 80)
        for root, dirs, files in os.walk(MODEL_FOLDER_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"  ✅ {file} ({file_size:,} bytes)")
        print("=" * 80)
        print()
        
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

