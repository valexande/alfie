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
from typing import Optional
from aiokafka import AIOKafkaConsumer
import requests
import pandas as pd
from io import BytesIO
import tempfile
import zipfile
import shap
import joblib
import matplotlib

matplotlib.use('Agg')  # Must come BEFORE pyplot
import matplotlib.pyplot as plt
import base64
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("kafka_xai_consumer")

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_XAI_TRIGGER_TOPIC = os.getenv("KAFKA_XAI_TRIGGER_TOPIC", "xai-trigger-events")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "xai-consumer")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Flask service endpoints for XAI analysis
UNIVERSAL_MODEL_EXPLAINABILITY_URL = os.getenv("UNIVERSAL_MODEL_EXPLAINABILITY_URL", "http://localhost:5010/explain-model")
FLEXIBLE_DATA_INTERPRETABILITY_URL = os.getenv("FLEXIBLE_DATA_INTERPRETABILITY_URL", "http://localhost:5001/analyze-data")


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
                     task_id: Optional[str] = None) -> dict:
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

        logger.info(f"Uploading to: {url}")
        logger.info(f"Files: {list(files.keys())}")
        logger.info(f"Data: {data}")
        
        headers = {"X-Task-ID": task_id} if task_id else None
        r = requests.post(url, files=files, data=data, headers=headers, timeout=120)
        
        if r.status_code != 200:
            logger.error(f"Upload failed with status {r.status_code}")
            logger.error(f"Response: {r.text}")
            logger.error(f"Request URL: {url}")
            logger.error(f"Request data: {data}")
            logger.error(f"Request files: {list(files.keys())}")
        
        r.raise_for_status()
        return r.json()


def plot_to_base64(plot_func):
    """Convert matplotlib plot to base64 string"""
    buf = BytesIO()
    plot_func()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_alert_counts(df):
    """Plot alert counts by demographic categories"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Number of Alerts per Demographic Category', fontsize=16)

    df[df['alert'] == 1]['age'].value_counts().sort_index().plot(kind='bar', ax=axs[0, 0])
    axs[0, 0].set_title('Alerts per Age')
    axs[0, 0].set_xlabel('Age')
    axs[0, 0].set_ylabel('Number of Alerts')

    df[df['alert'] == 1]['race_label'].value_counts().plot(kind='bar', ax=axs[0, 1])
    axs[0, 1].set_title('Alerts per Race')
    axs[0, 1].set_xlabel('Race')
    axs[0, 1].set_ylabel('Number of Alerts')

    df[df['alert'] == 1]['gender_label'].value_counts().plot(kind='bar', ax=axs[1, 0])
    axs[1, 0].set_title('Alerts per Gender')
    axs[1, 0].set_xlabel('Gender')
    axs[1, 0].set_ylabel('Number of Alerts')

    df[df['alert'] == 1]['ethnicity_label'].value_counts().plot(kind='bar', ax=axs[1, 1])
    axs[1, 1].set_title('Alerts per Ethnicity')
    axs[1, 1].set_xlabel('Ethnicity')
    axs[1, 1].set_ylabel('Number of Alerts')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_group_metrics(df, predictions, y_true, group_col, label_encoder):
    """Compute fairness metrics for different groups"""
    results = {}
    df_copy = df.copy()
    df_copy['prediction'] = predictions
    df_copy['true'] = y_true.values
    for group in df_copy[group_col].unique():
        subset = df_copy[df_copy[group_col] == group]
        acc = accuracy_score(subset['true'], subset['prediction'])
        prec = precision_score(subset['true'], subset['prediction'], zero_division=0)
        rec = recall_score(subset['true'], subset['prediction'], zero_division=0)
        label = label_encoder.inverse_transform([group])[0]
        results[label] = {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2)
        }
    return results


def plot_time_series(df):
    """Plot heart rate time series with alerts"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='frame_timestamp', y='heart_rate', label='Heart Rate')
    sns.scatterplot(data=df[df['alert']], x='frame_timestamp', y='heart_rate', color='red', label='Alert')
    plt.xticks(rotation=45)
    plt.title('Heart Rate over Time with Alerts')
    plt.tight_layout()


def plot_clusters(df):
    """Plot driver state clustering"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='heart_rate', y='alert', hue='Cluster', palette='viridis')
    plt.title('Driver State Clustering')
    plt.tight_layout()


def find_files_in_dir(directory, extensions):
    """Find files with specific extensions in directory"""
    found_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    if ext not in found_files:
                        found_files[ext] = []
                    found_files[ext].append(os.path.join(root, file))
    return found_files


def call_model_explanation_endpoint(data_dir, model_dir, user_level='expert'):
    """Call the universal model explainability Flask service"""
    logger.info("Calling universal model explainability endpoint...")
    
    # Find files
    data_files = find_files_in_dir(data_dir, ['.csv'])
    model_files = find_files_in_dir(model_dir, ['.pkl'])

    # Load main data file - prioritize files with target columns (alert, target, label)
    data_file = None
    target_files = []

    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    logger.info(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        df_test = pd.read_csv(csv_file)
        logger.info(f"Checking file {csv_file} with columns: {list(df_test.columns)}")
        # Check for common target column names
        if any(col in df_test.columns for col in ['alert', 'target', 'label']):
            target_files.append((csv_file, len(df_test)))
            logger.info(f"Found target file: {csv_file} with {len(df_test)} rows")

    if target_files:
        # Choose the largest file with target column
        data_file = max(target_files, key=lambda x: x[1])[0]
        logger.info(f"Using target file: {data_file}")
    else:
        data_file = csv_files[0] if csv_files else None
        if not data_file:
            raise FileNotFoundError("No CSV file found in data directory")
        logger.warning(f"No target column found, using first CSV: {data_file}")

    # Find model file
    pkl_files = model_files.get('.pkl', [])
    if not pkl_files:
        raise FileNotFoundError("No .pkl file found in model directory")

    # Find the model file (any .pkl file that can be loaded)
    # Note: We don't strictly validate here since the Flask service will handle loading
    # Some models may have pickle protocol issues that the service can handle differently
    model_file = None

    for pkl_file in pkl_files:
        try:
            loaded_obj = joblib.load(pkl_file)
            # Check if it's a model (has predict method) or can be used as model
            if hasattr(loaded_obj, 'predict') or hasattr(loaded_obj, 'fit'):
                model_file = pkl_file
                logger.info(f"Found model file: {pkl_file} (validated locally)")
                break
        except Exception as e:
            logger.debug(f"Could not validate {pkl_file} locally: {e}")
            # Continue to next file - the service might be able to load it
            continue

    if not model_file:
        # If no model found with predict/fit, use the first .pkl file
        # The Flask service will attempt to load it and handle errors appropriately
        model_file = pkl_files[0]
        logger.info(f"Using model file: {model_file} (will be validated by service)")

    # Call universal model explainability Flask endpoint
    flask_url = UNIVERSAL_MODEL_EXPLAINABILITY_URL
    
    with open(data_file, 'rb') as df, open(model_file, 'rb') as mf:
        files = {
            'data_file': df,
            'model_file': mf
        }
        data = {'user_level': user_level}
        
        logger.info(f"Calling universal model explainability endpoint: {flask_url}")
        logger.info(f"  Data file: {data_file}")
        logger.info(f"  Model file: {model_file}")
        logger.info(f"  User level: {user_level}")
        
        try:
            response = requests.post(flask_url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                logger.info("Universal model explainability endpoint called successfully")
                return response.text
            else:
                logger.error(f"Universal model explainability endpoint failed: {response.status_code}")
                logger.error(f"Response: {response.text[:1000]}")
                
                # Provide helpful error message based on status code
                if response.status_code == 400:
                    error_msg = (
                        f"Model or data loading error. The service could not process the files.\n"
                        f"Model file: {model_file}\n"
                        f"Data file: {data_file}\n"
                        f"Service response: {response.text[:500]}"
                    )
                    raise Exception(error_msg)
                else:
                    raise Exception(f"Flask endpoint error: {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"Could not connect to universal model explainability service at {flask_url}.\n"
                f"Please ensure the service is running. You can start it with:\n"
                f"  python flexible-scripts/universal_model_explainability.py"
            )
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to universal model explainability service failed: {e}")
            raise Exception(f"Request failed: {e}") from e


def call_driver_data_endpoint(data_dir, user_level='expert'):
    """Call the flexible data interpretability Flask service"""
    logger.info("Calling flexible data interpretability endpoint...")

    # Find CSV files
    data_files = find_files_in_dir(data_dir, ['.csv'])

    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    logger.info(f"Found {len(csv_files)} CSV files for data analysis")

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")

    # For flexible data interpretability, we can use any CSV file
    # Prefer the largest file or one with more columns
    data_file = None
    if len(csv_files) == 1:
        data_file = csv_files[0]
    else:
        # Choose the file with the most columns/rows
        file_scores = []
        for csv_file in csv_files:
            try:
                df_test = pd.read_csv(csv_file)
                score = len(df_test.columns) * len(df_test)  # Simple scoring
                file_scores.append((csv_file, score, len(df_test)))
            except Exception as e:
                logger.warning(f"Could not read {csv_file}: {e}")
                continue
        
        if file_scores:
            data_file = max(file_scores, key=lambda x: x[1])[0]
            logger.info(f"Selected data file: {data_file} with {max(file_scores, key=lambda x: x[1])[2]} rows")
        else:
            data_file = csv_files[0]
            logger.warning(f"Could not score files, using first CSV: {data_file}")

    logger.info(f"Using data file: {data_file}")

    # Call flexible data interpretability Flask endpoint
    flask_url = FLEXIBLE_DATA_INTERPRETABILITY_URL
    
    with open(data_file, 'rb') as cf:
        files = {
            'csv_file': cf
        }
        data = {'user_level': user_level}
        
        logger.info(f"Calling flexible data interpretability endpoint: {flask_url}")
        logger.info(f"  Data file: {data_file}")
        logger.info(f"  User level: {user_level}")
        
        try:
            response = requests.post(flask_url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                logger.info("Flexible data interpretability endpoint called successfully")
                return response.text
            else:
                logger.error(f"Flexible data interpretability endpoint failed: {response.status_code}")
                logger.error(f"Response: {response.text[:1000]}")
                
                # Provide helpful error message based on status code
                if response.status_code == 400:
                    error_msg = (
                        f"Data loading error. The service could not process the file.\n"
                        f"Data file: {data_file}\n"
                        f"Service response: {response.text[:500]}"
                    )
                    raise Exception(error_msg)
                else:
                    raise Exception(f"Flask endpoint error: {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"Could not connect to flexible data interpretability service at {flask_url}.\n"
                f"Please ensure the service is running. You can start it with:\n"
                f"  python flexible-scripts/flexible-data-interpretability.py"
            )
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to flexible data interpretability service failed: {e}")
            raise Exception(f"Request failed: {e}") from e


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
        is_folder = event.get("is_folder", False)
        file_count = event.get("file_count", 1)
        is_model_folder = event.get("is_model_folder", False)
        model_file_count = event.get("model_file_count", 1)

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
        logger.info("=" * 80)
        logger.info("Generating XAI explanations")
        logger.info(f"  - Model: {model_meta.get('name')}")
        logger.info(f"  - Framework: {model_meta.get('framework')}")
        logger.info(f"  - Level: {level}")
        logger.info("=" * 80)

        # Create temporary directories for extracted files
        with tempfile.TemporaryDirectory() as temp_analysis_dir:
            # Set up directories for analysis
            data_analysis_dir = os.path.join(temp_analysis_dir, 'data')
            model_analysis_dir = os.path.join(temp_analysis_dir, 'model')
            os.makedirs(data_analysis_dir, exist_ok=True)
            os.makedirs(model_analysis_dir, exist_ok=True)

            # Copy extracted files to analysis directories
            if is_folder:
                logger.info(f"Copying {len(dataset_extracted_files)} dataset files to analysis directory")
                for file_path in dataset_extracted_files:
                    filename = os.path.basename(file_path)
                    import shutil
                    shutil.copy2(file_path, os.path.join(data_analysis_dir, filename))
            else:
                # Handle single dataset file
                logger.info("Processing single dataset file")
                df = read_csv_with_encoding(dataset_bytes)
                csv_path = os.path.join(data_analysis_dir, 'dataset.csv')
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved single dataset to {csv_path}")

            if is_model_folder:
                logger.info(f"Copying {len(model_extracted_files)} model files to analysis directory")
                for file_path in model_extracted_files:
                    filename = os.path.basename(file_path)
                    import shutil
                    shutil.copy2(file_path, os.path.join(model_analysis_dir, filename))
            else:
                # Handle single model file
                logger.info("Processing single model file")
                model_path = os.path.join(model_analysis_dir, 'model.pkl')
                with open(model_path, 'wb') as f:
                    f.write(model_bytes)
                logger.info(f"Saved single model to {model_path}")

            # Generate XAI reports using Flask endpoints
            # Initialize report paths as None - will be set when reports are successfully generated
            html_file_path_model = None
            html_file_path_data = None
            html_file_path_combined = None
            model_report_html = None
            driver_report_html = None
            
            # Generate model explanation report using Flask endpoint
            try:
                logger.info("Generating model explanation report using Flask endpoint...")
                model_report_html = call_model_explanation_endpoint(data_analysis_dir, model_analysis_dir, user_level=level)

                # Save model report
                # Use reports directory if available (Docker), otherwise current directory
                reports_dir = os.getenv("XAI_REPORTS_DIR", ".")
                os.makedirs(reports_dir, exist_ok=True)
                model_report_path = os.path.join(reports_dir, f"model_explanation_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(model_report_path, 'w', encoding='utf-8') as f:
                    f.write(model_report_html)
                logger.info(f"Model explanation report saved: {model_report_path}")
                html_file_path_model = model_report_path
            except Exception as e:
                logger.error(f"Error generating model explanation report: {e}", exc_info=True)
                logger.warning("Model explanation report generation failed, continuing with data analysis...")

            # Generate driver data analysis report using Flask endpoint
            try:
                logger.info("Generating driver data analysis report using Flask endpoint...")
                driver_report_html = call_driver_data_endpoint(data_analysis_dir, user_level=level)

                # Save driver report
                reports_dir = os.getenv("XAI_REPORTS_DIR", ".")
                os.makedirs(reports_dir, exist_ok=True)
                driver_report_path = os.path.join(reports_dir, f"driver_analysis_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(driver_report_path, 'w', encoding='utf-8') as f:
                    f.write(driver_report_html)
                logger.info(f"Driver analysis report saved: {driver_report_path}")
                html_file_path_data = driver_report_path
            except Exception as e:
                logger.error(f"Error generating driver data analysis report: {e}", exc_info=True)
                logger.warning("Driver data analysis report generation failed, continuing...")

            # Create combined report if we have at least one report
            if model_report_html or driver_report_html:
                try:
                    logger.info("Creating combined XAI report...")

                    # Extract the body content from reports (if available)
                    model_body = ""
                    if model_report_html:
                        model_body_match = re.search(r'<body>(.*?)</body>', model_report_html, re.DOTALL)
                        model_body = model_body_match.group(1) if model_body_match else model_report_html
                    else:
                        model_body = "<p><em>Model explanation report could not be generated.</em></p>"

                    driver_body = ""
                    if driver_report_html:
                        driver_body_match = re.search(r'<body>(.*?)</body>', driver_report_html, re.DOTALL)
                        driver_body = driver_body_match.group(1) if driver_body_match else driver_report_html
                    else:
                        driver_body = "<p><em>Driver data analysis report could not be generated.</em></p>"

                    # Create combined HTML report
                    combined_html = f"""
                    <html>
                    <head>
                        <title>Combined XAI Analysis Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            img {{ max-width: 100%; height: auto; }}
                            .table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
                            .table td, .table th {{ border: 1px solid #ddd; padding: 8px; }}
                            .table th {{ background-color: #f2f2f2; }}
                            .section {{ margin-bottom: 60px; border-bottom: 2px solid #333; padding-bottom: 30px; }}
                            .section h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                            .section h2 {{ color: #34495e; margin-top: 30px; }}
                        </style>
                    </head>
                    <body>
                        <div class="section">
                            <h1>XAI Analysis Report</h1>
                            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            <p><strong>User:</strong> {user_id}</p>
                            <p><strong>Dataset:</strong> {dataset_id}</p>
                            <p><strong>Model:</strong> {model_id}</p>
                            <p><strong>Level:</strong> {level}</p>
                            <p>This comprehensive report combines both model explanation and driver data analysis to provide a complete understanding of AI model performance and driver behavior patterns.</p>
                        </div>

                        <div class="section">
                            <h1>Model Explanation Analysis</h1>
                            {model_body}
                        </div>

                        <div class="section">
                            <h1>Driver Data Analysis</h1>
                            {driver_body}
                        </div>

                        <div class="section">
                            <h1>Summary</h1>
                            <p>This combined analysis provides both technical model insights and practical driver behavior patterns. The model explanation section shows how the AI makes decisions, while the driver analysis reveals real-world patterns in driver alertness.</p>
                            <ul>
                                <li><strong>Model Performance:</strong> Detailed classification metrics and SHAP analysis</li>
                                <li><strong>Feature Importance:</strong> Which factors most influence alertness predictions</li>
                                <li><strong>Fairness Analysis:</strong> How the model performs across different demographic groups</li>
                                <li><strong>Driver Patterns:</strong> Real-time analysis of heart rate, yawning, and eye closure</li>
                                <li><strong>Anomaly Detection:</strong> Identification of unusual driver behavior patterns</li>
                                <li><strong>Clustering Analysis:</strong> Classification of different driver states</li>
                            </ul>
                        </div>
                    </body>
                    </html>
                    """

                    # Save combined report
                    reports_dir = os.getenv("XAI_REPORTS_DIR", ".")
                    os.makedirs(reports_dir, exist_ok=True)
                    combined_report_path = os.path.join(reports_dir, f"combined_xai_report_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                    with open(combined_report_path, 'w', encoding='utf-8') as f:
                        f.write(combined_html)
                    logger.info(f"Combined XAI report saved: {combined_report_path}")
                    html_file_path_combined = combined_report_path
                except Exception as e:
                    logger.error(f"Error creating combined report: {e}", exc_info=True)
                    logger.warning("Combined report creation failed, but individual reports will still be uploaded if available")

        # Upload model explanation report
        if html_file_path_model and os.path.exists(html_file_path_model):
            try:
                logger.info(f"Uploading model explanation report: {html_file_path_model}")
                logger.info(f"File size: {os.path.getsize(html_file_path_model)} bytes")
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
        if html_file_path_data and os.path.exists(html_file_path_data):
            try:
                logger.info(f"Uploading data explanation report: {html_file_path_data}")
                logger.info(f"File size: {os.path.getsize(html_file_path_data)} bytes")
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

        # Upload combined report as model_explanation (since API only accepts model_explanation or data_explanation)
        if html_file_path_combined and os.path.exists(html_file_path_combined):
            try:
                logger.info(f"Uploading combined XAI report as model_explanation: {html_file_path_combined}")
                logger.info(f"File size: {os.path.getsize(html_file_path_combined)} bytes")
                result_combined = upload_xai_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    model_id=model_id,
                    model_version=model_version,
                    report_type="model_explanation",  # Use valid report type
                    level=level,
                    html_file_path=html_file_path_combined,
                    task_id=task_id
                )
                logger.info(f"✅ Combined XAI report uploaded successfully!")
                logger.info(f"   Report type: model_explanation (combined report)")
                logger.info(f"   Level: {level}")
                logger.info(f"   Response: {json.dumps(result_combined, indent=2, default=str)}")
                logger.info("   XAI events will be automatically sent by the DW")
            except Exception as e:
                logger.error(f"Failed to upload combined XAI report: {e}", exc_info=True)
        else:
            logger.warning(f"Combined HTML file not found: {html_file_path_combined}")
            logger.info(f"Skipping combined report - create {html_file_path_combined} in the root directory to test")

        # Summary of uploaded reports
        uploaded_reports = []
        if html_file_path_model and os.path.exists(html_file_path_model):
            uploaded_reports.append("model_explanation")
        if html_file_path_data and os.path.exists(html_file_path_data):
            uploaded_reports.append("data_explanation")
        if html_file_path_combined and os.path.exists(html_file_path_combined):
            uploaded_reports.append("combined_report")
        
        if uploaded_reports:
            logger.info(f"✅ XAI processing completed for model {model_id}")
            logger.info(f"   Successfully uploaded reports: {', '.join(uploaded_reports)}")
        else:
            logger.warning(f"⚠️ XAI processing completed for model {model_id} but no reports were uploaded")
            logger.warning("   This may indicate that both model and data analysis services failed")

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
