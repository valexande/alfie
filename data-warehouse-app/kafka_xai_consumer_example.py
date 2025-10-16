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


def fetch_dataset_metadata(user_id: str, dataset_id: str) -> dict:
    """Fetch dataset metadata from the Data Warehouse API"""
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


def download_dataset_file(user_id: str, dataset_id: str) -> bytes:
    """Download dataset file (single file or folder as ZIP)"""
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


def upload_xai_report(user_id: str, dataset_id: str, model_id: str, 
                     report_type: str, level: str, html_file_path: str) -> dict:
    """
    Upload XAI report to Data Warehouse
    This will automatically trigger an xai-events message
    """
    url = f"{API_BASE}/xai-reports/upload/{user_id}"
    
    with open(html_file_path, 'rb') as f:
        files = {'file': (os.path.basename(html_file_path), f, 'text/html')}
        data = {
            'dataset_id': dataset_id,
            'model_id': model_id,
            'report_type': report_type,
            'level': level
        }
        
        r = requests.post(url, files=files, data=data, timeout=120)
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


def analyze_model_explanation(data_dir, model_dir, user_level='expert'):
    """Analyze model explanation with downloaded data"""
    logger.info("Starting model explanation analysis...")
    
    # Find files
    data_files = find_files_in_dir(data_dir, ['.csv'])
    model_files = find_files_in_dir(model_dir, ['.pkl'])
    
    # Load main data file - prioritize files with 'alert' column
    data_file = None
    alert_files = []
    
    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        df_test = pd.read_csv(csv_file)
        logger.info(f"Checking file {csv_file} with columns: {list(df_test.columns)}")
        if 'alert' in df_test.columns:
            alert_files.append((csv_file, len(df_test)))
            logger.info(f"Found alert file: {csv_file} with {len(df_test)} rows")
    
    if alert_files:
        # Choose the largest file with alert column
        data_file = max(alert_files, key=lambda x: x[1])[0]
        logger.info(f"Using alert file: {data_file}")
    else:
        data_file = csv_files[0] if csv_files else None
        logger.warning(f"No alert column found, using first CSV: {data_file}")
    
    # Load model and encoder
    pkl_files = model_files.get('.pkl', [])
    if not pkl_files:
        raise FileNotFoundError("No .pkl file found in model directory")
    
    # Find the correct files
    model = None
    label_encoders = None
    
    for pkl_file in pkl_files:
        try:
            loaded_obj = joblib.load(pkl_file)
            if hasattr(loaded_obj, 'predict'):
                model = loaded_obj
                logger.info(f"Found model file: {pkl_file}")
            elif isinstance(loaded_obj, dict) and 'gender' in loaded_obj:
                label_encoders = loaded_obj
                logger.info(f"Found encoder file: {pkl_file}")
        except:
            continue
    
    if not model:
        raise FileNotFoundError("No model file found in model directory")
    if not label_encoders:
        raise FileNotFoundError("No encoder file found in model directory")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Data columns: {list(df.columns)}")
    
    # Preprocess
    categorical_cols = ['gender', 'ethnicity', 'race']
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Check if alert column exists, if not create a dummy one
    if 'alert' not in df.columns:
        logger.warning("No 'alert' column found, creating dummy alert column")
        df['alert'] = 0
    
    X = df.drop(columns=['alert'])
    y = df['alert']
    
    # Predict with sklearn version compatibility fix
    try:
        predictions = model.predict(X)
        report_dict = classification_report(y, predictions, output_dict=True)
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            logger.info("Fixing sklearn version compatibility issue...")
            # Add missing attribute to fix compatibility
            for estimator in model.estimators_:
                if not hasattr(estimator, 'monotonic_cst'):
                    estimator.monotonic_cst = None
            # Try again
            predictions = model.predict(X)
            report_dict = classification_report(y, predictions, output_dict=True)
        else:
            raise e
    
    report_html = pd.DataFrame(report_dict).T.to_html(classes="table table-bordered")
    
    # SHAP explainability
    # Use sample for SHAP to avoid memory issues with large datasets
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Use the correct shap_values based on array dimensions
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1]  # Binary classification, use positive class
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]  # 3D array, get positive class (class 1)
    else:
        shap_vals = shap_values  # Single output
    
    shap_bar_b64 = plot_to_base64(lambda: shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False))
    shap_full_b64 = plot_to_base64(lambda: shap.summary_plot(shap_vals, X_sample, show=False))
    
    # Reverse-transform encoded labels
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            df[col + '_label'] = label_encoders[col].inverse_transform(df[col])
    
    # Alert breakdown plot
    alert_plot_b64 = plot_alert_counts(df)
    
    # Fairness metrics
    fairness_html_blocks = []
    for group in categorical_cols:
        if group in df.columns and group in label_encoders:
            fairness_df = pd.DataFrame.from_dict(
                compute_group_metrics(df, predictions, y, group, label_encoders[group]),
                orient='index'
            )
            fairness_html_blocks.append(
                f"<h3>Fairness for {group.capitalize()}</h3>" + fairness_df.to_html(classes="table table-striped"))
    
    # Generate HTML report
    if user_level == "beginner":
        beginner_html = f"""
        <html>
        <head><title>Model Summary for Beginners</title></head>
        <body>
        <h1>Model Summary</h1>
        <p>This model helps predict if a driver is alert or not using things like age, gender, and race.</p>
        <ul>
            <li>We looked at how the model behaves for different groups of people.</li>
            <li>We used special charts to see which features the model uses most.</li>
            <li>We checked that the model treats people fairly, no matter who they are.</li>
        </ul>
        <h3>Important Factors the Model Uses</h3>
        <img src="data:image/png;base64,{shap_bar_b64}" />

        <h3>How Alerts Are Spread Among Groups</h3>
        <img src="data:image/png;base64,{alert_plot_b64}" />
        </body>
        </html>
        """
        return beginner_html
    else:
        # Expert Report
        html_template = f"""
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                img {{ max-width: 100%; height: auto; }}
                .table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
                .table td, .table th {{ border: 1px solid #ddd; padding: 8px; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Model Explanation Report</h1>
            <h2>Classification Report</h2>
            {report_html}

            <h2>SHAP Summary Bar Plot</h2>
            <img src="data:image/png;base64,{shap_bar_b64}" />

            <h2>SHAP Full Summary Plot</h2>
            <img src="data:image/png;base64,{shap_full_b64}" />

            <h2>Alert Distribution by Demographics</h2>
            <img src="data:image/png;base64,{alert_plot_b64}" />

            <h2>Fairness Metrics</h2>
            {''.join(fairness_html_blocks)}
        </body>
        </html>
        """
        return html_template


def analyze_driver_data(data_dir, user_level='expert'):
    """Analyze driver data with downloaded files"""
    logger.info("Starting driver analysis...")
    
    # Find CSV files
    data_files = find_files_in_dir(data_dir, ['.csv'])
    
    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    logger.info(f"Found {len(csv_files)} CSV files for driver analysis")
    
    # Look for specific files
    frame_file = None
    hr_file = None
    
    for csv_file in csv_files:
        df_test = pd.read_csv(csv_file)
        logger.info(f"Checking driver file {csv_file} with columns: {list(df_test.columns)}")
        if 'frame_timestamp' in df_test.columns:
            frame_file = csv_file
            logger.info(f"Found frame file: {csv_file}")
        elif 'heart_rate' in df_test.columns or 'timestamp' in df_test.columns:
            hr_file = csv_file
            logger.info(f"Found heart rate file: {csv_file}")
    
    if not frame_file or not hr_file:
        # Use any available CSV files
        if len(csv_files) >= 2:
            frame_file = csv_files[0]
            hr_file = csv_files[1]
        else:
            raise FileNotFoundError("Need at least 2 CSV files for driver analysis")
    
    logger.info(f"Using frame file: {frame_file}")
    logger.info(f"Using heart rate file: {hr_file}")
    
    frame_df = pd.read_csv(frame_file)
    heart_rate_df = pd.read_csv(hr_file)
    
    # Convert timestamps
    frame_df['frame_timestamp'] = pd.to_datetime(frame_df['frame_timestamp'])
    heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])
    
    # Merge
    merged_df = pd.merge_asof(frame_df.sort_values('frame_timestamp'),
                              heart_rate_df.sort_values('timestamp'),
                              left_on='frame_timestamp',
                              right_on='timestamp',
                              direction='nearest')
    
    # Analysis
    correlation_results = merged_df[['heart_rate', 'eyes_closed', 'yawning', 'alert']].corr().round(2)
    corr_html = correlation_results.to_html(classes="table table-hover")
    
    merged_df['Heart Rate Z-Score'] = zscore(merged_df['heart_rate'])
    anomalies = merged_df[(merged_df['Heart Rate Z-Score'].abs() > 2) |
                          ((merged_df['yawning'] | merged_df['eyes_closed']) & ~merged_df['alert'])]
    anomalies_html = anomalies[['frame_timestamp', 'heart_rate', 'eyes_closed', 'yawning', 'alert']].head(20).to_html(classes="table table-bordered")
    
    ts_plot_b64 = plot_to_base64(lambda: plot_time_series(merged_df))
    features = merged_df[['heart_rate', 'eyes_closed', 'yawning', 'alert']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto').fit(features)
    merged_df['Cluster'] = kmeans.labels_
    cluster_plot_b64 = plot_to_base64(lambda: plot_clusters(merged_df))
    
    if user_level == "beginner":
        summary_html = f"""
        <html>
        <head><title>Driver Summary</title></head>
        <body>
        <h1>Driver Summary for Beginners</h1>
        <p>This report shows how heart rate and signs like yawning or closed eyes help tell if a driver is alert or sleepy.</p>
        <ul>
          <li>We found strange heart patterns that might mean tiredness.</li>
          <li>We also found some times when the driver looked sleepy but wasn't marked as alert.</li>
        </ul>
        <h3>Heart Rate Over Time</h3>
        <img src="data:image/png;base64,{ts_plot_b64}" />
        <h3>Driver Types (Clusters)</h3>
        <img src="data:image/png;base64,{cluster_plot_b64}" />
        </body></html>
        """
        return summary_html
    else:
        # Full expert report
        html_template = f"""
        <html>
        <head><title>Driver Analysis Report</title></head>
        <body>
            <h1>Driver Analysis Report</h1>
            <h2>Correlation Matrix</h2>{corr_html}
            <h2>Detected Anomalies (First 20 rows)</h2>{anomalies_html}
            <h2>Heart Rate Time Series with Alerts</h2><img src="data:image/png;base64,{ts_plot_b64}" />
            <h2>Driver State Clustering</h2><img src="data:image/png;base64,{cluster_plot_b64}" />
        </body></html>
        """
        return html_template


async def process_xai_trigger(event: dict) -> None:
    """
    Process an XAI trigger event from Agentic Core
    
    Event structure:
    {
        "event_type": "xai-trigger.reported",
        "user_id": "user123",
        "dataset_id": "dataset123",
        "model_id": "model123",
        "version": "v1",
        "level": "beginner",
        "timestamp": "2025-10-10T12:00:00.000000"
    }
    """
    try:
        user_id = event.get("user_id")
        dataset_id = event.get("dataset_id")
        model_id = event.get("model_id")
        version = event.get("version", "v1")
        level = event.get("level", "beginner")
        
        # Get folder info from trigger event
        is_folder = event.get("is_folder", False)
        file_count = event.get("file_count", 1)
        is_model_folder = event.get("is_model_folder", False)
        model_file_count = event.get("model_file_count", 1)
        
        if not user_id or not dataset_id or not model_id:
            logger.warning("Missing required fields in event; skipping")
            return
        
        logger.info(f"Processing XAI trigger for model {model_id}")
        logger.info(f"  User: {user_id}")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Level: {level}")
        logger.info(f"  Dataset type: {'FOLDER' if is_folder else 'SINGLE FILE'} ({file_count} file(s))")
        logger.info(f"  Model type: {'FOLDER' if is_model_folder else 'SINGLE FILE'} ({model_file_count} file(s))")
        
        # Step 1: Fetch and download dataset
        try:
            dataset_meta = fetch_dataset_metadata(user_id, dataset_id)
            logger.info(f"Dataset metadata fetched: {dataset_meta.get('name')}")
            
            # Download dataset (single file or ZIP)
            dataset_bytes = download_dataset_file(user_id, dataset_id)
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
            model_meta = fetch_model_metadata(user_id, model_id, version)
            logger.info(f"Model metadata fetched: {model_meta.get('name')}")
            logger.info(f"  Framework: {model_meta.get('framework')}")
            logger.info(f"  Files: {len(model_meta.get('files', []))}")
            
            # Download model (single file or folder as ZIP)
            model_bytes = download_model_file(user_id, model_id, version)
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
            
            # Generate XAI reports
            try:
                # Generate model explanation report
                logger.info("Generating model explanation report...")
                model_report_html = analyze_model_explanation(data_analysis_dir, model_analysis_dir, user_level=level)
                
                # Save model report
                model_report_path = f"model_explanation_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(model_report_path, 'w', encoding='utf-8') as f:
                    f.write(model_report_html)
                logger.info(f"Model explanation report saved: {model_report_path}")
                
                # Generate driver data analysis report
                logger.info("Generating driver data analysis report...")
                driver_report_html = analyze_driver_data(data_analysis_dir, user_level=level)
                
                # Save driver report
                driver_report_path = f"driver_analysis_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(driver_report_path, 'w', encoding='utf-8') as f:
                    f.write(driver_report_html)
                logger.info(f"Driver analysis report saved: {driver_report_path}")
                
                # Create combined report
                logger.info("Creating combined XAI report...")
                
                # Extract the body content from both reports
                model_body_match = re.search(r'<body>(.*?)</body>', model_report_html, re.DOTALL)
                model_body = model_body_match.group(1) if model_body_match else model_report_html
                
                driver_body_match = re.search(r'<body>(.*?)</body>', driver_report_html, re.DOTALL)
                driver_body = driver_body_match.group(1) if driver_body_match else driver_report_html
                
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
                combined_report_path = f"combined_xai_report_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(combined_report_path, 'w', encoding='utf-8') as f:
                    f.write(combined_html)
                logger.info(f"Combined XAI report saved: {combined_report_path}")
                
                # Set the report paths for upload
                html_file_path_model = model_report_path
                html_file_path_data = driver_report_path
                html_file_path_combined = combined_report_path
                
            except Exception as e:
                logger.error(f"Error generating XAI reports: {e}", exc_info=True)
                # Fallback to dummy files if XAI generation fails
                html_file_path_model = f"model-{level}.html"
                html_file_path_data = f"data-{level}.html"
                html_file_path_combined = f"combined-{level}.html"
        
        # Upload model explanation report
        if os.path.exists(html_file_path_model):
            try:
                logger.info(f"Uploading model explanation report: {html_file_path_model}")
                result_model = upload_xai_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    report_type="model_explanation",
                    level=level,
                    html_file_path=html_file_path_model
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
                    model_id=model_id,
                    report_type="data_explanation",
                    level=level,
                    html_file_path=html_file_path_data
                )
                logger.info(f"✅ Data explanation report uploaded successfully!")
                logger.info(f"   Report type: data_explanation")
                logger.info(f"   Level: {level}")
                logger.info(f"   Response: {json.dumps(result_data, indent=2, default=str)}")
            except Exception as e:
                logger.error(f"Failed to upload data explanation report: {e}", exc_info=True)
        else:
            logger.warning(f"Data HTML file not found: {html_file_path_data}")
            logger.info(f"Skipping data explanation - create {html_file_path_data} in the root directory to test")
        
        # Upload combined report
        if os.path.exists(html_file_path_combined):
            try:
                logger.info(f"Uploading combined XAI report: {html_file_path_combined}")
                result_combined = upload_xai_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    report_type="combined_explanation",
                    level=level,
                    html_file_path=html_file_path_combined
                )
                logger.info(f"✅ Combined XAI report uploaded successfully!")
                logger.info(f"   Report type: combined_explanation")
                logger.info(f"   Level: {level}")
                logger.info(f"   Response: {json.dumps(result_combined, indent=2, default=str)}")
                logger.info("   XAI events will be automatically sent by the DW")
            except Exception as e:
                logger.error(f"Failed to upload combined XAI report: {e}", exc_info=True)
        else:
            logger.warning(f"Combined HTML file not found: {html_file_path_combined}")
            logger.info(f"Skipping combined report - create {html_file_path_combined} in the root directory to test")
        
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
