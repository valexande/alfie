import requests
import zipfile
import os
import tempfile
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use('Agg')  # ✅ Must come BEFORE pyplot
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from datetime import datetime

def download_file(url, filename):
    """Download a file from URL and save it locally"""
    print(f"[DOWNLOAD] Downloading {filename} from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"[SUCCESS] Downloaded {filename}")
    return filename

def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory"""
    print(f"[EXTRACT] Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[SUCCESS] Extracted to {extract_to}")

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

def plot_to_base64(plot_func):
    buf = io.BytesIO()
    plot_func()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_alert_counts(df):
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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def compute_group_metrics(df, predictions, y_true, group_col, label_encoder):
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
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='frame_timestamp', y='heart_rate', label='Heart Rate')
    sns.scatterplot(data=df[df['alert']], x='frame_timestamp', y='heart_rate', color='red', label='Alert')
    plt.xticks(rotation=45)
    plt.title('Heart Rate over Time with Alerts')
    plt.tight_layout()

def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='heart_rate', y='alert', hue='Cluster', palette='viridis')
    plt.title('Driver State Clustering')
    plt.tight_layout()

def analyze_model_explanation(data_dir, model_dir, user_level='expert'):
    """Analyze model explanation with downloaded data"""
    print("[ANALYSIS] Starting model explanation analysis...")
    
    # Find files
    data_files = find_files_in_dir(data_dir, ['.csv'])
    model_files = find_files_in_dir(model_dir, ['.pkl'])
    
    # Load main data file - prioritize files with 'alert' column
    data_file = None
    alert_files = []
    
    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        df_test = pd.read_csv(csv_file)
        print(f"[DEBUG] Checking file {csv_file} with columns: {list(df_test.columns)}")
        if 'alert' in df_test.columns:
            alert_files.append((csv_file, len(df_test)))
            print(f"[DEBUG] Found alert file: {csv_file} with {len(df_test)} rows")
    
    if alert_files:
        # Choose the largest file with alert column
        data_file = max(alert_files, key=lambda x: x[1])[0]
        print(f"[INFO] Using alert file: {data_file}")
    else:
        data_file = csv_files[0] if csv_files else None  # Take first CSV if no alert column found
        print(f"[WARNING] No alert column found, using first CSV: {data_file}")
    
    # Load model and encoder
    pkl_files = model_files.get('.pkl', [])
    if not pkl_files:
        raise FileNotFoundError("No .pkl file found in model directory")
    
    # Find the correct files
    model_file = None
    encoder_file = None
    
    model = None
    label_encoders = None
    
    for pkl_file in pkl_files:
        try:
            loaded_obj = joblib.load(pkl_file)
            if hasattr(loaded_obj, 'predict'):
                model = loaded_obj
                print(f"[DEBUG] Found model file: {pkl_file}")
            elif isinstance(loaded_obj, dict) and 'gender' in loaded_obj:
                label_encoders = loaded_obj
                print(f"[DEBUG] Found encoder file: {pkl_file}")
        except:
            continue
    
    if not model:
        raise FileNotFoundError("No model file found in model directory")
    if not label_encoders:
        raise FileNotFoundError("No encoder file found in model directory")
    df = pd.read_csv(data_file)
    
    print(f"[DEBUG] Loaded data with shape: {df.shape}")
    print(f"[DEBUG] Data columns: {list(df.columns)}")
    
    # Preprocess
    categorical_cols = ['gender', 'ethnicity', 'race']
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Check if alert column exists, if not create a dummy one
    if 'alert' not in df.columns:
        print("[WARNING] No 'alert' column found, creating dummy alert column")
        df['alert'] = 0  # Create dummy alert column
    
    X = df.drop(columns=['alert'])
    y = df['alert']
    
    # Predict with sklearn version compatibility fix
    try:
        predictions = model.predict(X)
        report_dict = classification_report(y, predictions, output_dict=True)
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            print(f"[INFO] Fixing sklearn version compatibility issue...")
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
    print("[ANALYSIS] Starting driver analysis...")
    
    # Find CSV files
    data_files = find_files_in_dir(data_dir, ['.csv'])
    
    # Get all CSV files
    csv_files = data_files.get('.csv', [])
    print(f"[DEBUG] Found {len(csv_files)} CSV files for driver analysis")
    
    # Look for specific files
    frame_file = None
    hr_file = None
    
    for csv_file in csv_files:
        df_test = pd.read_csv(csv_file)
        print(f"[DEBUG] Checking driver file {csv_file} with columns: {list(df_test.columns)}")
        if 'frame_timestamp' in df_test.columns:
            frame_file = csv_file
            print(f"[DEBUG] Found frame file: {csv_file}")
        elif 'heart_rate' in df_test.columns or 'timestamp' in df_test.columns:
            hr_file = csv_file
            print(f"[DEBUG] Found heart rate file: {csv_file}")
    
    if not frame_file or not hr_file:
        # Use any available CSV files
        if len(csv_files) >= 2:
            frame_file = csv_files[0]
            hr_file = csv_files[1]
        else:
            raise FileNotFoundError("Need at least 2 CSV files for driver analysis")
    
    print(f"[INFO] Using frame file: {frame_file}")
    print(f"[INFO] Using heart rate file: {hr_file}")
    
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

def main():
    """Main function to download data and run analysis"""
    print("[START] Starting automatic UC2 analysis...")
    
    # URLs for downloading data
    data_url = 'http://160.40.52.44:8000/datasets/user9/xai-uc2-data/download'
    model_url = 'http://160.40.52.44:8000/ai-models/user9/automl_xai-uc2-data_1760336857/download?version=v1'
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        data_zip = os.path.join(temp_dir, 'data.zip')
        model_zip = os.path.join(temp_dir, 'model.zip')
        
        data_extract_dir = os.path.join(temp_dir, 'data_extracted')
        model_extract_dir = os.path.join(temp_dir, 'model_extracted')
        os.makedirs(data_extract_dir, exist_ok=True)
        os.makedirs(model_extract_dir, exist_ok=True)
        
        try:
            # Download files
            download_file(data_url, data_zip)
            download_file(model_url, model_zip)
            
            # Extract files
            extract_zip(data_zip, data_extract_dir)
            extract_zip(model_zip, model_extract_dir)
            
            # Run model explanation analysis
            print("\n" + "="*50)
            print("MODEL EXPLANATION ANALYSIS")
            print("="*50)
            
            model_report = analyze_model_explanation(data_extract_dir, model_extract_dir, user_level='expert')
            
            # Run driver analysis
            print("\n" + "="*50)
            print("DRIVER ANALYSIS")
            print("="*50)
            
            driver_report = analyze_driver_data(data_extract_dir, user_level='expert')
            
            # Combine both reports into a single comprehensive report
            print("\n" + "="*50)
            print("COMBINING REPORTS")
            print("="*50)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_report_file = f"combined_analysis_report_{timestamp}.html"
            
            # Extract the body content from both reports
            import re
            
            # Extract model report body content
            model_body_match = re.search(r'<body>(.*?)</body>', model_report, re.DOTALL)
            model_body = model_body_match.group(1) if model_body_match else model_report
            
            # Extract driver report body content  
            driver_body_match = re.search(r'<body>(.*?)</body>', driver_report, re.DOTALL)
            driver_body = driver_body_match.group(1) if driver_body_match else driver_report
            
            # Create combined HTML report
            combined_html = f"""
            <html>
            <head>
                <title>Combined UC2 Analysis Report</title>
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
                    <h1>UC2 Driver Alertness Analysis Report</h1>
                    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>This comprehensive report combines both model explanation and driver data analysis to provide a complete understanding of driver alertness patterns and AI model performance.</p>
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
            with open(combined_report_file, 'w', encoding='utf-8') as f:
                f.write(combined_html)
            print(f"[SAVE] Combined analysis report saved to: {combined_report_file}")
            
            # Upload report to API
            print("\n" + "="*50)
            print("UPLOADING REPORT TO API")
            print("="*50)
            
            upload_url = 'http://160.40.52.44:8000/xai-reports/upload/user10'
            
            # Determine report type and level based on user_level
            report_type = 'model_explanation'
            level = 'expert'  # Since we're running for expert level
            
            try:
                with open(combined_report_file, 'rb') as f:
                    files = {'file': (combined_report_file, f, 'text/html')}
                    data = {
                        'dataset_id': 'xai-uc2-data',
                        'model_id': 'automl_xai-uc2-data_1760336857',
                        'report_type': report_type,
                        'level': level
                    }
                    
                    print(f"[UPLOAD] Uploading report to {upload_url}")
                    print(f"[UPLOAD] Report type: {report_type}, Level: {level}")
                    
                    response = requests.post(upload_url, files=files, data=data)
                    response.raise_for_status()
                    
                    print(f"[SUCCESS] Report uploaded successfully!")
                    print(f"[RESPONSE] {response.text}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to upload report: {str(e)}")
                print(f"[INFO] Report saved locally at: {combined_report_file}")
            
            print("\n[SUCCESS] Analysis completed successfully!")
            print(f"[REPORTS] Combined report generated:")
            print(f"   - {combined_report_file}")
            
        except Exception as e:
            print(f"[ERROR] Error during analysis: {str(e)}")
            raise

if __name__ == '__main__':
    main()
