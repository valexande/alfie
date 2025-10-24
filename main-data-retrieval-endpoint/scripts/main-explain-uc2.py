from flask import Flask, request, render_template_string
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

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "uc2-explanation-api"}, 200


@app.route('/explain-uc2-data', methods=['POST'])
def explain_data():
    """Driver data analysis endpoint"""
    try:
        print("✅ Data analysis request received")

        user_level = request.form.get('user_level', 'expert').lower()
        print(f"User level: {user_level}")

        # Read uploaded files
        frame_file = request.files['frame_file']
        hr_file = request.files['hr_file']
        
        print("Files received, loading...")

        # Load files
        frame_df = pd.read_csv(frame_file)
        print(f"Frame data loaded successfully, shape: {frame_df.shape}")
        
        heart_rate_df = pd.read_csv(hr_file)
        print(f"Heart rate data loaded successfully, shape: {heart_rate_df.shape}")

        # Convert timestamps
        frame_df['frame_timestamp'] = pd.to_datetime(frame_df['frame_timestamp'])
        heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])

        # Merge data
        merged_df = pd.merge_asof(frame_df.sort_values('frame_timestamp'),
                                  heart_rate_df.sort_values('timestamp'),
                                  left_on='frame_timestamp',
                                  right_on='timestamp',
                                  direction='nearest')

        print(f"Merged data shape: {merged_df.shape}")

        # Analysis
        correlation_results = merged_df[['heart_rate', 'eyes_closed', 'yawning', 'alert']].corr().round(2)
        corr_html = correlation_results.to_html(classes="table table-hover")

        merged_df['Heart Rate Z-Score'] = zscore(merged_df['heart_rate'])
        anomalies = merged_df[(merged_df['Heart Rate Z-Score'].abs() > 2) |
                              ((merged_df['yawning'] | merged_df['eyes_closed']) & ~merged_df['alert'])]
        anomalies_html = anomalies[['frame_timestamp', 'heart_rate', 'eyes_closed', 'yawning', 'alert']].head(20).to_html(
            classes="table table-bordered")

        # Generate plots
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
            return render_template_string(summary_html)
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
            return render_template_string(html_template)
    
    except Exception as e:
        print(f"❌ Error in explain_data function: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@app.route('/explain-uc2-model', methods=['POST'])
def explain():
    try:
        print("✅ Request received")

        user_level = request.form.get('user_level', 'expert').lower()  # 'beginner' or 'expert'
        print(f"User level: {user_level}")

        # Read uploaded files
        model_file = request.files['model_file']
        encoder_file = request.files['encoder_file']
        data_file = request.files['data_file']
        
        print("Files received, loading...")

        # Load files
        model = joblib.load(model_file)
        print("Model loaded successfully")
        
        label_encoders = joblib.load(encoder_file)
        print("Encoders loaded successfully")
        
        df = pd.read_csv(data_file)
        print(f"Data loaded successfully, shape: {df.shape}")

        # Preprocess
        categorical_cols = ['gender', 'ethnicity', 'race']
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])

        X = df.drop(columns=['alert'])
        y = df['alert']

        # Predict with sklearn version compatibility fix
        try:
            predictions = model.predict(X)
            report_dict = classification_report(y, predictions, output_dict=True)
        except AttributeError as e:
            if 'monotonic_cst' in str(e):
                print("Fixing sklearn version compatibility issue...")
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
        
        # Handle SHAP values shape for binary classification
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals = shap_values[1]  # Use positive class for binary classification
        elif shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]  # 3D array, get positive class
        else:
            shap_vals = shap_values
        
        shap_bar_b64 = plot_to_base64(lambda: shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False))
        shap_full_b64 = plot_to_base64(lambda: shap.summary_plot(shap_vals, X_sample, show=False))

        # Reverse-transform encoded labels
        for col in categorical_cols:
            df[col + '_label'] = label_encoders[col].inverse_transform(df[col])

        # Alert breakdown plot
        alert_plot_b64 = plot_alert_counts(df)

        # Fairness metrics
        fairness_html_blocks = []
        for group in categorical_cols:
            fairness_df = pd.DataFrame.from_dict(
                compute_group_metrics(df, predictions, y, group, label_encoders[group]),
                orient='index'
            )
            fairness_html_blocks.append(
                f"<h3>Fairness for {group.capitalize()}</h3>" + fairness_df.to_html(classes="table table-striped"))

        # Beginner Summary
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
            return render_template_string(beginner_html)

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
        return render_template_string(html_template)
    
    except Exception as e:
        print(f"❌ Error in explain function: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


def plot_to_base64(plot_func):
    buf = io.BytesIO()
    plot_func()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


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


@app.route('/explain-uc2-data', methods=['POST'])
def driver_analysis():
    print("✅ Driver analysis request received")

    user_level = request.form.get('user_level', 'expert')

    frame_file = request.files['frame_file']
    hr_file = request.files['hr_file']

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
        return render_template_string(summary_html)

    # Full expert report
    html_template = f"""
    <html>
    <head><title>Driver Analysis Report</title> ... </head>
    <body>
        <h1>Driver Analysis Report</h1>
        <h2>Correlation Matrix</h2>{corr_html}
        <h2>Detected Anomalies (First 20 rows)</h2>{anomalies_html}
        <h2>Heart Rate Time Series with Alerts</h2><img src="data:image/png;base64,{ts_plot_b64}" />
        <h2>Driver State Clustering</h2><img src="data:image/png;base64,{cluster_plot_b64}" />
    </body></html>
    """
    return render_template_string(html_template)


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
