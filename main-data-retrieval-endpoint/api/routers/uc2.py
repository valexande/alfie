"""
UC2 Router - Domain-specific endpoints for driver analysis.

Provides:
- Driver data analysis (heart rate + video frames)
- Driver alertness model explanation with fairness metrics
"""

from typing import Optional
import io
import base64

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np

from api.schemas import UserLevel
from xai_core.model_loader import load_model_from_bytes
from xai_core.explainer_service import ExplainerService
from xai_core.utils import detect_target_column, safe_compute

# Optional imports for visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

router = APIRouter(tags=["UC2 - Driver Analysis"])


# ============================================================================
# Driver Data Analysis
# ============================================================================

@router.post(
    "/explain-uc2-data",
    response_class=HTMLResponse,
    summary="Analyze driver alertness data"
)
async def explain_uc2_data(
    frame_file: UploadFile = File(
        ..., 
        description="CSV with video frame data (frame_timestamp, eyes_closed, yawning, alert)"
    ),
    hr_file: UploadFile = File(
        ..., 
        description="CSV with heart rate data (timestamp, heart_rate)"
    ),
    user_level: UserLevel = Form(
        default=UserLevel.expert,
        description="Report detail level"
    )
):
    """
    Analyze driver alertness data from video frames and heart rate.
    
    **Input Files:**
    - `frame_file`: CSV with columns: frame_timestamp, eyes_closed, yawning, alert
    - `hr_file`: CSV with columns: timestamp, heart_rate
    
    **Analysis Includes:**
    - Correlation analysis between heart rate and alertness indicators
    - Anomaly detection (abnormal heart rate patterns)
    - Driver state clustering
    - Time series visualization
    """
    try:
        # Read uploaded files
        frame_bytes = await frame_file.read()
        hr_bytes = await hr_file.read()
        
        frame_df = pd.read_csv(io.BytesIO(frame_bytes))
        hr_df = pd.read_csv(io.BytesIO(hr_bytes))
        
        print(f"Frame data shape: {frame_df.shape}")
        print(f"Heart rate data shape: {hr_df.shape}")
        
        # Convert timestamps
        frame_df['frame_timestamp'] = pd.to_datetime(frame_df['frame_timestamp'])
        hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
        
        # Merge data using time-based join
        merged_df = pd.merge_asof(
            frame_df.sort_values('frame_timestamp'),
            hr_df.sort_values('timestamp'),
            left_on='frame_timestamp',
            right_on='timestamp',
            direction='nearest'
        )
        
        print(f"Merged data shape: {merged_df.shape}")
        
        # Generate analysis
        analysis = _analyze_driver_data(merged_df)
        
        # Generate plots
        plots = _generate_driver_plots(merged_df)
        
        # Build HTML report
        html = _build_driver_data_report(
            merged_df, analysis, plots, user_level.value
        )
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        print(f"Error in explain_uc2_data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Driver Model Explanation
# ============================================================================

@router.post(
    "/explain-uc2-model",
    response_class=HTMLResponse,
    summary="Explain driver alertness model with fairness analysis"
)
async def explain_uc2_model(
    model_file: UploadFile = File(..., description="Trained model file"),
    encoder_file: UploadFile = File(..., description="Label encoders pickle"),
    data_file: UploadFile = File(..., description="CSV with demographics and features"),
    user_level: UserLevel = Form(default=UserLevel.expert)
):
    """
    Generate explainability report for driver alertness model with fairness analysis.
    
    **Includes:**
    - Standard model explainability (SHAP, feature importance)
    - Fairness metrics per demographic group (gender, race, ethnicity)
    - Alert distribution analysis
    """
    try:
        import joblib
        
        # Read files
        model_bytes = await model_file.read()
        encoder_bytes = await encoder_file.read()
        data_bytes = await data_file.read()
        
        # Load model
        model_info = load_model_from_bytes(model_bytes, model_file.filename)
        
        # Load encoders
        label_encoders = joblib.load(io.BytesIO(encoder_bytes))
        
        # Load data
        df = pd.read_csv(io.BytesIO(data_bytes))
        
        # Preprocess categorical columns
        categorical_cols = ['gender', 'ethnicity', 'race']
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        # Prepare features and target
        target_col = 'alert' if 'alert' in df.columns else detect_target_column(df)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Generate predictions
        model = model_info.sklearn_compatible_model
        y_pred = model.predict(X)
        
        # Compute fairness metrics
        fairness_metrics = {}
        for group_col in categorical_cols:
            if group_col in df.columns and group_col in label_encoders:
                fairness_metrics[group_col] = _compute_group_metrics(
                    df, y_pred, y, group_col, label_encoders[group_col]
                )
        
        # Generate explainability report
        service = ExplainerService(model_info=model_info, X=X, y=y)
        base_html = service.generate_html_report(mode=user_level.value)
        
        # Inject fairness analysis
        fairness_html = _generate_fairness_html(fairness_metrics)
        
        # Combine reports
        final_html = _inject_fairness_section(base_html, fairness_html)
        
        return HTMLResponse(content=final_html)
        
    except Exception as e:
        print(f"Error in explain_uc2_model: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def _analyze_driver_data(df: pd.DataFrame) -> dict:
    """Analyze merged driver data."""
    from scipy.stats import zscore
    
    analysis = {}
    
    # Correlation analysis
    corr_cols = ['heart_rate', 'eyes_closed', 'yawning', 'alert']
    available_cols = [c for c in corr_cols if c in df.columns]
    if len(available_cols) > 1:
        analysis['correlation'] = df[available_cols].corr().round(3).to_dict()
    
    # Anomaly detection (z-score > 2)
    if 'heart_rate' in df.columns:
        df['hr_zscore'] = zscore(df['heart_rate'].fillna(df['heart_rate'].mean()))
        hr_anomalies = df[df['hr_zscore'].abs() > 2]
        analysis['hr_anomalies'] = len(hr_anomalies)
    
    # Fatigue anomalies (yawning/eyes_closed but not alert)
    if all(c in df.columns for c in ['yawning', 'eyes_closed', 'alert']):
        fatigue_anomalies = df[
            ((df['yawning'] == 1) | (df['eyes_closed'] == 1)) & 
            (df['alert'] == 0)
        ]
        analysis['fatigue_anomalies'] = len(fatigue_anomalies)
    
    # Clustering
    try:
        from sklearn.cluster import KMeans
        
        cluster_cols = ['heart_rate', 'eyes_closed', 'yawning', 'alert']
        available_cluster_cols = [c for c in cluster_cols if c in df.columns]
        
        if len(available_cluster_cols) >= 2:
            features = df[available_cluster_cols].fillna(0)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            df['cluster'] = kmeans.fit_predict(features)
            analysis['n_clusters'] = 4
    except:
        pass
    
    analysis['n_samples'] = len(df)
    
    return analysis


def _generate_driver_plots(df: pd.DataFrame) -> dict:
    """Generate visualization plots for driver data."""
    plots = {}
    
    # Time series plot
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'frame_timestamp' in df.columns and 'heart_rate' in df.columns:
            ax.plot(df['frame_timestamp'], df['heart_rate'], label='Heart Rate', alpha=0.7)
            
            if 'alert' in df.columns:
                alert_df = df[df['alert'] == 1]
                ax.scatter(
                    alert_df['frame_timestamp'], 
                    alert_df['heart_rate'],
                    color='red', 
                    label='Alert',
                    s=50,
                    zorder=5
                )
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Heart Rate')
            ax.set_title('Heart Rate Over Time with Alerts')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plots['time_series'] = _fig_to_base64(fig)
    except Exception as e:
        print(f"Time series plot failed: {e}")
    
    # Cluster plot
    try:
        if 'cluster' in df.columns and 'heart_rate' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(
                df['heart_rate'], 
                df.get('alert', np.zeros(len(df))),
                c=df['cluster'],
                cmap='viridis',
                alpha=0.6
            )
            
            ax.set_xlabel('Heart Rate')
            ax.set_ylabel('Alert')
            ax.set_title('Driver State Clustering')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            
            plots['clusters'] = _fig_to_base64(fig)
    except Exception as e:
        print(f"Cluster plot failed: {e}")
    
    return plots


def _build_driver_data_report(
    df: pd.DataFrame, 
    analysis: dict, 
    plots: dict, 
    mode: str
) -> str:
    """Build HTML report for driver data analysis."""
    
    # Correlation table
    corr_html = ""
    if 'correlation' in analysis:
        corr_df = pd.DataFrame(analysis['correlation'])
        corr_html = f"""
        <div class="section">
            <h2>Correlation Matrix</h2>
            {corr_df.to_html(classes='table')}
        </div>
        """
    
    # Plots
    plots_html = ""
    if 'time_series' in plots:
        plots_html += f"""
        <div class="section">
            <h2>Heart Rate Time Series</h2>
            <img src="data:image/png;base64,{plots['time_series']}" alt="Time Series"/>
        </div>
        """
    
    if 'clusters' in plots:
        plots_html += f"""
        <div class="section">
            <h2>Driver State Clustering</h2>
            <img src="data:image/png;base64,{plots['clusters']}" alt="Clusters"/>
        </div>
        """
    
    # Build full HTML
    if mode == 'beginner':
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Driver Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
                h1 {{ color: #667eea; }}
                h2 {{ color: #764ba2; }}
                img {{ max-width: 100%; border-radius: 8px; }}
                .highlight {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Driver Summary</h1>
            <div class="highlight">
                <p>This report shows how heart rate and signs like yawning or closed eyes help tell if a driver is alert or sleepy.</p>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <ul>
                    <li><strong>{analysis.get('n_samples', 0)}</strong> data points analyzed</li>
                    <li><strong>{analysis.get('hr_anomalies', 0)}</strong> unusual heart rate patterns detected</li>
                    <li><strong>{analysis.get('fatigue_anomalies', 0)}</strong> potential fatigue events found</li>
                </ul>
            </div>
            
            {plots_html}
        </body>
        </html>
        """
    else:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Driver Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .section {{ margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }}
                h1 {{ color: #667eea; }}
                h2 {{ color: #764ba2; margin-bottom: 15px; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .table th, .table td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                .table th {{ background: #f0f0f0; }}
                img {{ max-width: 100%; border-radius: 8px; margin: 15px 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #667eea; }}
            </style>
        </head>
        <body>
            <h1>Driver Analysis Report</h1>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div style="color: #666; font-size: 0.9em;">Total Samples</div>
                        <div style="font-size: 1.5em; font-weight: bold;">{analysis.get('n_samples', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <div style="color: #666; font-size: 0.9em;">HR Anomalies</div>
                        <div style="font-size: 1.5em; font-weight: bold;">{analysis.get('hr_anomalies', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <div style="color: #666; font-size: 0.9em;">Fatigue Events</div>
                        <div style="font-size: 1.5em; font-weight: bold;">{analysis.get('fatigue_anomalies', 0)}</div>
                    </div>
                </div>
            </div>
            
            {corr_html}
            {plots_html}
        </body>
        </html>
        """
    
    return html


def _compute_group_metrics(
    df: pd.DataFrame, 
    predictions: np.ndarray, 
    y_true: pd.Series,
    group_col: str,
    label_encoder
) -> dict:
    """Compute accuracy, precision, recall per demographic group."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    results = {}
    df_copy = df.copy()
    df_copy['prediction'] = predictions
    df_copy['true'] = y_true.values
    
    for group in df_copy[group_col].unique():
        subset = df_copy[df_copy[group_col] == group]
        
        if len(subset) < 5:  # Skip small groups
            continue
        
        acc = accuracy_score(subset['true'], subset['prediction'])
        prec = precision_score(subset['true'], subset['prediction'], zero_division=0)
        rec = recall_score(subset['true'], subset['prediction'], zero_division=0)
        
        # Decode label
        try:
            label = label_encoder.inverse_transform([group])[0]
        except:
            label = str(group)
        
        results[label] = {
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3)
        }
    
    return results


def _generate_fairness_html(fairness_metrics: dict) -> str:
    """Generate HTML section for fairness metrics."""
    
    html_blocks = []
    
    for group_name, metrics in fairness_metrics.items():
        if not metrics:
            continue
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = group_name.title()
        
        html_blocks.append(f"""
        <div class="section">
            <h3>Fairness Metrics: {group_name.title()}</h3>
            {metrics_df.to_html(classes='table')}
        </div>
        """)
    
    if not html_blocks:
        return ""
    
    return f"""
    <div class="section">
        <h2>Fairness Analysis</h2>
        <p>Performance metrics broken down by demographic groups to identify potential bias.</p>
        {''.join(html_blocks)}
    </div>
    """


def _inject_fairness_section(base_html: str, fairness_html: str) -> str:
    """Inject fairness section into existing report."""
    if not fairness_html:
        return base_html
    
    # Try to inject before closing body tag
    if '</body>' in base_html:
        return base_html.replace('</body>', f'{fairness_html}</body>')
    
    # Fallback: append to end
    return base_html + fairness_html


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
