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
from xai_core.model_loader import load_model_from_bytes, load_vision_model_from_bytes
from xai_core.explainer_service import ExplainerService
from xai_core.utils import detect_target_column, safe_compute

# Optional vision imports
try:
    import torch
    from PIL import Image as PILImage
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
# Vision Model Explanation  (PyTorch image classifier)
# ============================================================================

@router.post(
    "/explain-vision-model",
    response_class=HTMLResponse,
    summary="Explain a PyTorch image-classification model (cardboard / glass / metal)",
)
async def explain_vision_model(
    model_file: UploadFile = File(
        ..., description="Trained PyTorch model file (.pt)"
    ),
    labels_file: UploadFile = File(
        ..., description="labels.json with id2label mapping"
    ),
    image_files: list[UploadFile] = File(
        ..., description="One or more image files to classify and explain"
    ),
    user_level: UserLevel = Form(default=UserLevel.expert),
):
    """
    Run inference and generate a GradCAM-based explainability report for a
    PyTorch image-classification model.

    **Inputs:**
    - `model_file`: `.pt` file saved with ``torch.save(model, path)`` or
      as a state-dict.
    - `labels_file`: JSON with ``{"id2label": {"0": "cardboard", ...}}``
      (same format produced by the training pipeline).
    - `image_files`: One or more JPEG/PNG images to classify.

    **Report includes:**
    - Per-image predicted class + confidence bar
    - Class probability breakdown
    - GradCAM saliency overlay (expert mode)
    - Aggregate class distribution across all uploaded images
    """
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="PyTorch / Pillow not installed in this environment.",
        )

    try:
        model_bytes = await model_file.read()
        labels_bytes = await labels_file.read()

        vision_info = load_vision_model_from_bytes(
            model_bytes, labels_bytes, model_file.filename
        )

        results = []
        for img_upload in image_files:
            img_bytes = await img_upload.read()
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

            pred = vision_info.predict_image(pil_img)

            gradcam_b64 = None
            if user_level == UserLevel.expert:
                gradcam_b64 = _compute_gradcam(vision_info, pil_img)

            thumb_b64 = _pil_to_base64(pil_img.resize((224, 224)))

            results.append({
                "filename": img_upload.filename,
                "predicted_class": pred["predicted_class"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "thumbnail_b64": thumb_b64,
                "gradcam_b64": gradcam_b64,
            })

        html = _build_vision_report(results, vision_info, user_level.value)
        return HTMLResponse(content=html)

    except Exception as e:
        print(f"Error in explain_vision_model: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Vision helper functions
# ============================================================================

def _pil_to_base64(img: "PILImage.Image") -> str:
    """Encode a PIL image as base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _compute_gradcam(
    vision_info: "VisionModelInfo",
    pil_img: "PILImage.Image",
) -> Optional[str]:
    """
    Compute a GradCAM saliency map for the top predicted class and return
    a base64 PNG showing the original image alongside the overlay.

    Works with any architecture supported by ``VisionModelInfo.get_gradcam_target()``:
    - CNN families (ResNet, EfficientNet, ConvNeXt, VGG, MobileNet, DenseNet)
      via activation × gradient weights on the target conv layer.
    - Vision Transformers (ViT) via the last LayerNorm block; the spatial
      tokens are reshaped back to a 2-D grid.

    Returns None without raising if the architecture is incompatible or if
    any step fails.
    """
    try:
        import torch
        import numpy as np

        model = vision_info.model
        target_layer = vision_info.get_gradcam_target()
        if target_layer is None:
            print("GradCAM: no suitable layer found")
            return None

        transform = vision_info._build_transform()
        tensor = transform(pil_img.convert("RGB")).unsqueeze(0)  # (1,C,H,W)

        activations: list = []
        gradients: list = []

        def fwd_hook(_, __, output):
            activations.append(output)

        def bwd_hook(_, __, grad_output):
            gradients.append(grad_output[0])

        fwd_h = target_layer.register_forward_hook(fwd_hook)
        bwd_h = target_layer.register_full_backward_hook(bwd_hook)

        model.eval()
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, pred_idx].backward()

        fwd_h.remove()
        bwd_h.remove()

        if not activations or not gradients:
            return None

        act  = activations[0].detach().squeeze(0)   # (C, H, W)  or  (N, C)  for ViT
        grad = gradients[0].detach().squeeze(0)

        # ViT path: activations are (seq_len, C) → reshape to spatial grid
        if act.dim() == 2:
            seq_len, c = act.shape
            # drop CLS token if present (seq_len = 1 + H*W)
            if int(seq_len ** 0.5) ** 2 != seq_len:
                act  = act[1:]   # drop CLS
                grad = grad[1:]
                seq_len -= 1
            grid = int(seq_len ** 0.5)
            act  = act.reshape(grid, grid, c).permute(2, 0, 1)   # (C, g, g)
            grad = grad.reshape(grid, grid, c).permute(2, 0, 1)

        weights = grad.mean(dim=(1, 2))                          # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)
        cam = cam.numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        size = vision_info.input_size
        img_np = np.array(pil_img.resize((size, size)))
        cam_up = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize((size, size))
        ) / 255.0

        heatmap = cm.get_cmap("jet")(cam_up)[:, :, :3]
        overlay = (0.55 * img_np / 255.0 + 0.45 * heatmap).clip(0, 1)

        pred_label = vision_info.labels.get(pred_idx, str(pred_idx))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_np);  axes[0].set_title("Original");  axes[0].axis("off")
        axes[1].imshow(overlay); axes[1].set_title(f"GradCAM → {pred_label}"); axes[1].axis("off")
        plt.tight_layout()
        return _fig_to_base64(fig)

    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None


def _build_vision_report(results: list, vision_info, mode: str) -> str:
    """Build the HTML explainability report for vision model results."""

    # Aggregate class distribution
    from collections import Counter
    class_counts = Counter(r["predicted_class"] for r in results)

    # ---- Per-image cards ----
    cards_html = ""
    for r in results:
        prob_bars = ""
        for cls_name, prob in sorted(r["probabilities"].items(),
                                     key=lambda x: -x[1]):
            bar_width = int(prob * 100)
            color = "#667eea" if cls_name == r["predicted_class"] else "#b0b0c0"
            prob_bars += f"""
            <div style="margin:4px 0">
              <span style="display:inline-block;width:90px;font-size:0.85em">{cls_name}</span>
              <span style="display:inline-block;background:{color};
                           width:{bar_width}%;height:14px;border-radius:3px;
                           vertical-align:middle"></span>
              <span style="font-size:0.8em;margin-left:6px">{prob:.1%}</span>
            </div>"""

        gradcam_html = ""
        if r.get("gradcam_b64"):
            gradcam_html = f"""
            <div style="margin-top:10px">
              <img src="data:image/png;base64,{r['gradcam_b64']}"
                   style="max-width:100%;border-radius:6px"/>
            </div>"""

        cards_html += f"""
        <div style="background:#fff;border-radius:8px;padding:16px;
                    box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:16px">
          <div style="display:flex;gap:16px;align-items:flex-start">
            <img src="data:image/png;base64,{r['thumbnail_b64']}"
                 style="width:112px;height:112px;object-fit:cover;border-radius:6px"/>
            <div style="flex:1">
              <div style="font-weight:bold;font-size:1.05em">{r['filename']}</div>
              <div style="color:#667eea;font-size:1.3em;font-weight:bold;margin:4px 0">
                {r['predicted_class']}
                <span style="font-size:0.75em;color:#555">
                  ({r['confidence']:.1%} confidence)
                </span>
              </div>
              {prob_bars}
            </div>
          </div>
          {gradcam_html}
        </div>"""

    # ---- Class distribution bar chart ----
    dist_rows = ""
    total = len(results) or 1
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / total
        dist_rows += f"""
        <div style="margin:6px 0">
          <span style="display:inline-block;width:100px">{cls_name}</span>
          <span style="display:inline-block;background:#764ba2;
                       width:{int(pct*200)}px;height:16px;border-radius:3px;
                       vertical-align:middle"></span>
          <span style="margin-left:8px">{count} / {total} ({pct:.0%})</span>
        </div>"""

    gradcam_note = (
        "<p style='color:#555;font-size:0.9em'>"
        "GradCAM overlays highlight the image regions most influential "
        "for each prediction (red = high influence).</p>"
        if mode == "expert" else ""
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Vision Model Report</title>
  <style>
    body {{font-family:Arial,sans-serif;margin:40px;background:#f4f4f8;
           line-height:1.5;color:#222}}
    h1 {{color:#667eea}} h2 {{color:#764ba2;margin-top:28px}}
    .meta {{background:#fff;border-radius:8px;padding:16px;
            box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:20px}}
    .meta span {{margin-right:24px;font-size:0.9em;color:#555}}
    .meta strong {{color:#222}}
  </style>
</head>
<body>
  <h1>Vision Model — Explainability Report</h1>

  <div class="meta">
    <span>Model type: <strong>PyTorch image classifier</strong></span>
    <span>Classes: <strong>{', '.join(vision_info.labels.values())}</strong></span>
    <span>Images analysed: <strong>{len(results)}</strong></span>
  </div>

  <h2>Class Distribution</h2>
  <div style="background:#fff;border-radius:8px;padding:16px;
              box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:20px">
    {dist_rows}
  </div>

  <h2>Per-Image Results</h2>
  {gradcam_note}
  {cards_html}
</body>
</html>"""
    return html


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
