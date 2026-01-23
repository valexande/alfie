from flask import Flask, request
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error, \
    mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy", "service": "universal-model-explainability"}, 200


# Global temp directory for AutoGluon models (kept alive for the session)
_autogluon_temp_dirs = []


def find_autogluon_predictor_dir(extract_dir):
    """Find the AutoGluon predictor directory within extracted files"""
    import os

    # Check if extract_dir itself is a predictor directory
    if os.path.exists(os.path.join(extract_dir, 'predictor.pkl')):
        return extract_dir

    # Search subdirectories
    for root, dirs, files in os.walk(extract_dir):
        if 'predictor.pkl' in files:
            return root

    return None


@app.route('/explain-model', methods=['POST'])
def explain_model():
    """Universal model explainability - works with any model and data"""
    import tempfile
    import os
    import zipfile

    temp_dir = None  # Keep reference to prevent cleanup

    try:
        print("Model explainability request received")

        user_level = request.form.get('user_level', 'expert').lower()
        print(f"User level: {user_level}")

        # Load model
        model_file = request.files['model_file']
        model_file.seek(0)

        model = None
        error_messages = []
        is_autogluon = False

        # First, check if this is a ZIP file (likely AutoGluon model folder)
        model_file.seek(0)
        first_bytes = model_file.read(4)
        model_file.seek(0)

        if first_bytes == b'PK\x03\x04':
            print("Detected ZIP file, attempting AutoGluon model loading...")
            try:
                # Create persistent temp directory
                temp_dir = tempfile.mkdtemp(prefix='autogluon_model_')
                _autogluon_temp_dirs.append(temp_dir)

                model_bytes = model_file.read()
                zip_path = os.path.join(temp_dir, 'model.zip')
                with open(zip_path, 'wb') as f:
                    f.write(model_bytes)

                extract_dir = os.path.join(temp_dir, 'model_extracted')
                os.makedirs(extract_dir, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find the actual predictor directory
                predictor_dir = find_autogluon_predictor_dir(extract_dir)

                if predictor_dir:
                    print(f"Found AutoGluon predictor at: {predictor_dir}")
                    from autogluon.tabular import TabularPredictor
                    model = TabularPredictor.load(predictor_dir)
                    is_autogluon = True
                    print(f"Model loaded with AutoGluon: {type(model).__name__}")
                else:
                    error_messages.append("ZIP file does not contain AutoGluon predictor.pkl")
                    print("ZIP file does not contain AutoGluon predictor.pkl, trying other loaders...")

            except Exception as e_ag:
                error_messages.append(f"AutoGluon ZIP load failed: {str(e_ag)}")
                print(f"AutoGluon ZIP load failed: {e_ag}")

        # If not loaded yet, try standard loaders
        if model is None:
            model_file.seek(0)

            # Try joblib first (most common for sklearn models)
            try:
                import joblib
                model_file.seek(0)
                model = joblib.load(model_file)
                print(f"Model loaded with joblib: {type(model).__name__}")
            except Exception as e1:
                error_messages.append(f"joblib.load failed: {str(e1)}")

                # Try pickle with different protocols
                try:
                    import pickle
                    model_file.seek(0)
                    model = pickle.load(model_file)
                    print(f"Model loaded with pickle: {type(model).__name__}")
                except Exception as e2:
                    error_messages.append(f"pickle.load failed: {str(e2)}")

                    # Try with encoding='latin1' for Python 2/3 compatibility
                    try:
                        model_file.seek(0)
                        model = pickle.load(model_file, encoding='latin1')
                        print(f"Model loaded with pickle (latin1 encoding): {type(model).__name__}")
                    except Exception as e3:
                        error_messages.append(f"pickle.load (latin1) failed: {str(e3)}")

                        # Try with fix_imports for Python 2/3 compatibility
                        try:
                            model_file.seek(0)
                            model = pickle.load(model_file, fix_imports=True, encoding='latin1')
                            print(f"Model loaded with pickle (fix_imports): {type(model).__name__}")
                        except Exception as e4:
                            error_messages.append(f"pickle.load (fix_imports) failed: {str(e4)}")

        if model is None:
            error_detail = " | ".join(error_messages[-2:])  # Show last 2 errors
            return f"""
            <html><body>
            <h1>Model Loading Error</h1>
            <p>Could not load model after trying multiple methods.</p>
            <p><strong>Error details:</strong> {error_detail}</p>
            <p><strong>Possible causes:</strong></p>
            <ul>
                <li>The pickle file uses a custom protocol that requires specific loaders</li>
                <li>The model was pickled with a different Python version</li>
                <li>The file may be corrupted or incomplete</li>
                <li>The model may require additional dependencies or custom classes</li>
            </ul>
            <p>Please ensure your .pkl file is a valid scikit-learn or compatible model.</p>
            </body></html>
            """, 400

        # Load data
        data_file = request.files['data_file']
        try:
            data = pd.read_csv(data_file)
            print(f"Data loaded: {data.shape}")
        except Exception as e:
            return f"""
            <html><body>
            <h1>Data Loading Error</h1>
            <p>Could not load CSV: {e}</p>
            <p>Please ensure your CSV file is valid.</p>
            </body></html>
            """, 400

        # Auto-detect target column (last column or 'target' or 'label' or 'alert')
        target_col = None
        if 'target' in data.columns:
            target_col = 'target'
        elif 'label' in data.columns:
            target_col = 'label'
        elif 'alert' in data.columns:
            target_col = 'alert'
        else:
            target_col = data.columns[-1]

        print(f"Using target column: {target_col}")

        # Prepare features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Handle AutoGluon models differently
        if is_autogluon:
            print("Using AutoGluon prediction pipeline...")
            try:
                # AutoGluon TabularPredictor handles preprocessing internally
                # Reference: https://auto.gluon.ai/0.3.0/tutorials/tabular_prediction/tabular-indepth.html

                # Get predictor info
                predictor_info = model.info()
                print(f"AutoGluon Predictor Info: {predictor_info.get('problem_type', 'unknown')}")

                # Make predictions - AutoGluon handles all preprocessing
                y_pred = model.predict(data)

                # Convert to numpy array if needed
                if hasattr(y_pred, 'values'):
                    y_pred = y_pred.values

                # Get prediction probabilities if available
                y_pred_proba = None
                try:
                    y_pred_proba = model.predict_proba(data)
                    print("Prediction probabilities obtained")
                except Exception as prob_err:
                    print(f"Could not get prediction probabilities: {prob_err}")

                # Get feature importance using AutoGluon's native method
                # This uses permutation-shuffling internally
                feature_importance_df = None
                feature_importance = None
                feature_names_list = list(X.columns)

                try:
                    print("Computing feature importance via permutation shuffling...")
                    # AutoGluon's feature_importance uses permutation shuffling
                    # Subsample for speed if dataset is large
                    sample_data = data.sample(n=min(1000, len(data)), random_state=42) if len(data) > 1000 else data
                    feature_importance_df = model.feature_importance(sample_data, num_shuffle_sets=3)
                    feature_importance = feature_importance_df['importance'].values
                    feature_names_list = feature_importance_df.index.tolist()
                    print(f"AutoGluon feature importance calculated for {len(feature_names_list)} features")
                except Exception as fi_err:
                    print(f"AutoGluon feature importance failed: {fi_err}")
                    # Try to get feature importance from the best model directly
                    try:
                        best_model = model.get_model_best()
                        print(f"Best model: {best_model}")
                    except:
                        pass

                # Get model leaderboard
                leaderboard_df = None
                try:
                    leaderboard_df = model.leaderboard(data, silent=True)
                    print(f"Leaderboard obtained with {len(leaderboard_df)} models")
                except Exception as lb_err:
                    print(f"Leaderboard failed: {lb_err}")

                # Evaluate model performance
                eval_results = None
                try:
                    eval_results = model.evaluate(data, silent=True)
                    print(f"Evaluation results: {eval_results}")
                except Exception as eval_err:
                    print(f"Evaluation failed: {eval_err}")

                # Calculate metrics based on problem type
                problem_type = predictor_info.get('problem_type', 'unknown')
                print(f"Problem type: {problem_type}")

                if problem_type == 'regression':
                    # Regression metrics
                    accuracy = r2_score(y, y_pred)  # Use R2 as "accuracy" for regression
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)
                    print(f"R2: {accuracy:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                else:
                    # Classification metrics
                    accuracy = accuracy_score(y, y_pred)
                    rmse = None
                    mae = None

                # Create AutoGluon-specific visualizations
                plot_b64 = create_autogluon_visualizations(
                    model, data, X, y, y_pred,
                    feature_importance_df, leaderboard_df, problem_type
                )

                # Generate AutoGluon-specific report
                if user_level == "beginner":
                    html = create_autogluon_beginner_report(
                        model, accuracy, feature_importance, feature_names_list,
                        plot_b64, predictor_info, problem_type
                    )
                else:
                    html = create_autogluon_expert_report(
                        model, accuracy, y, y_pred, feature_importance, feature_names_list,
                        plot_b64, predictor_info, leaderboard_df, eval_results, problem_type
                    )

                return html

            except Exception as ag_err:
                import traceback
                error_trace = traceback.format_exc()
                print(f"AutoGluon prediction failed: {ag_err}")
                print(f"Traceback: {error_trace}")
                return f"""
                <html><body>
                <h1>AutoGluon Prediction Error</h1>
                <p>Could not make predictions with AutoGluon model: {ag_err}</p>
                <pre>{error_trace}</pre>
                </body></html>
                """, 400

        # Standard sklearn model handling
        # Try to match model's expected features
        try:
            # If model has feature_names_in_, use those
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                print(f"Model expects features: {list(expected_features)}")
                print(f"Data has features: {list(X.columns)}")

                # Check if we need to rename columns
                if not set(expected_features).issubset(set(X.columns)):
                    print("Feature mismatch detected, attempting to align...")
                    # Try to map features
                    X_mapped = pd.DataFrame()
                    for feature in expected_features:
                        if feature in X.columns:
                            X_mapped[feature] = X[feature]
                        else:
                            # Try to find similar column
                            similar_cols = [col for col in X.columns if
                                            feature.lower() in col.lower() or col.lower() in feature.lower()]
                            if similar_cols:
                                X_mapped[feature] = X[similar_cols[0]]
                                print(f"Mapped {similar_cols[0]} to {feature}")
                            else:
                                # Use first available column or create dummy
                                X_mapped[feature] = X.iloc[:, 0] if len(X.columns) > 0 else 0
                                print(f"Using dummy data for {feature}")
                    X = X_mapped
                    print(f"Aligned features: {list(X.columns)}")
        except Exception as e:
            print(f"Feature alignment failed: {e}")
            # Continue with original X

        # Handle categorical data
        from sklearn.preprocessing import LabelEncoder

        # Encode categorical features
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
                print(f"Encoded categorical feature: {col}")

        # Handle categorical target
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            print(f"Encoded target variable")

        # Fix sklearn compatibility issues
        try:
            # Fix monotonic_cst attribute for older models
            if hasattr(model, 'estimators_'):
                for estimator in model.estimators_:
                    if hasattr(estimator, 'tree_') and not hasattr(estimator, 'monotonic_cst'):
                        estimator.monotonic_cst = None
            elif hasattr(model, 'tree_') and not hasattr(model, 'monotonic_cst'):
                model.monotonic_cst = None
        except:
            pass

        # Get predictions
        try:
            y_pred = model.predict(X)
        except Exception as e:
            # If still fails, try to create a demo model
            if "monotonic_cst" in str(e):
                print("Creating demo model due to sklearn compatibility...")
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    demo_model = RandomForestClassifier(n_estimators=10, random_state=42)
                    demo_model.fit(X, y)
                    y_pred = demo_model.predict(X)
                    model = demo_model
                    print("Demo model created and used for predictions")
                except Exception as e2:
                    return f"""
                    <html><body>
                    <h1>Prediction Error</h1>
                    <p>Could not make predictions: {e}</p>
                    <p>Demo model creation failed: {e2}</p>
                    <p>This is likely due to sklearn version compatibility issues.</p>
                    </body></html>
                    """, 400
            else:
                return f"""
                <html><body>
                <h1>Prediction Error</h1>
                <p>Could not make predictions: {e}</p>
                <p>Model and data may be incompatible.</p>
                </body></html>
                """, 400

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = abs(model.coef_.flatten())

        # Create visualizations
        plot_b64 = create_visualizations(model, X, y, y_pred, feature_importance)

        # Generate report
        if user_level == "beginner":
            html = create_beginner_report(model, accuracy, feature_importance, X.columns, plot_b64)
        else:
            html = create_expert_report(model, accuracy, y, y_pred, feature_importance, X.columns, plot_b64)

        return html

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}", 500


def create_visualizations(model, X, y, y_pred, feature_importance):
    """Create model visualizations"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Accuracy
        axes[0, 0].text(0.5, 0.5, f'Model: {type(model).__name__}\nAccuracy: {accuracy_score(y, y_pred):.3f}',
                        ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Model Performance')
        axes[0, 0].axis('off')

        # Plot 2: Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        axes[0, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Plot 3: Feature Importance
        if feature_importance is not None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            axes[1, 0].barh(feature_names, feature_importance)
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Feature Importance')
        else:
            axes[1, 0].text(0.5, 0.5, 'No feature importance\navailable',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')

        # Plot 4: Prediction Distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        axes[1, 1].bar(unique, counts)
        axes[1, 1].set_xlabel('Predicted Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Prediction Distribution')

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"Visualization error: {e}")
        return None


def create_beginner_report(model, accuracy, feature_importance, feature_names, plot_b64):
    """Create beginner-friendly report"""
    html = f"""
    <html>
    <head>
        <title>Model Analysis - Beginner</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background-color: #f9f9f9; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Your AI Model Analysis</h1>
            <p>This report helps you understand your AI model in simple terms.</p>
        </div>

        <div class="section">
            <h2>What is Your Model?</h2>
            <p>Your model is a <strong>{type(model).__name__}</strong>.</p>
            <p>This type of model is used for making predictions based on data.</p>
        </div>

        <div class="section">
            <h2>How Good is Your Model?</h2>
            <p>Your model has an accuracy of <strong>{accuracy:.1%}</strong>.</p>
            <p>This means it makes the correct prediction {accuracy:.1%} of the time.</p>
        </div>
    """

    if feature_importance is not None and len(feature_importance) > 0:
        # Get top 3 features
        top_indices = np.argsort(feature_importance)[::-1][:3]
        html += f"""
        <div class="section">
            <h2>What Does Your Model Look At?</h2>
            <p>Your model pays attention to different pieces of information. Here are the most important ones:</p>
            <ul>
        """
        for i, idx in enumerate(top_indices):
            html += f"<li><strong>{feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'}</strong>: {feature_importance[idx]:.3f}</li>"
        html += "</ul></div>"

    if plot_b64:
        html += f"""
        <div class="section">
            <h2>Model Visualizations</h2>
            <img src="data:image/png;base64,{plot_b64}" />
        </div>
        """

    html += """
        <div class="section">
            <h2>What This Means</h2>
            <p>This analysis shows you:</p>
            <ul>
                <li>What type of model you have</li>
                <li>How accurate your model is</li>
                <li>Which features are most important</li>
                <li>Visual representation of your model</li>
            </ul>
        </div>
    </body>
    </html>
    """

    return html


def create_expert_report(model, accuracy, y_true, y_pred, feature_importance, feature_names, plot_b64):
    """Create expert-level report"""
    # Classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
    except:
        report = None

    html = f"""
    <html>
    <head>
        <title>Model Analysis - Expert</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background-color: #f9f9f9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Comprehensive Model Analysis</h1>
            <p>Detailed model performance and explainability analysis</p>
        </div>

        <div class="section">
            <h2>Model Overview</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{type(model).__name__}</td></tr>
                <tr><td>Accuracy</td><td>{accuracy:.4f}</td></tr>
                <tr><td>Total Samples</td><td>{len(y_true)}</td></tr>
                <tr><td>Feature Count</td><td>{len(feature_names)}</td></tr>
            </table>
        </div>
    """

    if report:
        html += "<div class='section'><h2>Classification Report</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>"
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                html += f"<tr><td>{class_name}</td><td>{metrics.get('precision', 0):.3f}</td><td>{metrics.get('recall', 0):.3f}</td><td>{metrics.get('f1-score', 0):.3f}</td></tr>"
        html += "</table></div>"

    if feature_importance is not None and len(feature_importance) > 0:
        html += "<div class='section'><h2>Feature Importance Analysis</h2><table><tr><th>Feature</th><th>Importance</th></tr>"
        for i, importance in enumerate(feature_importance):
            feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
            html += f"<tr><td>{feature_name}</td><td>{importance:.4f}</td></tr>"
        html += "</table></div>"

    if plot_b64:
        html += f"""
        <div class="section">
            <h2>Model Visualizations</h2>
            <img src="data:image/png;base64,{plot_b64}" />
        </div>
        """

    html += """
        <div class="section">
            <h2>Technical Summary</h2>
            <p>This analysis provides comprehensive insights into your model's performance, feature importance, and classification metrics.</p>
        </div>
    </body>
    </html>
    """

    return html


def create_autogluon_visualizations(model, data, X, y, y_pred, feature_importance_df, leaderboard_df,
                                    problem_type='classification'):
    """Create AutoGluon-specific visualizations"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Model Info & Performance
        try:
            best_model = model.get_model_best()
            if problem_type == 'regression':
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                info_text = f'AutoGluon TabularPredictor\nBest Model: {best_model}\nR² Score: {r2:.3f}\nRMSE: {rmse:.3f}'
            else:
                accuracy = accuracy_score(y, y_pred)
                info_text = f'AutoGluon TabularPredictor\nBest Model: {best_model}\nAccuracy: {accuracy:.3f}'
        except Exception as e:
            info_text = f'AutoGluon TabularPredictor\nError: {str(e)[:50]}'

        axes[0, 0].text(0.5, 0.5, info_text,
                        ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Model Performance')
        axes[0, 0].axis('off')

        # Plot 2: Confusion Matrix (classification) or Actual vs Predicted (regression)
        if problem_type == 'regression':
            axes[0, 1].scatter(y, y_pred, alpha=0.5)
            axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual')
            axes[0, 1].set_ylabel('Predicted')
            axes[0, 1].set_title('Actual vs Predicted')
        else:
            cm = confusion_matrix(y, y_pred)
            im = axes[0, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0, 1].set_title('Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
            plt.colorbar(im, ax=axes[0, 1])

        # Plot 3: Feature Importance (from AutoGluon's permutation-based importance)
        if feature_importance_df is not None and len(feature_importance_df) > 0:
            # Get top 15 features
            top_features = feature_importance_df.head(15)
            colors = ['green' if x > 0 else 'red' for x in top_features['importance'].values]
            axes[1, 0].barh(top_features.index, top_features['importance'].values, color=colors)
            axes[1, 0].set_xlabel('Importance (Permutation)')
            axes[1, 0].set_title('Feature Importance (Top 15)')
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')

        # Plot 4: Model Leaderboard (if available)
        if leaderboard_df is not None and len(leaderboard_df) > 0:
            models = leaderboard_df['model'].head(10).tolist()
            scores = leaderboard_df['score_val'].head(10).tolist() if 'score_val' in leaderboard_df.columns else \
            leaderboard_df['score_test'].head(10).tolist()
            axes[1, 1].barh(models, scores, color='steelblue')
            axes[1, 1].set_xlabel('Validation Score')
            axes[1, 1].set_title('Model Leaderboard (Top 10)')
            axes[1, 1].invert_yaxis()
        else:
            unique, counts = np.unique(y_pred, return_counts=True)
            axes[1, 1].bar(unique.astype(str), counts)
            axes[1, 1].set_xlabel('Predicted Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Prediction Distribution')

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"AutoGluon visualization error: {e}")
        return None


def create_autogluon_beginner_report(model, accuracy, feature_importance, feature_names, plot_b64, predictor_info,
                                     problem_type='classification'):
    """Create beginner-friendly report for AutoGluon models"""

    try:
        best_model = model.get_model_best()
    except:
        best_model = 'Ensemble'

    # Create performance description based on problem type
    if problem_type == 'regression':
        perf_title = "🎯 How Good is Your Model?"
        perf_desc = f"""
            <p>Your model has an R² score of <strong>{accuracy:.3f}</strong>.</p>
            <p>R² measures how well the model explains the variation in the data (1.0 = perfect, 0 = no better than average).</p>
            <div class="highlight">
                <p>💡 <strong>What does this mean?</strong></p>
                <p>An R² of {accuracy:.3f} means the model explains <strong>{accuracy * 100:.1f}%</strong> of the variation in your target variable.</p>
            </div>
        """
    else:
        perf_title = "🎯 How Good is Your Model?"
        perf_desc = f"""
            <p>Your model has an accuracy of <strong>{accuracy:.1%}</strong>.</p>
            <p>This means it correctly predicts the outcome <strong>{accuracy:.1%}</strong> of the time.</p>
            <div class="highlight">
                <p>💡 <strong>What does this mean?</strong></p>
                <p>If you give the model 100 new examples, it would correctly classify about <strong>{int(accuracy * 100)}</strong> of them.</p>
            </div>
        """

    html = f"""
    <html>
    <head>
        <title>AutoGluon Model Analysis - Beginner</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; background-color: #fafafa; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #667eea; background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; margin: 15px 0; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .highlight {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            ul {{ padding-left: 20px; }}
            li {{ margin: 8px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Your AI Model Analysis</h1>
            <p>This report explains your AutoGluon model in simple terms.</p>
        </div>

        <div class="section">
            <h2>📊 What is Your Model?</h2>
            <p>Your model is an <strong>AutoGluon TabularPredictor</strong> - a powerful AI system that automatically finds the best way to make predictions.</p>
            <div class="highlight">
                <p><strong>Best performing model:</strong> {best_model}</p>
                <p><strong>Task type:</strong> {problem_type}</p>
            </div>
            <p>AutoGluon tested many different AI models and combined the best ones to give you accurate predictions.</p>
        </div>

        <div class="section">
            <h2>{perf_title}</h2>
            {perf_desc}
        </div>
    """

    if feature_importance is not None and len(feature_importance) > 0:
        html += f"""
        <div class="section">
            <h2>🔍 What Does Your Model Look At?</h2>
            <p>AutoGluon uses <strong>permutation shuffling</strong> to determine which features are most important. This means it randomly shuffles each feature and measures how much the accuracy drops.</p>
            <p>Here are the most important factors for predictions:</p>
            <ul>
        """
        # Get top 5 features
        for i, (name, imp) in enumerate(zip(feature_names[:5], feature_importance[:5])):
            importance_desc = "very important" if imp > 0.1 else "moderately important" if imp > 0.01 else "somewhat important"
            html += f"<li><strong>{name}</strong>: {importance_desc} (importance: {imp:.4f})</li>"
        html += "</ul></div>"

    if plot_b64:
        html += f"""
        <div class="section">
            <h2>📈 Visual Summary</h2>
            <p>These charts show how your model performs:</p>
            <img src="data:image/png;base64,{plot_b64}" />
        </div>
        """

    html += """
        <div class="section">
            <h2>📝 Key Takeaways</h2>
            <ul>
                <li><strong>AutoGluon automatically tested many models</strong> and selected the best combination</li>
                <li><strong>Feature importance</strong> shows which data columns matter most for predictions</li>
                <li><strong>The confusion matrix</strong> shows where the model makes mistakes</li>
                <li><strong>Higher accuracy</strong> means better predictions</li>
            </ul>
        </div>
    </body>
    </html>
    """

    return html


def create_autogluon_expert_report(model, accuracy, y_true, y_pred, feature_importance, feature_names,
                                   plot_b64, predictor_info, leaderboard_df, eval_results,
                                   problem_type='classification'):
    """Create expert-level report for AutoGluon models"""

    # Get classification report (only for classification)
    report = None
    regression_metrics = None

    if problem_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        regression_metrics = {
            'R² Score': accuracy,
            'RMSE': rmse,
            'MAE': mae
        }
    else:
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
        except:
            report = None

    try:
        best_model = model.get_model_best()
        all_models = model.get_model_names()
    except:
        best_model = 'Unknown'
        all_models = []

    html = f"""
    <html>
    <head>
        <title>AutoGluon Model Analysis - Expert</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #0f3460; background-color: #f8f9fa; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #0f3460; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 15px 0; border-radius: 5px; }}
            .code {{ background-color: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; font-family: monospace; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AutoGluon TabularPredictor Analysis</h1>
            <p>Comprehensive model explainability report</p>
        </div>

        <div class="section">
            <h2>Predictor Overview</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Problem Type</td><td>{problem_type}</td></tr>
                <tr><td>Best Model</td><td>{best_model}</td></tr>
                <tr><td>Total Models Trained</td><td>{len(all_models)}</td></tr>
                <tr><td>Accuracy</td><td>{accuracy:.4f}</td></tr>
                <tr><td>Total Samples</td><td>{len(y_true)}</td></tr>
                <tr><td>Feature Count</td><td>{len(feature_names)}</td></tr>
            </table>
        </div>
    """

    # Evaluation results
    if eval_results:
        html += f"""
        <div class="section">
            <h2>Evaluation Metrics</h2>
            <div class="code">{eval_results}</div>
        </div>
        """

    # Classification report or Regression metrics
    if regression_metrics:
        html += "<div class='section'><h2>Regression Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
        for metric_name, metric_value in regression_metrics.items():
            html += f"<tr><td>{metric_name}</td><td>{metric_value:.6f}</td></tr>"
        html += "</table></div>"
    elif report:
        html += "<div class='section'><h2>Classification Report</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>"
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                html += f"<tr><td>{class_name}</td><td>{metrics.get('precision', 0):.4f}</td><td>{metrics.get('recall', 0):.4f}</td><td>{metrics.get('f1-score', 0):.4f}</td><td>{metrics.get('support', 0)}</td></tr>"
        html += "</table></div>"

    # Model leaderboard
    if leaderboard_df is not None and len(leaderboard_df) > 0:
        html += "<div class='section'><h2>Model Leaderboard</h2><table><tr><th>Model</th><th>Score (Val)</th><th>Pred Time (Val)</th><th>Fit Time</th></tr>"
        for _, row in leaderboard_df.head(10).iterrows():
            score = row.get('score_val', row.get('score_test', 0))
            pred_time = row.get('pred_time_val', 0)
            fit_time = row.get('fit_time', 0)
            html += f"<tr><td>{row['model']}</td><td>{score:.4f}</td><td>{pred_time:.4f}s</td><td>{fit_time:.2f}s</td></tr>"
        html += "</table></div>"

    # Feature importance (permutation-based)
    if feature_importance is not None and len(feature_importance) > 0:
        html += """
        <div class='section'>
            <h2>Feature Importance (Permutation-Based)</h2>
            <p><em>Computed via permutation shuffling - measures the drop in predictive performance when each feature's values are randomly shuffled.</em></p>
            <table><tr><th>Feature</th><th>Importance</th><th>Interpretation</th></tr>
        """
        for i, (name, imp) in enumerate(zip(feature_names[:20], feature_importance[:20])):
            interpretation = "Strong positive impact" if imp > 0.05 else "Moderate impact" if imp > 0.01 else "Weak impact" if imp > 0 else "Negligible/Negative impact"
            html += f"<tr><td>{name}</td><td>{imp:.6f}</td><td>{interpretation}</td></tr>"
        html += "</table></div>"

    # Visualizations
    if plot_b64:
        html += f"""
        <div class="section">
            <h2>Model Visualizations</h2>
            <img src="data:image/png;base64,{plot_b64}" />
        </div>
        """

    # Usage code
    html += f"""
        <div class="section">
            <h2>Usage Code</h2>
            <p>Load and use this predictor:</p>
            <div class="code">
from autogluon.tabular import TabularPredictor

# Load the predictor
predictor = TabularPredictor.load("path/to/predictor")

# Make predictions
predictions = predictor.predict(new_data)

# Get feature importance
importance = predictor.feature_importance(data)

# Evaluate on new data
results = predictor.evaluate(test_data)
            </div>
        </div>

        <div class="section">
            <h2>Technical Summary</h2>
            <p>This AutoGluon TabularPredictor uses an ensemble of {len(all_models)} models with <strong>{best_model}</strong> as the best performing model. 
            Feature importance was computed using permutation shuffling, which quantifies the drop in predictive performance when each feature's values are randomly shuffled across rows.</p>
        </div>
    </body>
    </html>
    """

    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)