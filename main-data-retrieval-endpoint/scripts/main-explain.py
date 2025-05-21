from flask import Flask, request, render_template_string
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

app = Flask(__name__)


@app.route('/explain', methods=['POST'])
def explain():
    print("✅ Request received")

    # Read uploaded files
    model_file = request.files['model_file']
    encoder_file = request.files['encoder_file']
    data_file = request.files['data_file']

    # Load files
    model = joblib.load(model_file)
    label_encoders = joblib.load(encoder_file)
    df = pd.read_csv(data_file)

    # Preprocess
    categorical_cols = ['gender', 'ethnicity', 'race']
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    X = df.drop(columns=['alert'])
    y = df['alert']

    # Predict
    predictions = model.predict(X)
    report_dict = classification_report(y, predictions, output_dict=True)
    report_html = pd.DataFrame(report_dict).T.to_html(classes="table table-bordered")

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_bar_b64 = plot_to_base64(lambda: shap.summary_plot(shap_values[1], X, plot_type="bar", show=False))
    shap_full_b64 = plot_to_base64(lambda: shap.summary_plot(shap_values[1], X, show=False))

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
        fairness_html_blocks.append(f"<h3>Fairness for {group.capitalize()}</h3>" + fairness_df.to_html(classes="table table-striped"))

    # Render everything in HTML
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


if __name__ == '__main__':
    app.run(debug=True)
