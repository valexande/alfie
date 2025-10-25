from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come BEFORE pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "flexible-data-interpretability-api"}, 200

@app.route('/analyze-data', methods=['POST'])
def analyze_data():
    """Flexible data analysis endpoint for any CSV"""
    try:
        print("✅ Data analysis request received")

        user_level = request.form.get('user_level', 'expert').lower()
        print(f"User level: {user_level}")

        # Read uploaded CSV file
        csv_file = request.files['csv_file']
        print("CSV file received, loading...")

        # Load data
        df = pd.read_csv(csv_file)
        print(f"Data loaded successfully, shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Basic data info
        data_info = get_data_info(df)
        
        # Detect data types and preprocess
        df_processed, column_info = preprocess_data(df)
        
        # Generate analysis
        analysis_results = perform_analysis(df_processed, column_info)
        
        # Generate visualizations
        plots = generate_visualizations(df_processed, column_info)
        
        # Create report based on user level
        if user_level == "beginner":
            html_report = create_beginner_report(data_info, analysis_results, plots)
        else:
            html_report = create_expert_report(data_info, analysis_results, plots, df_processed, column_info)
        
        return html_report
    
    except Exception as e:
        print(f"❌ Error in analyze_data function: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

def get_data_info(df):
    """Get basic information about the dataset"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    return info

def preprocess_data(df):
    """Preprocess data and detect column types"""
    df_processed = df.copy()
    column_info = {}
    
    for col in df_processed.columns:
        col_info = {
            'original_dtype': str(df_processed[col].dtype),
            'null_count': df_processed[col].isnull().sum(),
            'null_percentage': (df_processed[col].isnull().sum() / len(df_processed)) * 100,
            'unique_count': df_processed[col].nunique(),
            'type': 'unknown'
        }
        
        # Detect column type
        if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            col_info['type'] = 'numeric'
            col_info['min'] = df_processed[col].min()
            col_info['max'] = df_processed[col].max()
            col_info['mean'] = df_processed[col].mean()
            col_info['std'] = df_processed[col].std()
        elif df_processed[col].dtype == 'object':
            # Check if it's actually numeric but stored as string
            try:
                pd.to_numeric(df_processed[col], errors='raise')
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                col_info['type'] = 'numeric'
                col_info['min'] = df_processed[col].min()
                col_info['max'] = df_processed[col].max()
                col_info['mean'] = df_processed[col].mean()
                col_info['std'] = df_processed[col].std()
            except:
                col_info['type'] = 'categorical'
                col_info['top_values'] = df_processed[col].value_counts().head(5).to_dict()
        elif df_processed[col].dtype == 'bool':
            col_info['type'] = 'boolean'
        elif 'datetime' in str(df_processed[col].dtype):
            col_info['type'] = 'datetime'
        else:
            col_info['type'] = 'categorical'
            col_info['top_values'] = df_processed[col].value_counts().head(5).to_dict()
        
        column_info[col] = col_info
    
    return df_processed, column_info

def perform_analysis(df, column_info):
    """Perform statistical analysis"""
    results = {}
    
    # Basic statistics
    numeric_cols = [col for col, info in column_info.items() if info['type'] == 'numeric']
    categorical_cols = [col for col, info in column_info.items() if info['type'] == 'categorical']
    
    if numeric_cols:
        results['numeric_summary'] = df[numeric_cols].describe()
        results['correlation_matrix'] = df[numeric_cols].corr()
    
    if categorical_cols:
        results['categorical_summary'] = {}
        for col in categorical_cols:
            results['categorical_summary'][col] = df[col].value_counts().head(10)
    
    # Outlier detection for numeric columns
    if numeric_cols:
        outliers = {}
        for col in numeric_cols:
            if not df[col].isnull().all():
                z_scores = np.abs(zscore(df[col].dropna()))
                outlier_indices = np.where(z_scores > 2)[0]
                outliers[col] = {
                    'count': len(outlier_indices),
                    'percentage': (len(outlier_indices) / len(df[col].dropna())) * 100
                }
        results['outliers'] = outliers
    
    return results

def generate_visualizations(df, column_info):
    """Generate various visualizations"""
    plots = {}
    
    numeric_cols = [col for col, info in column_info.items() if info['type'] == 'numeric']
    categorical_cols = [col for col, info in column_info.items() if info['type'] == 'categorical']
    
    # Distribution plots for numeric columns
    if numeric_cols:
        plots['numeric_distributions'] = plot_to_base64(lambda: plot_numeric_distributions(df, numeric_cols))
        plots['correlation_heatmap'] = plot_to_base64(lambda: plot_correlation_heatmap(df, numeric_cols))
    
    # Categorical plots
    if categorical_cols:
        plots['categorical_distributions'] = plot_to_base64(lambda: plot_categorical_distributions(df, categorical_cols))
    
    # Combined analysis
    if numeric_cols and len(numeric_cols) >= 2:
        plots['scatter_matrix'] = plot_to_base64(lambda: plot_scatter_matrix(df, numeric_cols))
    
    # Clustering if we have enough numeric columns
    if len(numeric_cols) >= 2:
        plots['clustering'] = plot_to_base64(lambda: plot_clustering(df, numeric_cols))
    
    return plots

def plot_to_base64(plot_func):
    """Convert plot to base64 string"""
    buf = io.BytesIO()
    plot_func()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_numeric_distributions(df, numeric_cols):
    """Plot distributions of numeric columns"""
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()

def plot_correlation_heatmap(df, numeric_cols):
    """Plot correlation heatmap"""
    if len(numeric_cols) < 2:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Need at least 2 numeric columns\nfor correlation analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Correlation Analysis')
        return
    
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')

def plot_categorical_distributions(df, categorical_cols):
    """Plot distributions of categorical columns"""
    n_cols = min(2, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Top Values in {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()

def plot_scatter_matrix(df, numeric_cols):
    """Plot scatter matrix for numeric columns"""
    if len(numeric_cols) < 2:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Need at least 2 numeric columns\nfor scatter matrix', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Scatter Matrix')
        return
    
    # Limit to 4 columns for readability
    cols_to_plot = numeric_cols[:4]
    pd.plotting.scatter_matrix(df[cols_to_plot], alpha=0.6, figsize=(12, 12))
    plt.suptitle('Scatter Matrix of Numeric Features', y=0.95)

def plot_clustering(df, numeric_cols):
    """Plot clustering analysis"""
    if len(numeric_cols) < 2:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Need at least 2 numeric columns\nfor clustering', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Clustering Analysis')
        return
    
    # Use first two numeric columns for 2D clustering
    X = df[numeric_cols[:2]].dropna()
    
    if len(X) < 4:  # Need at least 4 points for clustering
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Not enough data points\nfor clustering analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Clustering Analysis')
        return
    
    # Perform clustering
    n_clusters = min(4, len(X) // 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title(f'Clustering Analysis (k={n_clusters})')

def create_beginner_report(data_info, analysis_results, plots):
    """Create beginner-friendly HTML report"""
    html = f"""
    <html>
    <head>
        <title>Data Analysis Report - Beginner</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background-color: #f9f9f9; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
            .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Your Data Analysis Report</h1>
            <p>This report helps you understand your data in simple terms.</p>
        </div>
        
        <div class="section">
            <h2>📋 What's in Your Data?</h2>
            <p>Your dataset has <strong>{data_info['shape'][0]:,}</strong> rows and <strong>{data_info['shape'][1]}</strong> columns.</p>
            <p>This means you have {data_info['shape'][0]:,} individual records, each with {data_info['shape'][1]} different pieces of information.</p>
        </div>
    """
    
    # Add numeric columns info
    numeric_cols = list(analysis_results.get('numeric_summary', pd.DataFrame()).columns) if 'numeric_summary' in analysis_results else []
    if numeric_cols:
        html += f"""
        <div class="section">
            <h2>🔢 Numbers in Your Data</h2>
            <p>We found {len(numeric_cols)} columns with numbers: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}</p>
            <p>These numbers help us understand patterns and relationships in your data.</p>
        """
        
        if 'numeric_distributions' in plots:
            html += f'<h3>📈 How Your Numbers Are Spread Out</h3><img src="data:image/png;base64,{plots["numeric_distributions"]}" />'
        
        if 'correlation_heatmap' in plots:
            html += f'<h3>🔗 How Your Numbers Relate to Each Other</h3><img src="data:image/png;base64,{plots["correlation_heatmap"]}" />'
        
        html += "</div>"
    
    # Add categorical columns info
    categorical_cols = list(analysis_results.get('categorical_summary', {}).keys())
    if categorical_cols:
        html += f"""
        <div class="section">
            <h2>📝 Categories in Your Data</h2>
            <p>We found {len(categorical_cols)} columns with categories: {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}</p>
            <p>These help us group and compare different types of information.</p>
        """
        
        if 'categorical_distributions' in plots:
            html += f'<h3>📊 Most Common Categories</h3><img src="data:image/png;base64,{plots["categorical_distributions"]}" />'
        
        html += "</div>"
    
    # Add clustering if available
    if 'clustering' in plots:
        html += f"""
        <div class="section">
            <h2>🎯 Finding Patterns</h2>
            <p>We used special techniques to find groups of similar data points in your dataset.</p>
            <img src="data:image/png;base64,{plots['clustering']}" />
        </div>
        """
    
    html += """
        <div class="section">
            <h2>💡 What This Means</h2>
            <p>This analysis helps you understand:</p>
            <ul>
                <li>What types of information you have</li>
                <li>How your data is distributed</li>
                <li>Whether there are any unusual patterns</li>
                <li>How different parts of your data relate to each other</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

def create_expert_report(data_info, analysis_results, plots, df, column_info):
    """Create expert-level HTML report"""
    html = f"""
    <html>
    <head>
        <title>Data Analysis Report - Expert</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background-color: #f9f9f9; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .code {{ background-color: #f4f4f4; padding: 10px; border-radius: 3px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Comprehensive Data Analysis Report</h1>
            <p>Detailed statistical analysis and insights</p>
        </div>
        
        <div class="section">
            <h2>📋 Dataset Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Rows</td><td>{data_info['shape'][0]:,}</td></tr>
                <tr><td>Columns</td><td>{data_info['shape'][1]}</td></tr>
                <tr><td>Memory Usage</td><td>{data_info['memory_usage']:.2f} MB</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🔍 Column Analysis</h2>
            <table>
                <tr><th>Column</th><th>Type</th><th>Null Count</th><th>Null %</th><th>Unique Values</th></tr>
    """
    
    for col, info in column_info.items():
        html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{info['type']}</td>
                    <td>{info['null_count']}</td>
                    <td>{info['null_percentage']:.1f}%</td>
                    <td>{info['unique_count']}</td>
                </tr>
        """
    
    html += "</table></div>"
    
    # Add numeric analysis
    if 'numeric_summary' in analysis_results:
        html += f"""
        <div class="section">
            <h2>📊 Numeric Statistics</h2>
            {analysis_results['numeric_summary'].to_html(classes='table')}
        </div>
        """
        
        if 'numeric_distributions' in plots:
            html += f'<h3>Distribution Analysis</h3><img src="data:image/png;base64,{plots["numeric_distributions"]}" />'
        
        if 'correlation_heatmap' in plots:
            html += f'<h3>Correlation Matrix</h3><img src="data:image/png;base64,{plots["correlation_heatmap"]}" />'
    
    # Add categorical analysis
    if 'categorical_summary' in analysis_results:
        html += "<div class='section'><h2>📝 Categorical Analysis</h2>"
        for col, summary in analysis_results['categorical_summary'].items():
            html += f"<h3>{col}</h3>{summary.to_frame().to_html(classes='table')}"
        html += "</div>"
        
        if 'categorical_distributions' in plots:
            html += f'<h3>Categorical Distributions</h3><img src="data:image/png;base64,{plots["categorical_distributions"]}" />'
    
    # Add outlier analysis
    if 'outliers' in analysis_results:
        html += "<div class='section'><h2>🚨 Outlier Detection</h2><table><tr><th>Column</th><th>Outlier Count</th><th>Percentage</th></tr>"
        for col, outlier_info in analysis_results['outliers'].items():
            html += f"<tr><td>{col}</td><td>{outlier_info['count']}</td><td>{outlier_info['percentage']:.1f}%</td></tr>"
        html += "</table></div>"
    
    # Add advanced visualizations
    if 'scatter_matrix' in plots:
        html += f'<div class="section"><h2>🔗 Scatter Matrix</h2><img src="data:image/png;base64,{plots["scatter_matrix"]}" /></div>'
    
    if 'clustering' in plots:
        html += f'<div class="section"><h2>🎯 Clustering Analysis</h2><img src="data:image/png;base64,{plots["clustering"]}" /></div>'
    
    html += """
        <div class="section">
            <h2>💡 Technical Summary</h2>
            <p>This analysis provides comprehensive insights into your dataset including:</p>
            <ul>
                <li>Statistical summaries for all numeric variables</li>
                <li>Distribution analysis and outlier detection</li>
                <li>Correlation analysis between numeric features</li>
                <li>Categorical variable frequency analysis</li>
                <li>Clustering analysis to identify data patterns</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
