"""
Data Interpretability Service.

Provides comprehensive data analysis and visualization capabilities:
- Data profiling (shape, dtypes, memory usage)
- Missing value analysis
- Outlier detection (Z-score based)
- Correlation analysis
- Distribution analysis
- Beginner/Expert report generation
"""

import io
import base64
import warnings
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


class DataInterpretabilityService:
    """
    Service for comprehensive data analysis and interpretation.
    
    Generates detailed reports with visualizations for any CSV dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the service with a DataFrame.
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df
        self.df_processed = None
        self.column_info = {}
        self.data_info = {}
        self.analysis_results = {}
        self.plots = {}
        
        # Run initial processing
        self._initialize()
    
    def _initialize(self):
        """Initialize data info and preprocessing."""
        self.data_info = self._get_data_info()
        self.df_processed, self.column_info = self._preprocess_data()
        self.analysis_results = self._perform_analysis()
        self.plots = self._generate_visualizations()
    
    def _get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {str(k): str(v) for k, v in self.df.dtypes.to_dict().items()},
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
    
    def _preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """Preprocess data and detect column types."""
        df_processed = self.df.copy()
        column_info = {}
        
        for col in df_processed.columns:
            col_info = {
                'original_dtype': str(df_processed[col].dtype),
                'null_count': int(df_processed[col].isnull().sum()),
                'null_percentage': float((df_processed[col].isnull().sum() / len(df_processed)) * 100),
                'unique_count': int(df_processed[col].nunique()),
                'type': 'unknown'
            }
            
            # Detect column type
            if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info['type'] = 'numeric'
                col_info['min'] = float(df_processed[col].min()) if not pd.isna(df_processed[col].min()) else None
                col_info['max'] = float(df_processed[col].max()) if not pd.isna(df_processed[col].max()) else None
                col_info['mean'] = float(df_processed[col].mean()) if not pd.isna(df_processed[col].mean()) else None
                col_info['std'] = float(df_processed[col].std()) if not pd.isna(df_processed[col].std()) else None
            elif df_processed[col].dtype == 'object':
                # Check if it's actually numeric but stored as string
                try:
                    pd.to_numeric(df_processed[col], errors='raise')
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    col_info['type'] = 'numeric'
                    col_info['min'] = float(df_processed[col].min()) if not pd.isna(df_processed[col].min()) else None
                    col_info['max'] = float(df_processed[col].max()) if not pd.isna(df_processed[col].max()) else None
                    col_info['mean'] = float(df_processed[col].mean()) if not pd.isna(df_processed[col].mean()) else None
                    col_info['std'] = float(df_processed[col].std()) if not pd.isna(df_processed[col].std()) else None
                except:
                    non_null = df_processed[col].dropna().astype(str)
                    avg_chars = float(non_null.str.len().mean()) if len(non_null) else 0.0
                    avg_words = float(non_null.str.split().str.len().mean()) if len(non_null) else 0.0
                    unique_ratio = float(non_null.nunique() / len(non_null)) if len(non_null) else 0.0
                    if avg_chars >= 30 and avg_words >= 5 and unique_ratio >= 0.5:
                        col_info['type'] = 'text'
                        col_info['avg_characters'] = avg_chars
                        col_info['avg_words'] = avg_words
                        col_info['duplicate_count'] = int(non_null.duplicated().sum())
                    else:
                        col_info['type'] = 'categorical'
                        col_info['top_values'] = {str(k): int(v) for k, v in df_processed[col].value_counts().head(5).to_dict().items()}
            elif df_processed[col].dtype == 'bool':
                col_info['type'] = 'boolean'
            elif 'datetime' in str(df_processed[col].dtype):
                col_info['type'] = 'datetime'
            else:
                col_info['type'] = 'categorical'
                col_info['top_values'] = {str(k): int(v) for k, v in df_processed[col].value_counts().head(5).to_dict().items()}
            
            column_info[col] = col_info
        
        return df_processed, column_info
    
    def _perform_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis."""
        results = {}
        
        # Basic statistics
        numeric_cols = [col for col, info in self.column_info.items() if info['type'] == 'numeric']
        categorical_cols = [col for col, info in self.column_info.items() if info['type'] == 'categorical']
        
        if numeric_cols:
            results['numeric_summary'] = self.df_processed[numeric_cols].describe()
            correlation_cols = self._correlation_numeric_columns(numeric_cols)
            if len(correlation_cols) >= 2:
                results['correlation_matrix'] = self.df_processed[correlation_cols].corr()
        
        if categorical_cols:
            results['categorical_summary'] = {}
            for col in categorical_cols:
                results['categorical_summary'][col] = self.df_processed[col].value_counts().head(10)
        
        # Outlier detection for numeric columns
        if numeric_cols:
            outliers = {}
            for col in numeric_cols:
                if not self.df_processed[col].isnull().all():
                    try:
                        z_scores = np.abs(zscore(self.df_processed[col].dropna()))
                        outlier_indices = np.where(z_scores > 2)[0]
                        outliers[col] = {
                            'count': int(len(outlier_indices)),
                            'percentage': float((len(outlier_indices) / len(self.df_processed[col].dropna())) * 100)
                        }
                    except:
                        outliers[col] = {'count': 0, 'percentage': 0.0}
            results['outliers'] = outliers
        
        return results

    @staticmethod
    def _correlation_numeric_columns(numeric_cols: List[str]) -> List[str]:
        """Exclude generated target labels from correlation matrices."""
        return [col for col in numeric_cols if str(col).strip().lower() != 'label']
    
    def _plot_to_base64(self, plot_func) -> Optional[str]:
        """Convert plot to base64 string."""
        try:
            buf = io.BytesIO()
            plot_func()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"Plot generation failed: {e}")
            plt.close()
            return None
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate various visualizations."""
        plots = {}
        
        numeric_cols = [col for col, info in self.column_info.items() if info['type'] == 'numeric']
        categorical_cols = [col for col, info in self.column_info.items() if info['type'] == 'categorical']
        
        # Distribution plots for numeric columns
        if numeric_cols:
            plot = self._plot_to_base64(lambda: self._plot_numeric_distributions(numeric_cols))
            if plot:
                plots['numeric_distributions'] = plot
            
            correlation_cols = self._correlation_numeric_columns(numeric_cols)
            if len(correlation_cols) >= 2:
                plot = self._plot_to_base64(lambda: self._plot_correlation_heatmap(correlation_cols))
                if plot:
                    plots['correlation_heatmap'] = plot
        
        # Categorical plots
        if categorical_cols:
            plot = self._plot_to_base64(lambda: self._plot_categorical_distributions(categorical_cols))
            if plot:
                plots['categorical_distributions'] = plot
        
        return plots
    
    def _plot_numeric_distributions(self, numeric_cols: List[str]):
        """Plot distributions of numeric columns."""
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                self.df_processed[col].hist(bins=30, ax=axes[i], alpha=0.7, color='steelblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
    
    def _plot_correlation_heatmap(self, numeric_cols: List[str]):
        """Plot correlation heatmap."""
        if len(numeric_cols) < 2:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Need at least 2 numeric columns\nfor correlation analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Correlation Analysis')
            return
        
        corr_matrix = self.df_processed[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
    
    def _plot_categorical_distributions(self, categorical_cols: List[str]):
        """Plot distributions of categorical columns."""
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                value_counts = self.df_processed[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
                axes[i].set_title(f'Top Values in {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic data information."""
        return self.data_info
    
    def get_column_info(self) -> Dict[str, Dict]:
        """Get column type information."""
        return self.column_info
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return self.analysis_results
    
    def generate_beginner_report(self) -> str:
        """Create beginner-friendly HTML report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report - Beginner</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            margin: 40px; 
            line-height: 1.6; 
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 8px; 
            margin-bottom: 30px; 
        }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .section {{ 
            margin: 25px 0; 
            padding: 20px; 
            border-left: 4px solid #667eea; 
            background-color: #f9f9f9; 
            border-radius: 0 8px 8px 0;
        }}
        .section h2 {{ color: #667eea; margin-bottom: 15px; }}
        .section h3 {{ color: #764ba2; margin: 20px 0 10px 0; }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            margin: 15px 0; 
            border: 1px solid #ddd; 
            border-radius: 8px;
            display: block;
        }}
        .highlight {{ 
            background-color: #e3f2fd; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0;
            border-left: 4px solid #2196F3;
        }}
        ul {{ margin-left: 20px; margin-top: 10px; }}
        li {{ margin: 8px 0; }}
        strong {{ color: #667eea; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Your Data Analysis Report</h1>
            <p>This report helps you understand your data in simple terms.</p>
        </div>
        
        <div class="section">
            <h2>📋 What's in Your Data?</h2>
            <p>Your dataset has <strong>{self.data_info['shape'][0]:,}</strong> rows and <strong>{self.data_info['shape'][1]}</strong> columns.</p>
            <p>This means you have {self.data_info['shape'][0]:,} individual records, each with {self.data_info['shape'][1]} different pieces of information.</p>
            <div class="highlight">
                <p><strong>Memory Usage:</strong> {self.data_info['memory_usage']:.2f} MB</p>
            </div>
        </div>
"""
        
        # Add numeric columns info
        numeric_cols = [col for col, info in self.column_info.items() if info['type'] == 'numeric']
        if numeric_cols:
            html += f"""
        <div class="section">
            <h2>🔢 Numbers in Your Data</h2>
            <p>We found <strong>{len(numeric_cols)}</strong> columns with numbers: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}</p>
            <p>These numbers help us understand patterns and relationships in your data.</p>
"""
            
            if 'numeric_distributions' in self.plots:
                html += f'<h3>📈 How Your Numbers Are Spread Out</h3><img src="data:image/png;base64,{self.plots["numeric_distributions"]}" alt="Numeric Distributions" />'
            
            if 'correlation_heatmap' in self.plots:
                html += f'<h3>🔗 How Your Numbers Relate to Each Other</h3><p>Colors show relationships: <strong>red = positive</strong> (when one goes up, the other goes up), <strong>blue = negative</strong> (when one goes up, the other goes down)</p><img src="data:image/png;base64,{self.plots["correlation_heatmap"]}" alt="Correlation Heatmap" />'
            
            html += "</div>"
        
        # Add categorical columns info
        categorical_cols = [col for col, info in self.column_info.items() if info['type'] == 'categorical']
        if categorical_cols:
            html += f"""
        <div class="section">
            <h2>📝 Categories in Your Data</h2>
            <p>We found <strong>{len(categorical_cols)}</strong> columns with categories: {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}</p>
            <p>These help us group and compare different types of information.</p>
"""
            
            if 'categorical_distributions' in self.plots:
                html += f'<h3>📊 Most Common Categories</h3><img src="data:image/png;base64,{self.plots["categorical_distributions"]}" alt="Categorical Distributions" />'
            
            html += "</div>"
        
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
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_expert_report(self) -> str:
        """Create expert-level HTML report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report - Expert</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            margin: 40px; 
            line-height: 1.6;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 8px; 
            margin-bottom: 30px; 
        }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .section {{ 
            margin: 25px 0; 
            padding: 20px; 
            border-left: 4px solid #667eea; 
            background-color: #f9f9f9;
            border-radius: 0 8px 8px 0;
        }}
        .section h2 {{ color: #667eea; margin-bottom: 15px; font-size: 1.5em; }}
        .section h3 {{ color: #764ba2; margin: 20px 0 10px 0; }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            margin: 15px 0; 
            border: 1px solid #ddd;
            border-radius: 8px;
            display: block;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #e0e0e0; 
            padding: 10px 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #667eea; 
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4f8; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{ font-size: 0.9em; color: #666; }}
        .metric-card .value {{ font-size: 1.5em; font-weight: bold; color: #667eea; }}
        .warning {{ 
            background: #fff3cd; 
            border-left: 4px solid #ff9800; 
            padding: 15px; 
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comprehensive Data Analysis Report</h1>
            <p>Detailed statistical analysis and insights</p>
        </div>
        
        <div class="section">
            <h2>📋 Dataset Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Rows</div>
                    <div class="value">{self.data_info['shape'][0]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Columns</div>
                    <div class="value">{self.data_info['shape'][1]}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Memory Usage</div>
                    <div class="value">{self.data_info['memory_usage']:.2f} MB</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Column Analysis</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Dtype</th>
                    <th>Null Count</th>
                    <th>Null %</th>
                    <th>Unique Values</th>
                </tr>
"""
        
        for col, info in self.column_info.items():
            null_warning = ' ⚠️' if info['null_percentage'] > 10 else ''
            html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td>{info['type']}</td>
                    <td>{info['original_dtype']}</td>
                    <td>{info['null_count']}</td>
                    <td>{info['null_percentage']:.1f}%{null_warning}</td>
                    <td>{info['unique_count']}</td>
                </tr>
"""
        
        html += "</table></div>"
        
        # Add numeric analysis
        numeric_cols = [col for col, info in self.column_info.items() if info['type'] == 'numeric']
        if numeric_cols and 'numeric_summary' in self.analysis_results:
            html += f"""
        <div class="section">
            <h2>📊 Numeric Statistics</h2>
            {self.analysis_results['numeric_summary'].to_html(classes='table', float_format='%.3f', border=0)}
        </div>
"""
            
            if 'numeric_distributions' in self.plots:
                html += f'<div class="section"><h3>Distribution Analysis</h3><img src="data:image/png;base64,{self.plots["numeric_distributions"]}" alt="Numeric Distributions" /></div>'
            
            if 'correlation_heatmap' in self.plots:
                html += f'<div class="section"><h3>Correlation Matrix</h3><img src="data:image/png;base64,{self.plots["correlation_heatmap"]}" alt="Correlation Heatmap" /></div>'
        
        # Add categorical analysis
        if 'categorical_summary' in self.analysis_results and self.analysis_results['categorical_summary']:
            html += "<div class='section'><h2>📝 Categorical Analysis</h2>"
            for col, summary in self.analysis_results['categorical_summary'].items():
                html += f"<h3>{col}</h3>{summary.to_frame().to_html(classes='table', border=0)}"
            html += "</div>"
            
            if 'categorical_distributions' in self.plots:
                html += f'<div class="section"><h3>Categorical Distributions</h3><img src="data:image/png;base64,{self.plots["categorical_distributions"]}" alt="Categorical Distributions" /></div>'
        
        # Add outlier analysis
        if 'outliers' in self.analysis_results and self.analysis_results['outliers']:
            html += "<div class='section'><h2>🚨 Outlier Detection (Z-score > 2)</h2><table><tr><th>Column</th><th>Outlier Count</th><th>Percentage</th></tr>"
            for col, outlier_info in self.analysis_results['outliers'].items():
                warning = ' ⚠️' if outlier_info['percentage'] > 5 else ''
                html += f"<tr><td>{col}</td><td>{outlier_info['count']}</td><td>{outlier_info['percentage']:.1f}%{warning}</td></tr>"
            html += "</table></div>"
        
        html += """
        <div class="section">
            <h2>💡 Technical Summary</h2>
            <p>This analysis provides comprehensive insights into your dataset including:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>Statistical summaries for all numeric variables</li>
                <li>Distribution analysis and outlier detection (Z-score method)</li>
                <li>Pearson correlation analysis between numeric features</li>
                <li>Categorical variable frequency analysis</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_report(self, user_level: str = "expert") -> str:
        """
        Generate HTML report based on user level.
        
        Args:
            user_level: 'beginner' or 'expert'
            
        Returns:
            HTML report string
        """
        if user_level.lower() == "beginner":
            return self.generate_beginner_report()
        return self.generate_expert_report()
