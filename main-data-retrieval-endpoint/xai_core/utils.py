"""
Utility functions for XAI Core.
"""

from typing import Any, Callable, Optional, List
import pandas as pd
import numpy as np


def safe_compute(
    func: Callable, 
    default: Any = None,
    error_prefix: str = ""
) -> Any:
    """
    Execute function with error handling.
    
    Args:
        func: Function to execute
        default: Default value to return on error
        error_prefix: Prefix for error messages
        
    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        if error_prefix:
            print(f"Warning: {error_prefix}: {e}")
        return default


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect likely target column from common names.
    
    Args:
        df: DataFrame to search
        
    Returns:
        Target column name or None
    """
    common_names = [
        'target', 'label', 'class', 'y', 'outcome',
        'prediction', 'alert', 'target_variable',
        'response', 'output', 'result'
    ]
    
    # Case-insensitive search
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    for name in common_names:
        if name in df_columns_lower:
            return df_columns_lower[name]
    
    # Fallback: last column (common convention)
    if len(df.columns) > 0:
        return df.columns[-1]
    
    return None


def ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to numeric for visualization.
    
    Args:
        X: DataFrame with potential categorical columns
        
    Returns:
        DataFrame with all numeric columns
    """
    from sklearn.preprocessing import LabelEncoder
    
    X_numeric = X.copy()
    
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            try:
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
            except Exception:
                # Drop column if encoding fails
                X_numeric = X_numeric.drop(columns=[col])
    
    return X_numeric


def subsample_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    max_samples: int = 1000,
    random_state: int = 42
) -> tuple:
    """
    Subsample data for faster computation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        max_samples: Maximum number of samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_sample, y_sample)
    """
    if len(X) <= max_samples:
        return X, y
    
    np.random.seed(random_state)
    indices = np.random.choice(len(X), max_samples, replace=False)
    
    return X.iloc[indices], y.iloc[indices]


def fig_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64-encoded PNG string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64-encoded string
    """
    import io
    import base64
    import matplotlib.pyplot as plt
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> List[str]:
    """
    Validate DataFrame and return list of issues.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df is None:
        errors.append("DataFrame is None")
        return errors
    
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    if len(df.columns) == 0:
        errors.append("DataFrame has no columns")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        errors.append(f"Columns with all null values: {null_cols}")
    
    return errors
