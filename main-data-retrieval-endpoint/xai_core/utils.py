"""
Utility functions for XAI Core.
"""

from typing import Any, Callable, Optional, List
import csv
import pandas as pd
import numpy as np

# csv's implicit 128-KiB field cap is much smaller than pandas' text capacity.
# Configure this process-wide parser setting once, not around concurrent requests.
# This is syntax validation capacity, NOT an upload/memory quota or sandbox.
csv.field_size_limit(max(csv.field_size_limit(), 2**31 - 1))


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


class InputValidationError(ValueError):
    """Invalid evaluation data or descriptive metadata supplied by the caller."""


_TARGET_NAMES = {
    'target', 'label', 'labels', 'class', 'y', 'outcome', 'prediction',
    'alert', 'target_variable', 'response', 'output', 'result',
}


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Return a single recognized target; unlabeled/ambiguous EDA stays unlabeled."""
    candidates = [c for c in df.columns if str(c).lower() in _TARGET_NAMES]
    return candidates[0] if len(candidates) == 1 else None


def resolve_target_column(
    df: pd.DataFrame,
    explicit_target: Optional[str] = None,
    model_label: Optional[str] = None,
    *,
    required: bool = True,
) -> Optional[str]:
    """Resolve evaluation target without guessing; model label is authoritative."""
    if df.empty:
        raise InputValidationError('Evaluation data cannot be empty')
    if df.columns.has_duplicates:
        raise InputValidationError('Duplicate column names are not allowed')
    for name in (explicit_target, model_label):
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise InputValidationError('Target column name cannot be empty')
    if explicit_target is not None and model_label is not None and explicit_target != model_label:
        raise InputValidationError(
            f"Explicit target '{explicit_target}' conflicts with model target '{model_label}'"
        )
    target = model_label if model_label is not None else explicit_target
    if target is None:
        target = detect_target_column(df)
        if target is None:
            if not required:
                return None
            raise InputValidationError(
                'Target is ambiguous or unrecognized; supply target_col explicitly'
            )
    if target not in df.columns:
        raise InputValidationError(f"Target column '{target}' not found in data")
    values = df[target]
    if values.isna().any() or values.map(lambda v: isinstance(v, str) and not v.strip()).any():
        raise InputValidationError(f"Target column '{target}' contains empty values")
    return target


def read_evaluation_csv(data: bytes) -> pd.DataFrame:
    """Validate CSV headers and record widths before pandas can infer an index."""
    import io
    try:
        text = data.decode('utf-8-sig')
        records = csv.reader(io.StringIO(text), strict=True)
        header = next(row for row in records if row)
        if len(header) != len(set(header)):
            raise InputValidationError('Duplicate column names are not allowed')
        if any(not c.strip() for c in header):
            raise InputValidationError('Column names cannot be empty')
        for row in records:
            if row and len(row) != len(header):
                raise InputValidationError(
                    f'CSV record ending at line {records.line_num} has {len(row)} fields; '
                    f'expected {len(header)} fields'
                )
        return pd.read_csv(io.StringIO(text))
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise InputValidationError(f'Invalid or empty CSV: {exc}') from exc


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
