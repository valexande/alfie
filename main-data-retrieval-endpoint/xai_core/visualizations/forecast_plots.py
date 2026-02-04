"""
Time series forecast visualization functions.
"""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from xai_core.utils import fig_to_base64

warnings.filterwarnings('ignore')


def plot_forecast(
    forecasts: pd.DataFrame,
    title: str = "Time Series Forecast"
) -> Optional[str]:
    """
    Create forecast visualization with uncertainty bands.
    
    Args:
        forecasts: DataFrame with forecast values (may include quantiles)
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Find forecast columns
        mean_col = None
        quantile_cols = []
        
        for col in forecasts.columns:
            col_str = str(col).lower()
            if 'mean' in col_str:
                mean_col = col
            elif any(q in col_str for q in ['0.1', '0.5', '0.9', 'q0.', 'quantile']):
                quantile_cols.append(col)
        
        # Plot mean forecast
        if mean_col is not None:
            ax.plot(
                forecasts.index, 
                forecasts[mean_col], 
                'b-', 
                linewidth=2, 
                label='Mean Forecast'
            )
        elif len(quantile_cols) > 0:
            # Use median or first quantile
            median_col = None
            for col in quantile_cols:
                if '0.5' in str(col).lower():
                    median_col = col
                    break
            
            if median_col:
                ax.plot(
                    forecasts.index, 
                    forecasts[median_col], 
                    'b-', 
                    linewidth=2, 
                    label='Median Forecast'
                )
            else:
                ax.plot(
                    forecasts.index, 
                    forecasts[quantile_cols[0]], 
                    'b-', 
                    linewidth=2, 
                    label='Forecast'
                )
        else:
            # Use first numeric column
            numeric_cols = forecasts.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.plot(
                    forecasts.index, 
                    forecasts[numeric_cols[0]], 
                    'b-', 
                    linewidth=2, 
                    label='Forecast'
                )
        
        # Plot uncertainty bands if available
        lower_col = None
        upper_col = None
        
        for col in quantile_cols:
            col_str = str(col).lower()
            if '0.1' in col_str or 'lower' in col_str:
                lower_col = col
            elif '0.9' in col_str or 'upper' in col_str:
                upper_col = col
        
        if lower_col and upper_col:
            ax.fill_between(
                forecasts.index,
                forecasts[lower_col],
                forecasts[upper_col],
                alpha=0.2,
                color='b',
                label='90% CI'
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Forecast plot failed: {e}")
        return None


def plot_forecast_comparison(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Forecast vs Actual"
) -> Optional[str]:
    """
    Create comparison plot of forecast vs actual values.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        title: Plot title
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(actual.index, actual.values, 'k-', linewidth=2, label='Actual')
        ax.plot(forecast.index, forecast.values, 'b--', linewidth=2, label='Forecast')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Forecast comparison plot failed: {e}")
        return None
