"""
AutoGluon TimeSeries Explainer - For TimeSeriesPredictor.

Handles time series forecasting with uncertainty quantification.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')

# Optional imports
try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_TS_AVAILABLE = True
except ImportError:
    AUTOGLUON_TS_AVAILABLE = False


class AutoGluonTimeSeriesExplainer(BaseModelExplainer):
    """
    Explainer for AutoGluon TimeSeriesPredictor.
    
    Time series forecasting is fundamentally different from
    classification/regression:
    - SHAP is not directly applicable
    - Feature importance has limited meaning
    - Focus on forecast visualization and uncertainty
    
    Example:
        >>> from autogluon.timeseries import TimeSeriesPredictor
        >>> predictor = TimeSeriesPredictor.load("my_model")
        >>> explainer = AutoGluonTimeSeriesExplainer(predictor, ts_data, None)
        >>> forecasts = explainer.generate_forecasts()
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        **kwargs
    ):
        """
        Initialize TimeSeries explainer.
        
        Args:
            model: TimeSeriesPredictor instance
            X: TimeSeriesDataFrame or DataFrame with time series data
            y: Not used for time series (can be None or empty)
            **kwargs: Additional configuration
        """
        # For time series, y is not used - store it but don't validate
        self._y_provided = y is not None and len(y) > 0
        
        # Initialize with empty series if y not provided (time series doesn't need y)
        if y is None or len(y) == 0:
            y = pd.Series(dtype=float, name='target')
        
        # Bypass parent __init__ to avoid y validation
        self.model = model
        self.X = X
        self.y = y
        self.max_samples = kwargs.get('max_samples', 1000)
        self.config = kwargs
        
        # Cache for computed values
        self._feature_importance = None
        self._shap_values = None
        self._metrics = None
        self._predictions = None
        
        self.predictor = model
        self._forecasts = None
        
        # Validate X only
        self._validate()
    
    def _validate(self):
        """Override validation for time series - only X is required."""
        if self.X is None or len(self.X) == 0:
            raise ValueError("X (time series data) cannot be empty")
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'autogluon_timeseries'
    
    @property
    def problem_type(self) -> str:
        """Time series is always forecasting."""
        return 'forecasting'
    
    @property
    def is_forecasting(self) -> bool:
        """Override to return True."""
        return True
    
    @property
    def n_samples(self) -> int:
        """Get number of time series data points."""
        return len(self.X)
    
    @property
    def feature_names(self) -> list:
        """Get column names from time series data."""
        return list(self.X.columns) if hasattr(self.X, 'columns') else []
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Returns DataFrame with forecast values and potentially quantiles.
        """
        if X is None:
            X = self.X
        
        try:
            forecasts = self.predictor.predict(X)
            self._forecasts = forecasts
            return forecasts
        except Exception as e:
            print(f"Forecast generation failed: {e}")
            return pd.DataFrame()
    
    def generate_forecasts(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            quantiles: Quantile levels for prediction intervals
            
        Returns:
            DataFrame with point forecasts and quantile predictions
        """
        try:
            forecasts = self.predictor.predict(self.X, quantiles=quantiles)
            self._forecasts = forecasts
            return forecasts
        except Exception as e:
            print(f"Forecast with quantiles failed: {e}")
            return self.get_predictions()
    
    def get_feature_importance(self):
        """
        Feature importance is not applicable to time series forecasting.

        Time series models predict future values from temporal patterns, not
        from per-row feature weights, so there is no meaningful importance
        ranking to show.  Returning None causes the report to skip the section
        rather than display fabricated equal-weight bars.
        """
        print("Feature importance is not applicable for time series forecasting — skipping.")
        return None
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        SHAP is not applicable to time series forecasting.
        
        Returns None with explanation.
        """
        print("Warning: SHAP analysis is not applicable to time series forecasting models.")
        print("Time series models predict future values based on historical patterns,")
        print("which doesn't fit the SHAP framework of feature attribution.")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get forecast statistics and model info.
        
        Note: Traditional metrics like accuracy don't apply to forecasting.
        """
        if self._metrics is not None:
            return self._metrics
        
        metrics = {
            'model_type': self.model_type,
            'problem_type': self.problem_type,
            'n_samples': self.n_samples,
        }
        
        # Generate forecasts if not done
        if self._forecasts is None:
            self.get_predictions()
        
        # Add forecast statistics
        if self._forecasts is not None and len(self._forecasts) > 0:
            metrics.update(self._get_forecast_statistics())
        
        # Add model info
        metrics.update(self._get_model_info())
        
        self._metrics = metrics
        return metrics
    
    def _get_forecast_statistics(self) -> Dict[str, Any]:
        """Calculate statistics on forecasts."""
        stats = {}
        
        forecasts = self._forecasts
        
        try:
            # Find mean or point forecast column
            if 'mean' in forecasts.columns:
                mean_col = forecasts['mean']
            else:
                # Use first numeric column
                numeric_cols = forecasts.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    mean_col = forecasts[numeric_cols[0]]
                else:
                    return stats
            
            stats['forecast_mean'] = round(float(mean_col.mean()), 4)
            stats['forecast_std'] = round(float(mean_col.std()), 4)
            stats['forecast_min'] = round(float(mean_col.min()), 4)
            stats['forecast_max'] = round(float(mean_col.max()), 4)
            stats['forecast_horizon'] = len(mean_col)
            
        except Exception as e:
            print(f"Forecast statistics failed: {e}")
        
        return stats
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get TimeSeriesPredictor model information."""
        info = {}
        
        try:
            info['model_names'] = self.predictor.get_model_names()
        except Exception:
            pass
        
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            if len(leaderboard) > 0:
                info['best_model'] = leaderboard.iloc[0].name
        except Exception:
            pass
        
        return info
    
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """Get model leaderboard."""
        try:
            return self.predictor.leaderboard(silent=True)
        except Exception:
            return None
    
    def generate_plots(self) -> Dict[str, str]:
        """Generate forecast visualization plots."""
        from xai_core.visualizations import plot_forecast
        
        plots = {}
        
        # Generate forecasts if needed
        if self._forecasts is None:
            self.generate_forecasts()
        
        if self._forecasts is not None and len(self._forecasts) > 0:
            try:
                forecast_plot = plot_forecast(self._forecasts)
                if forecast_plot:
                    plots['forecast'] = forecast_plot
            except Exception as e:
                print(f"Forecast plot failed: {e}")
        
        return plots
    
    def generate_report(self, mode: str = 'expert') -> str:
        """Generate HTML report for time series model."""
        from xai_core.report_builder import ReportBuilder
        return ReportBuilder(self).build_timeseries_report(mode)
