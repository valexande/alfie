"""
Configuration settings for the XAI API.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "XAI Explainability API"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 5000
    workers: int = 1
    
    # CORS Settings
    cors_origins: str = "*"
    
    # Model Settings
    max_shap_samples: int = 300
    max_upload_size_mb: int = 500
    
    # Temp Directory for model extraction
    temp_dir: Optional[str] = None
    
    model_config = {
        "env_prefix": "XAI_",
        "env_file": ".env",
        "extra": "ignore"
    }
    
    @property
    def temp_directory(self) -> str:
        """Get temp directory, creating if needed."""
        import tempfile
        if self.temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)
            return self.temp_dir
        return tempfile.gettempdir()


# Global settings instance
settings = Settings()
