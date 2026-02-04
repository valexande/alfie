"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import io
import pandas as pd
import pickle

# Import the app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Test health response has correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "features" in data
        
        assert data["status"] == "healthy"
        assert data["service"] == "xai-explainability"
    
    def test_health_features_dict(self, client):
        """Test features is a dictionary."""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["features"], dict)
        assert "autogluon_eda" in data["features"]


class TestRootEndpoint:
    """Tests for / endpoint."""
    
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_docs_links(self, client):
        """Test root response contains documentation links."""
        response = client.get("/")
        data = response.json()
        
        assert "docs" in data
        assert "redoc" in data


class TestExplainModelEndpoint:
    """Tests for /explain-model endpoint."""
    
    def test_missing_files_returns_422(self, client):
        """Test missing files returns validation error."""
        response = client.post("/explain-model")
        assert response.status_code == 422
    
    def test_missing_model_file_returns_422(self, client):
        """Test missing model file returns validation error."""
        # Create dummy data file
        df = pd.DataFrame({'a': [1, 2, 3], 'target': [0, 1, 0]})
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        response = client.post(
            "/explain-model",
            files={"data_file": ("data.csv", csv_buffer, "text/csv")}
        )
        
        assert response.status_code == 422
    
    def test_missing_data_file_returns_422(self, client):
        """Test missing data file returns validation error."""
        # Create dummy model file (will fail to load, but tests validation)
        model_buffer = io.BytesIO(b"dummy model data")
        
        response = client.post(
            "/explain-model",
            files={"model_file": ("model.pkl", model_buffer, "application/octet-stream")}
        )
        
        assert response.status_code == 422


class TestExplainUC2DataEndpoint:
    """Tests for /explain-uc2-data endpoint."""
    
    def test_missing_files_returns_422(self, client):
        """Test missing files returns validation error."""
        response = client.post("/explain-uc2-data")
        assert response.status_code == 422
    
    def test_valid_files_returns_200(self, client):
        """Test valid files returns HTML response."""
        # Create frame data
        frame_df = pd.DataFrame({
            'frame_timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'eyes_closed': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'yawning': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'alert': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        })
        
        # Create heart rate data
        hr_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'heart_rate': [72, 75, 80, 78, 85, 90, 76, 88, 74, 95]
        })
        
        # Convert to CSV buffers
        frame_buffer = io.BytesIO()
        frame_df.to_csv(frame_buffer, index=False)
        frame_buffer.seek(0)
        
        hr_buffer = io.BytesIO()
        hr_df.to_csv(hr_buffer, index=False)
        hr_buffer.seek(0)
        
        response = client.post(
            "/explain-uc2-data",
            files={
                "frame_file": ("frames.csv", frame_buffer, "text/csv"),
                "hr_file": ("heart_rate.csv", hr_buffer, "text/csv")
            }
        )
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Driver" in response.text


class TestExplainModelEDAEndpoint:
    """Tests for /explain-model/eda endpoint."""
    
    def test_missing_data_file_returns_422(self, client):
        """Test missing data file returns validation error."""
        response = client.post("/explain-model/eda")
        assert response.status_code == 422
    
    def test_valid_data_returns_200(self, client):
        """Test valid data file returns HTML report."""
        # Create dummy data file
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        response = client.post(
            "/explain-model/eda",
            files={"data_file": ("data.csv", csv_buffer, "text/csv")}
        )
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Dataset Analysis Report" in response.text
        assert "Statistical Summary" in response.text
        assert "Correlation Analysis" in response.text

    def test_with_target_col(self, client):
        """Test specifying target column works."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3], 
            'my_target': [0, 1, 0]
        })
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        response = client.post(
            "/explain-model/eda",
            files={"data_file": ("data.csv", csv_buffer, "text/csv")},
            data={"target_col": "my_target"}
        )
        
        assert response.status_code == 200
        assert "Target Distribution" in response.text


class TestOpenAPISchema:
    """Tests for OpenAPI schema."""
    
    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_openapi_contains_paths(self, client):
        """Test OpenAPI schema contains expected paths."""
        response = client.get("/openapi.json")
        data = response.json()
        
        assert "paths" in data
        assert "/health" in data["paths"]
        assert "/explain-model" in data["paths"]
        assert "/explain-uc2-data" in data["paths"]
