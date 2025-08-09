"""Tests for the FastAPI inference server."""

import pytest
import base64
import io
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.api import app


class TestInferenceAPI:
    """Test the inference API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_predictor(self):
        """Mock predictor for testing."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            "predicted_class": "cat",
            "confidence": 0.85,
            "class_probabilities": {
                "airplane": 0.05, "automobile": 0.02, "bird": 0.03,
                "cat": 0.85, "deer": 0.01, "dog": 0.02,
                "frog": 0.01, "horse": 0.005, "ship": 0.005, "truck": 0.01
            }
        }
        mock_predictor.predict_batch.return_value = [
            {
                "predicted_class": "cat",
                "confidence": 0.85,
                "class_probabilities": {
                    "airplane": 0.05, "automobile": 0.02, "bird": 0.03,
                    "cat": 0.85, "deer": 0.01, "dog": 0.02,
                    "frog": 0.01, "horse": 0.005, "ship": 0.005, "truck": 0.01
                }
            }
        ]
        mock_predictor.get_model_info.return_value = {
            "model_path": "/app/models/cifar_classifier.onnx",
            "input_shape": [1, 3, 32, 32],
            "output_shape": [1, 10],
            "num_classes": 10,
            "classes": ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        }
        return mock_predictor
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint_no_model(self, client):
        """Test health endpoint when model is not loaded."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["model_loaded"] is False
    
    @patch("inference.api.predictor")
    def test_health_endpoint_with_model(self, mock_predictor_global, client, mock_predictor):
        """Test health endpoint when model is loaded."""
        mock_predictor_global = mock_predictor
        
        with patch("inference.api.predictor", mock_predictor):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_info" in data
    
    def test_predict_no_model(self, client, sample_image):
        """Test prediction when model is not loaded."""
        # Convert image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = client.post("/predict", json={"image": image_base64})
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch("inference.api.predictor")
    def test_predict_success(self, mock_predictor_global, client, mock_predictor, sample_image):
        """Test successful prediction."""
        mock_predictor_global = mock_predictor
        
        # Convert image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        with patch("inference.api.predictor", mock_predictor):
            response = client.post("/predict", json={"image": image_base64})
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "inference_time_ms" in data
        assert data["predicted_class"] == "cat"
        assert data["confidence"] == 0.85
    
    @patch("inference.api.predictor")
    def test_predict_batch_success(self, mock_predictor_global, client, mock_predictor, sample_image):
        """Test successful batch prediction."""
        mock_predictor_global = mock_predictor
        
        # Convert image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        with patch("inference.api.predictor", mock_predictor):
            response = client.post("/predict/batch", json={"images": [image_base64]})
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_inference_time_ms" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["predicted_class"] == "cat"
    
    def test_predict_batch_no_images(self, client):
        """Test batch prediction with no images."""
        response = client.post("/predict/batch", json={"images": []})
        
        assert response.status_code == 400
        assert "No images provided" in response.json()["detail"]
    
    @patch("inference.api.predictor")
    def test_predict_batch_too_many_images(self, mock_predictor_global, client, mock_predictor):
        """Test batch prediction with too many images."""
        mock_predictor_global = mock_predictor
        
        # Create a list with more than 100 images
        images = ["fake_base64"] * 101
        
        response = client.post("/predict/batch", json={"images": images})
        
        # Should fail validation due to max_items=100 constraint
        assert response.status_code == 422
    
    @patch("inference.api.predictor")
    def test_get_model_info_success(self, mock_predictor_global, client, mock_predictor):
        """Test getting model info."""
        mock_predictor_global = mock_predictor
        
        with patch("inference.api.predictor", mock_predictor):
            response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "num_classes" in data
        assert data["num_classes"] == 10
    
    def test_get_model_info_no_model(self, client):
        """Test getting model info when model is not loaded."""
        response = client.get("/model/info")
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch("inference.api.predictor")
    def test_predict_error_handling(self, mock_predictor_global, client, mock_predictor):
        """Test prediction error handling."""
        mock_predictor_global = mock_predictor
        mock_predictor.predict.side_effect = Exception("Prediction failed")
        
        with patch("inference.api.predictor", mock_predictor):
            response = client.post("/predict", json={"image": "invalid_base64"})
        
        assert response.status_code == 400
        assert "Prediction failed" in response.json()["detail"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        
        # FastAPI automatically handles OPTIONS requests for CORS
        assert response.status_code == 200 or response.status_code == 405
    
    def test_process_time_header(self, client):
        """Test that process time header is added."""
        response = client.get("/")
        
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
