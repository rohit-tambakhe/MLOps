"""Tests for inference components."""

import pytest
import numpy as np
import base64
import io
from PIL import Image
from unittest.mock import Mock, patch
from inference.onnx_predictor import ONNXPredictor


class TestONNXPredictor:
    """Test the ONNX predictor."""
    
    @pytest.fixture
    def mock_onnx_session(self):
        """Mock ONNX runtime session."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.run.return_value = [np.random.randn(1, 10)]
        return mock_session
    
    @pytest.fixture
    def predictor(self, mock_onnx_session, temp_dir):
        """Create a predictor with mocked ONNX session."""
        model_path = f"{temp_dir}/model.onnx"
        
        with patch("onnxruntime.InferenceSession") as mock_session_class:
            mock_session_class.return_value = mock_onnx_session
            with patch("os.path.exists", return_value=True):
                predictor = ONNXPredictor(model_path)
                predictor.session = mock_onnx_session
                return predictor
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.session is not None
        assert predictor.input_name == "input"
        assert predictor.output_name == "output"
        assert len(predictor.classes) == 10
    
    def test_preprocess_image_numpy(self, predictor, sample_image):
        """Test image preprocessing with numpy array."""
        image_array = np.array(sample_image)
        processed = predictor._preprocess_image(image_array)
        
        assert processed.shape == (1, 3, 32, 32)
        assert processed.dtype == np.float32
        assert processed.min() >= -3  # Roughly normalized
        assert processed.max() <= 3
    
    def test_preprocess_image_pil(self, predictor, sample_image):
        """Test image preprocessing with PIL Image."""
        processed = predictor._preprocess_image(sample_image)
        
        assert processed.shape == (1, 3, 32, 32)
        assert processed.dtype == np.float32
    
    def test_decode_base64_image(self, predictor, sample_image):
        """Test base64 image decoding."""
        # Convert PIL image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        decoded_image = predictor._decode_base64_image(image_base64)
        
        assert isinstance(decoded_image, Image.Image)
        assert decoded_image.mode == "RGB"
    
    def test_decode_base64_image_with_prefix(self, predictor, sample_image):
        """Test base64 image decoding with data URL prefix."""
        # Convert PIL image to base64 with data URL prefix
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/png;base64,{image_base64}"
        
        decoded_image = predictor._decode_base64_image(data_url)
        
        assert isinstance(decoded_image, Image.Image)
        assert decoded_image.mode == "RGB"
    
    def test_predict_with_pil_image(self, predictor, sample_image):
        """Test prediction with PIL Image."""
        result = predictor.predict(sample_image)
        
        assert "predicted_class" in result
        assert "confidence" in result
        assert "class_probabilities" in result
        assert result["predicted_class"] in predictor.classes
        assert 0 <= result["confidence"] <= 1
        assert len(result["class_probabilities"]) == 10
        
        # Check probabilities sum to 1
        prob_sum = sum(result["class_probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_predict_with_base64(self, predictor, sample_image):
        """Test prediction with base64 string."""
        # Convert PIL image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        result = predictor.predict(image_base64)
        
        assert "predicted_class" in result
        assert "confidence" in result
        assert "class_probabilities" in result
    
    def test_predict_batch(self, predictor, sample_image):
        """Test batch prediction."""
        # Create multiple images
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        batch = [image_base64, sample_image, np.array(sample_image)]
        results = predictor.predict_batch(batch)
        
        assert len(results) == 3
        for result in results:
            if "error" not in result:
                assert "predicted_class" in result
                assert "confidence" in result
                assert "class_probabilities" in result
    
    def test_get_model_info(self, predictor):
        """Test getting model information."""
        info = predictor.get_model_info()
        
        assert "model_path" in info
        assert "input_shape" in info
        assert "output_shape" in info
        assert "num_classes" in info
        assert "classes" in info
        assert info["num_classes"] == 10
        assert len(info["classes"]) == 10
    
    def test_invalid_input_type(self, predictor):
        """Test prediction with invalid input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            predictor.predict(123)  # Invalid type
    
    def test_invalid_base64(self, predictor):
        """Test prediction with invalid base64 string."""
        with pytest.raises(ValueError, match="Failed to decode base64 image"):
            predictor.predict("invalid_base64_string")
