"""ONNX model predictor for inference."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import onnxruntime as ort
import base64
import io

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ONNXPredictor:
    """ONNX model predictor for CIFAR-10 classification."""
    
    def __init__(self, model_path: str):
        """Initialize the ONNX predictor."""
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # Image preprocessing parameters (same as training)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading ONNX model from: {self.model_path}")
        
        # Create inference session
        self.session = ort.InferenceSession(self.model_path)
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info("ONNX model loaded successfully")
        logger.info(f"Input name: {self.input_name}")
        logger.info(f"Output name: {self.output_name}")
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is RGB and has correct shape
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize to 32x32 if needed
        if image.shape[:2] != (32, 32):
            image = np.array(Image.fromarray(image).resize((32, 32)))
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Apply normalization (same as training)
        image = (image - self.mean) / self.std
        
        # Add batch dimension and transpose to CHW format
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        return image.astype(np.float32)
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith("data:image"):
                base64_string = base64_string.split(",", 1)[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")
    
    def predict(self, input_data: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Make a single prediction."""
        # Handle different input types
        if isinstance(input_data, str):
            # Assume base64 encoded image
            image = self._decode_base64_image(input_data)
        elif isinstance(input_data, (np.ndarray, Image.Image)):
            image = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Create class probabilities dictionary
        class_probabilities = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }
    
    def predict_batch(self, input_batch: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        results = []
        for input_data in input_batch:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                results.append({
                    "error": str(e),
                    "predicted_class": None,
                    "confidence": 0.0,
                    "class_probabilities": {}
                })
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        return {
            "model_path": self.model_path,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "num_classes": len(self.classes),
            "classes": self.classes,
        }
