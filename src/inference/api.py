"""FastAPI inference server for CIFAR-10 classification."""

import os
import sys
from pathlib import Path
import time
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.onnx_predictor import ONNXPredictor
from src.utils.logging import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR-10 Classification API",
    description="Production-ready API for CIFAR-10 image classification using ONNX",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Request/Response models
class PredictionRequest(BaseModel):
    """Single prediction request model."""
    image: str = Field(..., description="Base64 encoded image")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    images: List[str] = Field(..., description="List of base64 encoded images", max_items=100)

class PredictionResponse(BaseModel):
    """Prediction response model."""
    predicted_class: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Prediction confidence score")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_inference_time_ms: float = Field(..., description="Total inference time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Error details")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    
    model_path = os.getenv("MODEL_PATH", "models/cifar_classifier.onnx")
    
    try:
        logger.info(f"Loading model from: {model_path}")
        predictor = ONNXPredictor(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't raise exception to allow health checks to work
        predictor = None


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "CIFAR-10 Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None
    model_info = {}
    
    if model_loaded:
        try:
            model_info = predictor.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            model_info = {"error": str(e)}
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Single image prediction endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        result = predictor.predict(request.image)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            class_probabilities=result["class_probabilities"],
            inference_time_ms=inference_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch image prediction endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    try:
        start_time = time.time()
        results = predictor.predict_batch(request.images)
        total_inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        predictions = []
        for i, result in enumerate(results):
            if "error" in result:
                # Handle individual prediction errors
                predictions.append(PredictionResponse(
                    predicted_class="unknown",
                    confidence=0.0,
                    class_probabilities={},
                    inference_time_ms=0.0
                ))
            else:
                predictions.append(PredictionResponse(
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    class_probabilities=result["class_probabilities"],
                    inference_time_ms=total_inference_time / len(request.images)  # Average per image
                ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_inference_time_ms=total_inference_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return predictor.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


def main():
    """Main function to run the API server."""
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
