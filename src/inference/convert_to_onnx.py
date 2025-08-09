"""Convert PyTorch Lightning model to ONNX format."""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import onnx
import onnxsim
from onnxruntime import InferenceSession
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models import CIFARClassifier
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def convert_to_onnx(
    checkpoint_path: str,
    output_path: str = "models/cifar_classifier.onnx",
    input_shape: tuple = (1, 3, 32, 32),
    opset_version: int = 11,
    simplify: bool = True,
):
    """Convert PyTorch Lightning model to ONNX format."""
    logger.info(f"Converting model from {checkpoint_path} to ONNX format...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the trained model
    logger.info("Loading trained model...")
    model = CIFARClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    # Simplify the model
    if simplify:
        logger.info("Simplifying ONNX model...")
        onnx_model = onnx.load(output_path)
        simplified_model, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(simplified_model, output_path)
            logger.info("Model simplified successfully")
        else:
            logger.warning("Model simplification failed, keeping original")
    
    # Validate the ONNX model
    logger.info("Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference with ONNX Runtime
    logger.info("Testing ONNX Runtime inference...")
    session = InferenceSession(output_path)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    dummy_input_np = dummy_input.numpy()
    onnx_output = session.run([output_name], {input_name: dummy_input_np})[0]
    
    # Compare with PyTorch output
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    
    # Check if outputs are close
    max_diff = np.max(np.abs(torch_output - onnx_output))
    logger.info(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")
    
    if max_diff < 1e-5:
        logger.info("✓ ONNX conversion successful! Outputs match PyTorch model.")
    else:
        logger.warning(f"⚠ Large difference detected: {max_diff}")
    
    # Get model info
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"ONNX model size: {model_size:.2f} MB")
    logger.info(f"ONNX model saved to: {output_path}")
    
    return output_path


def benchmark_model(onnx_path: str, num_runs: int = 100):
    """Benchmark ONNX model inference speed."""
    logger.info(f"Benchmarking model: {onnx_path}")
    
    session = InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Create random input
    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        session.run([output_name], {input_name: dummy_input})
    
    # Benchmark
    import time
    start_time = time.time()
    for _ in range(num_runs):
        session.run([output_name], {input_name: dummy_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    throughput = 1000 / avg_time  # inferences per second
    
    logger.info(f"Average inference time: {avg_time:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} inferences/second")
    
    return avg_time, throughput


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/cifar_classifier.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after conversion"
    )
    
    args = parser.parse_args()
    
    # Convert to ONNX
    onnx_path = convert_to_onnx(args.checkpoint, args.output)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_model(onnx_path)


if __name__ == "__main__":
    main()
