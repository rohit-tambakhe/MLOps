"""Pytest configuration and fixtures."""

import os
import sys
import tempfile
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image():
    """Create a sample CIFAR-10 sized image."""
    # Create a random RGB image (32x32)
    image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for model testing."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for testing."""
    class MockWandB:
        run = None
        
        def init(self, *args, **kwargs):
            pass
        
        def log(self, *args, **kwargs):
            pass
        
        def finish(self):
            pass
    
    mock_wandb_instance = MockWandB()
    monkeypatch.setattr("wandb.init", mock_wandb_instance.init)
    monkeypatch.setattr("wandb.log", mock_wandb_instance.log)
    monkeypatch.setattr("wandb.finish", mock_wandb_instance.finish)
    monkeypatch.setattr("wandb.run", None)
    
    return mock_wandb_instance


@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        "num_classes": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "dropout_rate": 0.5,
        "hidden_dim": 512,
    }


@pytest.fixture
def data_config():
    """Sample data configuration."""
    return {
        "batch_size": 32,
        "num_workers": 0,  # Use 0 for testing
        "val_split": 0.1,
        "seed": 42,
    }
