"""Tests for model components."""

import pytest
import torch
import pytorch_lightning as pl
from models.cifar_classifier import CIFARClassifier


class TestCIFARClassifier:
    """Test the CIFAR classifier model."""
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = CIFARClassifier(**model_config)
        
        assert model.num_classes == 10
        assert model.learning_rate == 0.001
        assert isinstance(model, pl.LightningModule)
    
    def test_forward_pass(self, model_config, sample_tensor):
        """Test forward pass."""
        model = CIFARClassifier(**model_config)
        
        with torch.no_grad():
            output = model(sample_tensor)
        
        assert output.shape == (1, 10)
        assert torch.isfinite(output).all()
    
    def test_training_step(self, model_config, sample_batch):
        """Test training step."""
        model = CIFARClassifier(**model_config)
        
        loss = model.training_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
    
    def test_validation_step(self, model_config, sample_batch):
        """Test validation step."""
        model = CIFARClassifier(**model_config)
        
        loss = model.validation_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
    
    def test_test_step(self, model_config, sample_batch):
        """Test test step."""
        model = CIFARClassifier(**model_config)
        
        loss = model.test_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
    
    def test_configure_optimizers(self, model_config):
        """Test optimizer configuration."""
        model = CIFARClassifier(**model_config)
        
        optimizer_config = model.configure_optimizers()
        
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert optimizer_config["lr_scheduler"]["monitor"] == "val/loss"
    
    def test_predict_step(self, model_config, sample_tensor):
        """Test prediction step."""
        model = CIFARClassifier(**model_config)
        
        result = model.predict_step(sample_tensor, 0)
        
        assert "predictions" in result
        assert "probabilities" in result
        assert result["predictions"].shape == (1,)
        assert result["probabilities"].shape == (1, 10)
        
        # Check probabilities sum to 1
        prob_sum = result["probabilities"].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6)
    
    def test_model_parameters_count(self, model_config):
        """Test that model has reasonable number of parameters."""
        model = CIFARClassifier(**model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
        assert total_params < 10_000_000  # Less than 10M parameters
    
    def test_model_different_configs(self):
        """Test model with different configurations."""
        configs = [
            {"num_classes": 10, "hidden_dim": 256},
            {"num_classes": 10, "hidden_dim": 1024, "dropout_rate": 0.3},
            {"num_classes": 10, "learning_rate": 0.01},
        ]
        
        for config in configs:
            model = CIFARClassifier(**config)
            sample_input = torch.randn(2, 3, 32, 32)
            
            with torch.no_grad():
                output = model(sample_input)
            
            assert output.shape == (2, 10)
            assert torch.isfinite(output).all()
