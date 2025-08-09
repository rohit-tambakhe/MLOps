"""Hyperparameter optimization using Hydra and Optuna."""

import os
import sys
from pathlib import Path
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import wandb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models import CIFARClassifier
from src.data import CIFARDataModule
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def sweep(cfg: DictConfig) -> float:
    """Hyperparameter optimization function."""
    logger.info("Starting hyperparameter optimization trial...")
    logger.info(f"Trial parameters: {OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Initialize data module
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model)
    
    # Initialize logger with trial-specific name
    cfg.logger.name = f"{cfg.experiment_name}_trial_{hydra.core.hydra_config.HydraConfig.get().job.num}"
    experiment_logger = hydra.utils.instantiate(cfg.logger)
    
    # Initialize callbacks (with early stopping for efficiency)
    callbacks = []
    if "callbacks" in cfg:
        for callback_name, callback_conf in cfg.callbacks.items():
            if callback_conf is not None:
                # Reduce early stopping patience for sweeps
                if callback_name == "early_stopping":
                    callback_conf.patience = 5
                callbacks.append(hydra.utils.instantiate(callback_conf))
    
    # Initialize trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=experiment_logger,
        callbacks=callbacks,
    )
    
    # Log hyperparameters
    if hasattr(experiment_logger, "log_hyperparams"):
        experiment_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    try:
        # Train model
        trainer.fit(model, datamodule)
        
        # Get validation accuracy
        val_acc = trainer.callback_metrics.get("val/acc", 0.0)
        
        # Log final metric
        logger.info(f"Trial completed with validation accuracy: {val_acc:.4f}")
        
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()
        
        return float(val_acc)
        
    except Exception as e:
        logger.error(f"Trial failed with error: {e}")
        if wandb.run is not None:
            wandb.finish()
        return 0.0


if __name__ == "__main__":
    sweep()
