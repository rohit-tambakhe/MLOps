"""Training script with Hydra configuration management."""

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
def train(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting training with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model
    logger.info("Initializing model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Initialize logger
    logger.info("Initializing experiment logger...")
    experiment_logger = hydra.utils.instantiate(cfg.logger)
    
    # Initialize callbacks
    callbacks = []
    if "callbacks" in cfg:
        for callback_name, callback_conf in cfg.callbacks.items():
            if callback_conf is not None:
                logger.info(f"Initializing callback: {callback_name}")
                callbacks.append(hydra.utils.instantiate(callback_conf))
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=experiment_logger,
        callbacks=callbacks,
    )
    
    # Log hyperparameters
    if hasattr(experiment_logger, "log_hyperparams"):
        experiment_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    
    # Test the model
    logger.info("Testing model...")
    test_results = trainer.test(model, datamodule)
    
    # Log final test results
    logger.info(f"Test results: {test_results}")
    
    # Save final model
    final_model_path = os.path.join(cfg.checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    train()
