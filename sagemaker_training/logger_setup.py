#!/usr/bin/env python3
"""
Comprehensive logging setup for the ImageNet training project.
Creates log files based on the running script name.
"""

import logging
import os
import sys
from pathlib import Path


def setup_unified_logger(log_level=logging.INFO, unified_log_name="imagenet_training.log"):
    """
    Set up a unified logger that all components write to the same log file.
    
    Args:
        log_level: Logging level (default: INFO)
        unified_log_name: Name of the unified log file
    
    Returns:
        logger: Configured logger instance
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / unified_log_name

    # Create root logger that will be shared across all modules
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(fmt='%(levelname)s - [%(name)s] - %(message)s')

    # File handler (append mode)
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.unified_log_path = str(log_filename)
    return root_logger


def get_unified_logger(name=None):
    """
    Get a logger that writes to the unified log file.
    
    Args:
        name: Logger name (auto-detected if None)
    
    Returns:
        logger: Logger instance that writes to unified log
    """
    if name is None:
        name = Path(sys.argv[0]).stem
    root_logger = logging.getLogger()
    if not root_logger.handlers or not hasattr(root_logger, 'unified_log_path'):
        setup_unified_logger()
    return logging.getLogger(name)


def log_system_info(logger):
    """
    Log comprehensive system information.
    
    Args:
        logger: Logger instance
    """
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    try:
        import platform
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Architecture: {platform.architecture()}")
        logger.info(f"Processor: {platform.processor()}")
    except ImportError:
        logger.warning("Platform information not available")


def log_training_config(logger, args):
    """
    Log training configuration parameters.
    
    Args:
        logger: Logger instance
        args: Argument namespace from argparse
    """
    logger.info("Training Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")


def log_model_info(logger, model, model_name="Model"):
    """
    Log model information including parameters and architecture summary.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        model_name: Name of the model
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"{model_name} Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log model size in MB
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        logger.info(f"  Model size: {model_size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to log model info: {e}")


def log_dataset_info(logger, train_loader, val_loader=None):
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
    """
    try:
        logger.info("Dataset Information:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        logger.info(f"  Batch size: {train_loader.batch_size}")
        
        if val_loader:
            logger.info(f"  Validation batches: {len(val_loader)}")
            logger.info(f"  Validation samples: {len(val_loader.dataset)}")
            
        # Get sample batch to determine input shape
        sample_batch = next(iter(train_loader))
        inputs, targets = sample_batch
        logger.info(f"  Input shape: {inputs.shape}")
        logger.info(f"  Target shape: {targets.shape}")
        
    except Exception as e:
        logger.error(f"Failed to log dataset info: {e}")


# Convenience functions for different log levels
def log_error(logger, message, exception=None):
    """Log error message with optional exception details."""
    if exception:
        logger.error(f"{message}: {exception}")
    else:
        logger.error(message)


def log_warning(logger, message):
    """Log warning message."""
    logger.warning(message)


def log_info(logger, message):
    """Log info message."""
    logger.info(message)


def log_debug(logger, message):
    """Log debug message."""
    logger.debug(message)


# Example usage
if __name__ == "__main__":
    # Test the logger setup
    logger = setup_logger("test_logger")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    log_system_info(logger)
    
    logger.info("Logger test completed. Check the logs/ directory for output.")