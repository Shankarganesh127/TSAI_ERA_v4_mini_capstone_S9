#!/usr/bin/env python3
"""
SageMaker Training Logging Configuration

This module provides consistent logging setup for all SageMaker training components.
All logs are saved to both files and displayed to console with proper formatting.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_sagemaker_logger(name, log_level=logging.INFO):
    """
    Setup logger for SageMaker training components
    
    Args:
        name (str): Logger name (usually module name)
        log_level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s | %(message)s'
    )
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sagemaker_{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - clean output with Windows-compatible encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Handle Windows encoding issues
    if sys.platform.startswith('win'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initialization (with basic text for Windows compatibility)
    logger.info(f"[INIT] Logger '{name}' initialized")
    logger.info(f"[FILE] Detailed logs: {log_file}")
    
    return logger

def log_sagemaker_config(logger, config_dict, title="Configuration"):
    """Log configuration dictionary in a formatted way"""
    logger.info(f"⚙️  {title}:")
    for key, value in config_dict.items():
        logger.info(f"   {key}: {value}")

def log_7_step_pipeline_status(logger, step_status):
    """Log 7-step pipeline status"""
    logger.info("📋 7-Step Pipeline Status:")
    
    steps = [
        ("1️⃣ LR Range Test", step_status.get('lr_finder', 'enabled')),
        ("2️⃣ Pick LR bounds", step_status.get('lr_bounds', 'auto')),
        ("3️⃣ OneCycle LR", step_status.get('onecycle', 'enabled')),
        ("4️⃣ Choose batch size", step_status.get('batch_size', 'auto-detect')),
        ("5️⃣ Tune weight-decay", step_status.get('wd_search', 'enabled')),
        ("6️⃣ Full training", step_status.get('training', 'enabled')),
        ("7️⃣ Monitor", step_status.get('monitoring', 'enabled'))
    ]
    
    for step_name, status in steps:
        logger.info(f"   {step_name}: {status}")

def log_training_progress(logger, epoch, total_epochs, train_acc, val_acc, lr):
    """Log training progress in consistent format"""
    progress = (epoch / total_epochs) * 100
    logger.info(f"📊 Epoch {epoch}/{total_epochs} ({progress:.1f}%) | "
                f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {lr:.2e}")

def log_hyperparameters(logger, hyperparams):
    """Log hyperparameters in organized format"""
    logger.info("🔧 Hyperparameters:")
    
    # Group related parameters
    training_params = {}
    pipeline_params = {}
    aws_params = {}
    
    for key, value in hyperparams.items():
        if key in ['epochs', 'batch_size', 'lr_min', 'lr_max', 'weight_decay']:
            training_params[key] = value
        elif key in ['run_lr_finder', 'run_wd_search', 'quick_mode']:
            pipeline_params[key] = value
        elif key in ['data_dir', 'output_dir', 'num_workers']:
            aws_params[key] = value
    
    if training_params:
        logger.info("   Training:")
        for key, value in training_params.items():
            logger.info(f"     {key}: {value}")
    
    if pipeline_params:
        logger.info("   Pipeline Control:")
        for key, value in pipeline_params.items():
            logger.info(f"     {key}: {value}")
    
    if aws_params:
        logger.info("   AWS/SageMaker:")
        for key, value in aws_params.items():
            logger.info(f"     {key}: {value}")

def log_job_summary(logger, job_name, status, duration=None, results=None):
    """Log job completion summary"""
    logger.info("=" * 60)
    logger.info(f"🎯 SageMaker Job Summary: {job_name}")
    logger.info(f"   Status: {status}")
    
    if duration:
        logger.info(f"   Duration: {duration}")
    
    if results:
        logger.info("   Results:")
        for key, value in results.items():
            logger.info(f"     {key}: {value}")
    
    logger.info("=" * 60)

# Example usage for testing
if __name__ == "__main__":
    # Test the logging setup
    logger = setup_sagemaker_logger("test_logger")
    
    logger.info("🧪 Testing SageMaker logging configuration")
    
    # Test configuration logging
    test_config = {
        'job_name': 'test-job',
        'instance_type': 'ml.p3.2xlarge',
        'epochs': 30,
        'batch_size': 64
    }
    log_sagemaker_config(logger, test_config, "Test Configuration")
    
    # Test pipeline status
    pipeline_status = {
        'lr_finder': 'enabled',
        'batch_size': 'auto-detect',
        'wd_search': 'enabled'
    }
    log_7_step_pipeline_status(logger, pipeline_status)
    
    # Test hyperparameters
    hyperparams = {
        'epochs': 30,
        'batch_size': 64,
        'run_lr_finder': True,
        'data_dir': '/opt/ml/input/data/imagenet'
    }
    log_hyperparameters(logger, hyperparams)
    
    logger.info("✅ Logging configuration test completed")