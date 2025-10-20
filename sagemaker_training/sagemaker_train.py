#!/usr/bin/env python3
"""
SageMaker Training Entry Point for ImageNet ResNet50 Pipeline
This script acts as a bridge between SageMaker and your existing training pipeline.
It preserves all your existing code functionality while adapting it for SageMaker environment.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for existing module imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Import your existing modules without modification
from imagenet_training_pipeline import ImageNetTrainer
from logger_setup import setup_logger


def setup_sagemaker_environment():
    """Configure environment for SageMaker training"""
    
    # SageMaker environment variables
    sm_model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    sm_output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    sm_channel_training = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    sm_channel_validation = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
    
    # Create directories if they don't exist
    os.makedirs(sm_model_dir, exist_ok=True)
    os.makedirs(sm_output_data_dir, exist_ok=True)
    
    return {
        'model_dir': sm_model_dir,
        'output_data_dir': sm_output_data_dir,
        'training_dir': sm_channel_training,
        'validation_dir': sm_channel_validation
    }


def parse_sagemaker_args():
    """Parse arguments passed from SageMaker"""
    parser = argparse.ArgumentParser(description='SageMaker ImageNet Training')
    
    # SageMaker directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # Training hyperparameters (can be overridden by SageMaker hyperparameters)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr-max', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Pipeline configuration
    parser.add_argument('--quick-mode', action='store_true', 
                       help='Run quick validation mode (faster iterations)')
    parser.add_argument('--lr-finder', action='store_true', 
                       help='Run LR finder step')
    parser.add_argument('--full-pipeline', action='store_true', default=True,
                       help='Run complete 7-step pipeline')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50'], help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=False,
                       help='Use pretrained weights')
    
    # Mixed precision and optimization
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    return parser.parse_args()


def create_training_config(args, sagemaker_dirs):
    """Create configuration for your existing training pipeline"""
    
    # Check if we have separate train/val dirs or a single data dir
    if os.path.exists(os.path.join(args.train, 'train')) and os.path.exists(os.path.join(args.train, 'val')):
        # Standard ImageNet structure in training channel
        data_dir = args.train
    else:
        # Separate channels for train and validation
        data_dir = {
            'train': args.train,
            'val': args.validation if os.path.exists(args.validation) else args.train
        }
    
    config = {
        'data_dir': data_dir,
        'output_dir': sagemaker_dirs['output_data_dir'],
        'model_dir': sagemaker_dirs['model_dir'],
        
        # Training hyperparameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_max': args.lr_max,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'num_workers': args.num_workers,
        
        # Pipeline configuration
        'quick_mode': args.quick_mode,
        'run_lr_finder': args.lr_finder,
        'full_pipeline': args.full_pipeline,
        
        # Model configuration
        'model_name': args.model,
        'pretrained': args.pretrained,
        
        # Optimization
        'mixed_precision': args.mixed_precision,
        'gradient_clip': args.gradient_clip,
        
        # SageMaker specific
        'save_model': True,
        'save_checkpoints': True,
        'log_interval': 10,
    }
    
    return config


def save_training_artifacts(trainer, model_dir, output_dir):
    """Save model and training artifacts for SageMaker"""
    
    # Save the final model
    model_path = os.path.join(model_dir, 'model.pth')
    trainer.save_model(model_path)
    
    # Save training history and metrics
    history_path = os.path.join(output_dir, 'training_history.json')
    if hasattr(trainer, 'history'):
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)
    
    # Save final configuration
    config_path = os.path.join(output_dir, 'final_config.json')
    if hasattr(trainer, 'config'):
        with open(config_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_config = {}
            for k, v in trainer.config.items():
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except (TypeError, ValueError):
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2)
    
    # Save model summary/info
    info_path = os.path.join(output_dir, 'model_info.json')
    model_info = {
        'model_type': 'ResNet50',
        'num_classes': 1000,
        'input_size': [3, 224, 224],
        'framework': 'PyTorch'
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)


def main():
    """Main SageMaker training function"""
    
    # Setup SageMaker environment
    sagemaker_dirs = setup_sagemaker_environment()
    
    # Parse arguments
    args = parse_sagemaker_args()
    
    # Setup logging
    logger = setup_logger("sagemaker_train", log_file=os.path.join(sagemaker_dirs['output_data_dir'], 'training.log'))
    logger.info("Starting SageMaker ImageNet Training")
    logger.info(f"SageMaker directories: {sagemaker_dirs}")
    logger.info(f"Training arguments: {vars(args)}")
    
    try:
        # Create training configuration
        config = create_training_config(args, sagemaker_dirs)
        logger.info(f"Training configuration: {config}")
        
        # Initialize your existing trainer with SageMaker-adapted config
        trainer = ImageNetTrainer(config)
        
        # Run the training pipeline (uses your existing 7-step process)
        if config['full_pipeline']:
            logger.info("Running full 7-step ImageNet training pipeline")
            trainer.run_full_pipeline()
        else:
            logger.info("Running standard training")
            trainer.train()
        
        # Save training artifacts for SageMaker
        save_training_artifacts(trainer, sagemaker_dirs['model_dir'], sagemaker_dirs['output_data_dir'])
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()