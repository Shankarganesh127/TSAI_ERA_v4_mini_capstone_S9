#!/usr/bin/env python3
"""
Main entry point for ImageNet-1K training with ResNet50
TSAI ERAv4 Mini Capstone S9

This script serves as the main launcher for the ImageNet training project.
It provides a user-friendly interface and integrates all project components.
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from logger_setup import setup_logger, log_system_info

# Import train_imagenet conditionally
try:
    from train_imagenet import main as train_main
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    IMPORT_ERROR = str(e)


def check_dataset_availability():
    """Check for available datasets in common locations"""
    logger = setup_logger("main")
    
    # Common dataset locations
    possible_locations = [
        "./imagenet",
        "./datasets/imagenet", 
        "./data/imagenet",
        "../imagenet",
        "../datasets/imagenet",
        "C:/datasets/imagenet",
        "D:/datasets/imagenet"
    ]
    
    found_datasets = []
    for location in possible_locations:
        if os.path.exists(location):
            train_dir = os.path.join(location, "train")
            val_dir = os.path.join(location, "val")
            if os.path.exists(train_dir) and os.path.exists(val_dir):
                found_datasets.append(os.path.abspath(location))
    
    if found_datasets:
        logger.info("Found ImageNet datasets at:")
        for i, dataset in enumerate(found_datasets, 1):
            logger.info(f"  {i}. {dataset}")
        return found_datasets
    else:
        logger.warning("No ImageNet datasets found in common locations")
        logger.info("Common locations checked:")
        for location in possible_locations:
            logger.info(f"  - {location}")
        return []


def show_project_info():
    """Display project information and usage guide"""
    logger = setup_logger("main")
    
    logger.info("=" * 60)
    logger.info("TSAI ERAv4 Mini Capstone S9 - ImageNet Training")
    logger.info("=" * 60)
    logger.info("ResNet50 training on ImageNet-1K with modern tooling")
    logger.info("")
    
    # Log system information
    log_system_info(logger)
    
    logger.info("")
    logger.info("Project Features:")
    logger.info("  ✓ ResNet50 optimized for ImageNet-1K")
    logger.info("  ✓ Professional logging system")
    logger.info("  ✓ UV-based dependency management")
    logger.info("  ✓ Comprehensive data augmentation")
    logger.info("  ✓ Automatic checkpointing")
    logger.info("  ✓ Multi-GPU ready")


def parse_arguments():
    """Parse command line arguments with helpful defaults"""
    parser = argparse.ArgumentParser(
        description='TSAI ERAv4 Mini Capstone S9 - ImageNet Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with auto-detected dataset
  python main.py
  
  # Training with specific dataset
  python main.py --data-dir /path/to/imagenet
  
  # Quick training (fewer epochs)
  python main.py --epochs 10 --batch-size 128
  
  # Fine-tuning with pretrained weights
  python main.py --pretrained --epochs 20 --lr 0.01
  
  # CPU training (for testing)
  python main.py --batch-size 32 --epochs 1
        """
    )
    
    # Dataset options
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to ImageNet dataset (auto-detected if not specified)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of training epochs (default: 90)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    
    # Model options
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights for fine-tuning')
    
    # System options
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    
    # Convenience options
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (10 epochs, smaller batch)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode (1 epoch, very small batch)')
    parser.add_argument('--info', action='store_true',
                       help='Show project information and exit')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Show project info if requested
    if args.info:
        show_project_info()
        return
    
    # Check if training is available
    if not TRAINING_AVAILABLE:
        logger = setup_logger("main")
        logger.error("Training dependencies not available!")
        logger.error(f"Import error: {IMPORT_ERROR}")
        logger.error("Please install dependencies:")
        logger.error("  1. Run: cd setup && python setup_uv.py")
        logger.error("  2. Or manually: uv pip install torch torchvision")
        logger.error("  3. Then try running main.py again")
        logger.info("For project info only, use: python main.py --info")
        return 1
    
    # Initialize logger
    logger = setup_logger("main")
    
    logger.info("Starting TSAI ERAv4 Mini Capstone S9 - ImageNet Training")
    
    # Apply convenience modes
    if args.quick:
        logger.info("Quick training mode enabled")
        args.epochs = 10
        args.batch_size = 128
        
    if args.test:
        logger.info("Test training mode enabled")
        args.epochs = 1
        args.batch_size = 32
    
    # Auto-detect dataset if not specified
    if args.data_dir is None:
        logger.info("No dataset directory specified, attempting auto-detection...")
        available_datasets = check_dataset_availability()
        
        if available_datasets:
            args.data_dir = available_datasets[0]
            logger.info(f"Using auto-detected dataset: {args.data_dir}")
        else:
            logger.error("No dataset found! Please:")
            logger.error("1. Specify dataset path with --data-dir")
            logger.error("2. Use dataset_tools/imagenet_subset_downloader.ipynb to download data")
            logger.error("3. Or place ImageNet in one of the common locations")
            return 1
    
    # Validate dataset directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Dataset directory not found: {args.data_dir}")
        return 1
    
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        logger.error(f"Invalid dataset structure in {args.data_dir}")
        logger.error("Expected structure:")
        logger.error("  dataset/train/class_name/images...")
        logger.error("  dataset/val/class_name/images...")
        return 1
    
    logger.info(f"Dataset validated: {args.data_dir}")
    
    # Log final configuration
    logger.info("Training Configuration:")
    logger.info(f"  Dataset: {args.data_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Pretrained: {args.pretrained}")
    logger.info(f"  Save directory: {args.save_dir}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Launch training by directly calling train_imagenet main function
    logger.info("Launching ImageNet training...")
    logger.info("=" * 60)
    
    try:
        # Temporarily replace sys.argv to pass arguments to train_imagenet
        original_argv = sys.argv.copy()
        
        # Build argument list for train_imagenet
        train_args = [
            "train_imagenet.py",  # Script name
            "--data-dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--momentum", str(args.momentum),
            "--weight-decay", str(args.weight_decay),
            "--num-workers", str(args.num_workers),
            "--save-dir", args.save_dir
        ]
        
        if args.pretrained:
            train_args.append("--pretrained")
        
        # Set sys.argv for train_imagenet
        sys.argv = train_args
        
        # Call the training function
        train_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Checkpoints saved in: {args.save_dir}")
        logger.info("Logs available in: logs/")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)