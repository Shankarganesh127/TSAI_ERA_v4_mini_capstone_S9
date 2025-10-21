#!/usr/bin/env python3
"""
Simple SageMaker Job Launcher for 7-Step ImageNet Training

Usage Examples:
    # Full automated pipeline
    python launch_sagemaker.py --job-name my-job --role-arn <role> --s3-bucket s3://bucket

    # Quick development mode  
    python launch_sagemaker.py --job-name quick --role-arn <role> --s3-bucket s3://bucket --quick-mode --epochs 5

    # Custom hyperparameters
    python launch_sagemaker.py --job-name custom --role-arn <role> --s3-bucket s3://bucket --batch-size 64 --skip-lr-finder
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# All files are now in the same directory
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import logger from same directory
try:
    from logger_setup import setup_logger
except ImportError:
    from sagemaker_logging import setup_sagemaker_logger as setup_logger

try:
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
except ImportError:
    PyTorch = None
    TrainingInput = None

def main():
    # Setup logging
    logger = setup_logger("sagemaker_launcher")
    
    # Default S3 paths configuration - modify these for your datasets
    DEFAULT_CONFIG = {
        'train_data_path': None,  # Set to "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/" for your setup
        'val_data_path': None,    # Set to "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/" for your setup
        'bucket': None            # Set to your S3 bucket name
    }
    
    parser = argparse.ArgumentParser(description='Launch SageMaker ImageNet Training')
    
    # Required arguments
    parser.add_argument('--job-name', required=True, help='SageMaker training job name')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket (s3://bucket-name)')
    
    # Data configuration
    parser.add_argument('--train-data-s3', type=str, help='S3 path to training data (overrides default)')
    parser.add_argument('--val-data-s3', type=str, help='S3 path to validation data (separate channel)')
    parser.add_argument('--data-prefix', type=str, default='imagenet-data', 
                       help='Data prefix in S3 bucket (default: imagenet-data)')
    parser.add_argument('--distribution-mode', choices=['FastFile', 'File', 'Pipe'], default='FastFile',
                       help='Data distribution mode (default: FastFile for better performance)')
    
    # Instance configuration
    parser.add_argument('--instance-type', default='ml.p3.2xlarge', help='Instance type')
    parser.add_argument('--spot-training', action='store_true', help='Use spot instances')
    parser.add_argument('--volume-size', type=int, default=100, help='EBS volume size (GB)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size (auto-detect if not set)')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    
    # 7-Step pipeline control
    parser.add_argument('--skip-lr-finder', action='store_true', help='Skip LR range test (Step 1)')
    parser.add_argument('--skip-wd-search', action='store_true', help='Skip weight decay search (Step 5)')
    parser.add_argument('--quick-mode', action='store_true', help='Quick mode for development')
    parser.add_argument('--auto-confirm', action='store_true', help='Skip user confirmation prompt and launch immediately')
    
    # Manual hyperparameter overrides
    parser.add_argument('--lr-min', type=float, help='Manual minimum LR')
    parser.add_argument('--lr-max', type=float, help='Manual maximum LR')
    parser.add_argument('--weight-decay', type=float, help='Manual weight decay')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting SageMaker 7-Step ImageNet Training Job Launch")
    logger.info("=" * 60)
    
    # Check SageMaker availability
    if PyTorch is None:
        logger.error("❌ SageMaker SDK not available. Install with: pip install sagemaker")
        return
        
    # Determine data S3 path
    if args.train_data_s3:
        data_s3_path = args.train_data_s3
        logger.info(f"📂 Using custom training data path: {data_s3_path}")
    else:
        data_s3_path = f"{args.s3_bucket}/{args.data_prefix}/"
        logger.info(f"📂 Using default training data path: {data_s3_path}")
    
    # Validate S3 path format
    if not data_s3_path.startswith('s3://'):
        if data_s3_path.startswith('/'):
            data_s3_path = data_s3_path[1:]  # Remove leading slash
        data_s3_path = f"s3://{args.s3_bucket.replace('s3://', '')}/{data_s3_path}"
    
    logger.info(f"📂 Final S3 training data path: {data_s3_path}")
    
    # Configure training data input with optimal settings
    train_input = TrainingInput(
        s3_data=data_s3_path,
        distribution='FullyReplicated',  # Replicate data to all instances
        s3_data_type='S3Prefix',        # Treat as directory prefix
        input_mode=args.distribution_mode,  # Keep original case: FastFile, File, or Pipe
        compression=None                 # No compression for images
    )
    
    # Configure data inputs - support both single and multi-channel setups
    data_inputs = {'imagenet': train_input}  # Default: single channel with train/val subdirectories
    
    # Add separate validation channel if specified
    if args.val_data_s3:
        # Ensure proper S3 URL format for validation data
        val_s3_path = args.val_data_s3
        if not val_s3_path.startswith('s3://'):
            if val_s3_path.startswith('/'):
                val_s3_path = val_s3_path[1:]  # Remove leading slash
            val_s3_path = f"s3://{args.s3_bucket.replace('s3://', '')}/{val_s3_path}"
        
        val_input = TrainingInput(
            s3_data=val_s3_path,
            distribution='FullyReplicated',
            s3_data_type='S3Prefix',
            input_mode=args.distribution_mode,
            compression=None
        )
        data_inputs['validation'] = val_input
        
        logger.info(f"✅ S3 data inputs configured:")
        logger.info(f"   - imagenet (train): {data_s3_path}")
        logger.info(f"   - validation: {val_s3_path}")
        logger.info(f"   - Distribution: FullyReplicated") 
        logger.info(f"   - Input Mode: {args.distribution_mode}")
        logger.info(f"   - Data Type: S3Prefix")
    else:
        logger.info(f"✅ S3 data inputs configured:")
        logger.info(f"   - imagenet: {data_s3_path}")
        logger.info(f"   - Distribution: FullyReplicated") 
        logger.info(f"   - Input Mode: {args.distribution_mode}")
        logger.info(f"   - Data Type: S3Prefix")
        logger.info(f"   - Note: Using single channel (train/val as subdirectories)")
    
    # Prepare hyperparameters for 7-step training
    hyperparameters = {
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'run_lr_finder': str(not args.skip_lr_finder).lower(),
        'run_wd_search': str(not args.skip_wd_search).lower(),
        'quick_mode': str(args.quick_mode).lower(),
    }
    
    # Add optional hyperparameters
    if args.batch_size:
        hyperparameters['batch_size'] = args.batch_size
    if args.lr_min:
        hyperparameters['lr_min'] = args.lr_min
    if args.lr_max:
        hyperparameters['lr_max'] = args.lr_max
    if args.weight_decay:
        hyperparameters['weight_decay'] = args.weight_decay
    
    logger.info("🔧 7-Step Training Configuration:")
    logger.info(f"   - LR Finder: {'Enabled' if not args.skip_lr_finder else 'Disabled'}")
    logger.info(f"   - Weight Decay Search: {'Enabled' if not args.skip_wd_search else 'Disabled'}")
    logger.info(f"   - Quick Mode: {'Enabled' if args.quick_mode else 'Disabled'}")
    logger.info(f"   - Epochs: {args.epochs}")
    
    logger.info(f"🔧 Hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"   {key}: {value}")
    
    logger.info("=" * 60)
    
    # Confirm launch (skip if auto-confirm is enabled)
    if args.auto_confirm:
        logger.info("✅ Auto-confirm enabled - proceeding with job launch...")
    else:
        logger.info("⚠️  Awaiting user confirmation to launch SageMaker job...")
        response = input("\n🚀 Launch SageMaker job? (y/N): ")
        if response.lower() != 'y':
            logger.info("❌ Job launch cancelled by user")
            return
    
    # Create estimator
    logger.info("🔧 Creating SageMaker PyTorch estimator...")
    
    # Create optimized source directory to avoid uploading large files
    import tempfile
    import shutil
    
    # Use current sagemaker_training directory directly as source
    # All necessary files are already here - no need to copy
    current_dir = Path(__file__).parent
    logger.info("📦 Using sagemaker_training directory as source...")
    
    # List files that will be uploaded
    essential_files = []
    for file_path in current_dir.glob("*.py"):
        essential_files.append(file_path.name)
        logger.info(f"   📄 Will upload: {file_path.name}")
    
    # Also include config files
    for file_path in current_dir.rglob("*.json"):
        logger.info(f"   � Will upload: {file_path.relative_to(current_dir)}")
    
    logger.info(f"✅ Source package ready with {len(essential_files)} Python files")
    
    try:
        estimator = PyTorch(
            entry_point='sagemaker_wrapper.py',  # Direct file in sagemaker_training directory
            source_dir=str(current_dir),  # Use current sagemaker_training directory
            role=args.role_arn,
            framework_version='2.0.0', 
            py_version='py310',
            instance_count=1,
            instance_type=args.instance_type,
            hyperparameters=hyperparameters,
            use_spot_instances=args.spot_training,
            max_wait=43200 if args.spot_training else None,  # 12 hours (must be >= max_run)
            max_run=36000,  # 10 hours
            checkpoint_s3_uri=f"{args.s3_bucket}/checkpoints/{args.job_name}",
            output_path=f"{args.s3_bucket}/output/{args.job_name}",
            volume_size=args.volume_size,
            tags=[
                {'Key': 'Project', 'Value': 'ImageNet-7Step'},
                {'Key': 'QuickMode', 'Value': str(args.quick_mode)},
                {'Key': 'LRFinder', 'Value': str(not args.skip_lr_finder)},
                {'Key': 'WDSearch', 'Value': str(not args.skip_wd_search)}
            ]
        )
        logger.info("✅ SageMaker estimator created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create SageMaker estimator: {e}")
        raise
        
        # Launch training job with increased timeout
        try:
            logger.info("🚀 Launching SageMaker training job...")
            logger.info(f"📊 Data inputs: {data_inputs}")
            logger.info("⏳ Submitting job to AWS SageMaker API...")
            logger.info("💡 Source code upload may take 5-15 minutes for large projects")
            
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("SageMaker job submission timed out")
            
            # Increase timeout to 15 minutes for source code upload
            timeout_seconds = 900  # 15 minutes
            if hasattr(signal, 'SIGALRM'):  # Unix systems
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            start_time = time.time()
            try:
                logger.info("📡 Uploading source code and calling SageMaker API...")
                estimator.fit(
                    inputs=data_inputs,
                    job_name=args.job_name,
                    wait=False  # Don't wait for job completion, just submission
                )
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                    
                elapsed = time.time() - start_time
                logger.info(f"✅ Training job submitted successfully in {elapsed:.1f} seconds!")
                
            except TimeoutError:
                logger.error(f"⏰ SageMaker job submission timed out after {timeout_seconds//60} minutes")
                logger.error("💡 The job may still be submitting in the background")
                logger.error("🔍 Check AWS Console for job status")
                raise
        
            
            logger.info(f"🎯 Job Name: {args.job_name}")
            logger.info(f"📊 Instance: {args.instance_type}")
            logger.info(f"💰 Spot Training: {'Yes' if args.spot_training else 'No'}")
            logger.info("🔗 Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs")
            
            # Try to get job status
            try:
                import boto3
                logger.info("🔍 Checking job status...")
                sagemaker_client = boto3.client('sagemaker')
                response = sagemaker_client.describe_training_job(TrainingJobName=args.job_name)
                status = response['TrainingJobStatus']
                logger.info(f"📊 Current job status: {status}")
                
                if status == 'InProgress':
                    logger.info("🚀 Job is running! Check AWS Console for progress")
                elif status == 'Starting':
                    logger.info("⏳ Job is starting up...")
                elif status == 'Failed':
                    logger.error("❌ Job failed to start")
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"💥 Failure reason: {failure_reason}")
                
            except Exception as status_error:
                logger.warning(f"⚠️ Could not check job status: {status_error}")
                logger.info("💡 Job may still be starting - check AWS Console")
            
            logger.info("🎉 SageMaker training job launched successfully!")
            
        except Exception as e:
            logger.error(f"❌ SageMaker job launch failed: {e}")
            logger.error("Please check your AWS credentials, role permissions, and S3 bucket access")
            logger.error("💡 Common issues:")
            logger.error("   - Invalid role ARN or insufficient permissions")
            logger.error("   - S3 bucket doesn't exist or no access")
            logger.error("   - Instance type not available in region")
            logger.error("   - AWS API throttling or service issues")
            raise

if __name__ == "__main__":
    main()