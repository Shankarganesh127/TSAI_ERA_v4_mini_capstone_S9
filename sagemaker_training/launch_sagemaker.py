#!/usr/bin/env python3
"""
Simple SageMaker Job Launcher for 7-Step Model Training

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
    from logger_setup import setup_unified_logger, get_unified_logger
except ImportError:
    from logger_setup import setup_unified_logger, get_unified_logger

try:
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
except ImportError:
    PyTorch = None
    TrainingInput = None

def is_multi_gpu_instance(instance_type):
    """
    Check if the instance type has multiple GPUs and should use DDP distribution.

    Returns True for multi-GPU instances that benefit from PyTorch DDP.
    """
    multi_gpu_instances = {
        # P3 instances
        'ml.p3.8xlarge': 4,   # 4 GPUs
        'ml.p3.16xlarge': 8,  # 8 GPUs
        'ml.p3dn.24xlarge': 8, # 8 GPUs

        # P4 instances
        'ml.p4d.24xlarge': 8, # 8 GPUs
        'ml.p4de.24xlarge': 8, # 8 GPUs

        # G4dn instances
        'ml.g4dn.12xlarge': 4, # 4 GPUs
        'ml.g4dn.16xlarge': 1, # Only 1 GPU - not multi-GPU

        # G5 instances
        'ml.g5.12xlarge': 4,  # 4 GPUs
        'ml.g5.24xlarge': 4,  # 4 GPUs
        'ml.g5.48xlarge': 8,  # 8 GPUs
        
        'ml.g6.12xlarge': 4,  # 4 GPUs
        'ml.g6.8xlarge': 1,  # 1 GPUs
        'ml.g6.4xlarge': 1,  # 1 GPUs
        'ml.g6.2xlarge': 1,  # 1 GPUs
    }

    return multi_gpu_instances.get(instance_type, 0) > 1

def get_instance_specs(instance_type):
    """
    Get instance specifications for hyperparameter optimization.
    
    Returns dict with gpu_count, gpu_memory_gb, cpu_count
    """
    specs = {
        # P3 instances
        'ml.p3.2xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 8},
        'ml.p3.8xlarge': {'gpus': 4, 'gpu_memory_gb': 16, 'cpus': 32},
        'ml.p3.16xlarge': {'gpus': 8, 'gpu_memory_gb': 16, 'cpus': 64},
        'ml.p3dn.24xlarge': {'gpus': 8, 'gpu_memory_gb': 32, 'cpus': 96},

        # P4 instances
        'ml.p4d.24xlarge': {'gpus': 8, 'gpu_memory_gb': 40, 'cpus': 96},
        'ml.p4de.24xlarge': {'gpus': 8, 'gpu_memory_gb': 40, 'cpus': 96},

        # G4dn instances
        'ml.g4dn.xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 4},
        'ml.g4dn.2xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 8},
        'ml.g4dn.4xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 16},
        'ml.g4dn.8xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 32},
        'ml.g4dn.12xlarge': {'gpus': 4, 'gpu_memory_gb': 16, 'cpus': 48},
        'ml.g4dn.16xlarge': {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 64},

        # G5 instances
        'ml.g5.xlarge': {'gpus': 1, 'gpu_memory_gb': 24, 'cpus': 4},
        'ml.g5.2xlarge': {'gpus': 1, 'gpu_memory_gb': 24, 'cpus': 8},
        'ml.g5.4xlarge': {'gpus': 1, 'gpu_memory_gb': 24, 'cpus': 16},
        'ml.g5.8xlarge': {'gpus': 1, 'gpu_memory_gb': 24, 'cpus': 32},
        'ml.g5.12xlarge': {'gpus': 4, 'gpu_memory_gb': 24, 'cpus': 48},
        'ml.g5.16xlarge': {'gpus': 1, 'gpu_memory_gb': 24, 'cpus': 64},
        'ml.g5.24xlarge': {'gpus': 4, 'gpu_memory_gb': 24, 'cpus': 96},
        'ml.g5.48xlarge': {'gpus': 8, 'gpu_memory_gb': 24, 'cpus': 192},
        
        # G6 instances
        'ml.g6.2xlarge': {'gpus': 1, 'gpu_memory_gb': 32, 'cpus': 8},
        'ml.g6.4xlarge': {'gpus': 1, 'gpu_memory_gb': 32, 'cpus': 16},
        'ml.g6.8xlarge': {'gpus': 1, 'gpu_memory_gb': 32, 'cpus': 32},
        'ml.g6.12xlarge': {'gpus': 4, 'gpu_memory_gb': 32, 'cpus': 48},
        'ml.g6.16xlarge': {'gpus': 1, 'gpu_memory_gb': 32, 'cpus': 64},
        'ml.g6.24xlarge': {'gpus': 4, 'gpu_memory_gb': 32, 'cpus': 96},
        'ml.g6.48xlarge': {'gpus': 8, 'gpu_memory_gb': 32, 'cpus': 192},
    }
    
    return specs.get(instance_type, {'gpus': 1, 'gpu_memory_gb': 16, 'cpus': 8})

def calculate_optimal_hyperparameters(instance_type, is_quick_mode=False):
    """
    Calculate optimal hyperparameters based on instance specifications.
    
    Returns dict with optimal batch_size, num_workers, etc.
    """
    specs = get_instance_specs(instance_type)
    gpu_count = specs['gpus']
    gpu_memory_gb = specs['gpu_memory_gb']
    cpu_count = specs['cpus']
    
    return {
        'gpu_count': gpu_count,
        'gpu_memory_gb': gpu_memory_gb,
        'cpu_count': cpu_count
    }

def create_sagemaker_estimator(args, hyperparameters):
    """Create SageMaker PyTorch estimator with given args"""
    logger = get_unified_logger("sagemaker_estimator")
    if args.instance_count > 1:
        # Use SMDDP for multi-node distributed training
        distribution = {
            'smdistributed': {
                'dataparallel': {
                    'enabled': True
                }
            }
        }
        tags=[
            {'Key': 'Project', 'Value': 'Model-7Step'},
            {'Key': 'QuickMode', 'Value': str(args.quick_mode)},
            {'Key': 'LRFinder', 'Value': str(not args.skip_lr_finder)},
            {'Key': 'WDSearch', 'Value': str(not args.skip_wd_search)}
        ]
    else:
        # Single-node training - enable DDP only for multi-GPU instances
        distribution = None
        if is_multi_gpu_instance(args.instance_type):
            #distribution={'torch_distributed': {'enabled': True}}
            distribution = {
                'smdistributed': {
                    'dataparallel': {
                        'enabled': True
                    }
                }
            }
            logger.info(f"[DISTRIBUTION] Enabling PyTorch DDP for multi-GPU instance: {args.instance_type}")
        else:
            logger.info(f"[DISTRIBUTION] Single GPU/CPU instance ({args.instance_type}) - using standard training")

        tags=[
            {'Key': 'Project', 'Value': 'Model-7Step'},
            {'Key': 'QuickMode', 'Value': str(args.quick_mode)},
            {'Key': 'LRFinder', 'Value': str(not args.skip_lr_finder)},
            {'Key': 'WDSearch', 'Value': str(not args.skip_wd_search)},
            {'Key': 'MultiGPU', 'Value': str(is_multi_gpu_instance(args.instance_type))}
        ]
        
    estimator = PyTorch(
        entry_point='sagemaker_wrapper.py',
        source_dir=str(current_dir),
        role=args.role_arn,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        distribution=distribution,
        use_spot_instances=args.spot_training,
        max_wait=43200 if args.spot_training else None,
        max_run=36000,
        checkpoint_s3_uri=f"{args.s3_bucket}/checkpoints/{args.job_name}",
        output_path=f"{args.s3_bucket}/output/{args.job_name}",
        volume_size=args.volume_size,
        enable_sagemaker_metrics=True,
        tags=tags
    )
    return estimator
    
def main():
    # Setup unified logging
    setup_unified_logger()
    logger = get_unified_logger("sagemaker_launcher")
    
    parser = argparse.ArgumentParser(description='Launch SageMaker Model Training')
    
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
    parser.add_argument('--instance-type', default='ml.g6.12xlarge', help='Instance type')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances')
    parser.add_argument('--spot-training', action='store_true', help='Use spot instances')
    parser.add_argument('--volume-size', type=int, default=100, help='EBS volume size (GB)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    
    # 7-Step pipeline control
    parser.add_argument('--skip-lr-finder', action='store_true', help='Skip LR range test (Step 1)')
    parser.add_argument('--skip-wd-search', action='store_true', help='Skip weight decay search (Step 5)')
    parser.add_argument('--quick-mode', action='store_true', help='Quick mode for development')
    
    
    args = parser.parse_args()

    logger.info("🚀 Starting SageMaker 7-Step Model Training Job Launch")
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

    if (args.instance_count) > 1:
        distribution_instance = 'ShardedByS3Key'
    else:
        distribution_instance = 'FullyReplicated'  # Default distribution strategy

    # Configure training data input with optimal settings
    train_input = TrainingInput(
        s3_data=data_s3_path,
        distribution=distribution_instance,
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
            distribution=distribution_instance,
            s3_data_type='S3Prefix',
            input_mode=args.distribution_mode,
            compression=None
        )
        data_inputs['validation'] = val_input
        
        logger.info(f"✅ S3 data inputs configured:")
        logger.info(f"   - (train data): {data_s3_path}")
        logger.info(f"   - (validation data): {val_s3_path}")
        logger.info(f"   - Distribution: FullyReplicated") 
        logger.info(f"   - Input Mode: {args.distribution_mode}")
        logger.info(f"   - Data Type: S3Prefix")
    else:
        logger.info(f"✅ S3 data inputs configured:")
        logger.info(f"   - (train data): {data_s3_path}")
        logger.info(f"   - Distribution: FullyReplicated") 
        logger.info(f"   - Input Mode: {args.distribution_mode}")
        logger.info(f"   - Data Type: S3Prefix")
        logger.info(f"   - Note: Using single channel (train/val as subdirectories)")
    
    # Calculate optimal hyperparameters based on instance type
    logger.info(f"🧠 Calculating optimal hyperparameters for {args.instance_type}...")
    optimal_params = calculate_optimal_hyperparameters(args.instance_type, args.quick_mode)
    
    logger.info("📊 Instance Specifications:")
    logger.info(f"   - GPUs: {optimal_params['gpu_count']}")
    logger.info(f"   - GPU Memory: {optimal_params['gpu_memory_gb']}GB per GPU")
    logger.info(f"   - CPUs: {optimal_params['cpu_count']}")
    logger.info(f"   - Multi-GPU: {is_multi_gpu_instance(args.instance_type)}")
    
    # Prepare hyperparameters for 7-step training
    hyperparameters = {
        'epochs': str(args.epochs),
        'run_lr_finder': str(not args.skip_lr_finder).lower(),
        'run_wd_search': str(not args.skip_wd_search).lower(),
        'quick_mode': str(args.quick_mode).lower(),
    }
    
    logger.info("🔧 7-Step Training Configuration:")
    logger.info(f"   - LR Finder: {'Enabled' if not args.skip_lr_finder else 'Disabled'}")
    logger.info(f"   - Weight Decay Search: {'Enabled' if not args.skip_wd_search else 'Disabled'}")
    logger.info(f"   - Quick Mode: {'Enabled' if args.quick_mode else 'Disabled'}")
    logger.info(f"   - Epochs: {args.epochs}")
    
    logger.info(f"🔧 Hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"   {key}: {value}")
    
    logger.info("=" * 60)
    
    # Create estimator
    logger.info("🔧 Creating SageMaker PyTorch estimator...")
    
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
        estimator = create_sagemaker_estimator(args, hyperparameters)
        logger.info("✅ SageMaker estimator created successfully")
        
        # Launch training job with increased timeout
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
            
            # Get job name for reference
            training_job_name = estimator.latest_training_job.job_name
            logger.info(f"🎯 Training job name: {training_job_name}")
            logger.info("💡 Job is now running in the background - not streaming logs to avoid blocking")
            
        except Exception as fit_error:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
            logger.error(f"❌ SageMaker fit() call failed: {fit_error}")
            logger.error(f"❌ Error type: {type(fit_error).__name__}")
            import traceback
            logger.error(f"❌ Full error: {traceback.format_exc()}")
            raise
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
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        logger.error("Please check your AWS credentials, role permissions, and S3 bucket access")
        logger.error("💡 Common issues:")
        logger.error("   - Invalid role ARN or insufficient permissions")
        logger.error("   - S3 bucket doesn't exist or no access")
        logger.error("   - Instance type not available in region")
        logger.error("   - AWS API throttling or service issues")
        raise

if __name__ == "__main__":
        main()