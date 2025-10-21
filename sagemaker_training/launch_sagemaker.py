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

# Add parent directory to path for logger import
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Try to import from parent directory first, fallback to local
try:
    from logger_setup import setup_logger
except ImportError:
    # Fallback to local SageMaker logging
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
    
    parser = argparse.ArgumentParser(description='Launch SageMaker 7-step ImageNet training')
    
    # Required arguments
    parser.add_argument('--job-name', required=True, help='SageMaker training job name')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket (s3://bucket-name)')
    
    # Data configuration
    parser.add_argument('--train-data-s3', type=str, help='S3 path to training data (overrides default)')
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
            data_s3_path = f"s3:/{data_s3_path}"
        else:
            data_s3_path = f"s3://{data_s3_path}"
    
    logger.info(f"🔗 Final S3 data path: {data_s3_path}")
    
    # Create S3 data inputs for SageMaker
    try:
        from sagemaker.inputs import TrainingInput
        
        # Configure training data input with optimal settings
        train_input = TrainingInput(
            s3_data=data_s3_path,
            distribution='FullyReplicated',  # Replicate data to all instances
            s3_data_type='S3Prefix',        # Treat as directory prefix
            input_mode=args.distribution_mode,  # Keep original case: FastFile, File, or Pipe
            compression=None                 # No compression for images
        )
        
        data_inputs = {'imagenet': train_input}
        logger.info(f"✅ S3 data inputs configured:")
        logger.info(f"   - Source: {data_s3_path}")
        logger.info(f"   - Distribution: FullyReplicated") 
        logger.info(f"   - Input Mode: {args.distribution_mode}")
        logger.info(f"   - Data Type: S3Prefix")
        
    except ImportError:
        # Fallback for older SageMaker SDK versions
        logger.warning("⚠️ Using legacy S3 input format (consider upgrading SageMaker SDK)")
        data_inputs = {'imagenet': data_s3_path}
        
    except Exception as e:
        logger.error(f"❌ Failed to configure S3 inputs: {e}")
        logger.info("🔄 Using simple S3 path as fallback")
        data_inputs = {'imagenet': data_s3_path}
    
    # Build hyperparameters
    hyperparameters = {
        'epochs': args.epochs,
        'data_dir': '/opt/ml/input/data/imagenet',
        'output_dir': '/opt/ml/model',
        'run_lr_finder': str(not args.skip_lr_finder).lower(),
        'run_wd_search': str(not args.skip_wd_search).lower(),
        'quick_mode': str(args.quick_mode).lower(),
        'num_workers': args.num_workers
    }
    
    # Add optional parameters
    if args.batch_size:
        hyperparameters['batch_size'] = args.batch_size
        logger.info(f"🔧 Batch Size Override: {args.batch_size}")
    else:
        logger.info("🔄 Using automatic batch size detection (Step 4)")
        
    if args.lr_min:
        hyperparameters['lr_min'] = args.lr_min
        logger.info(f"📐 Manual LR Min: {args.lr_min:.2e}")
    if args.lr_max:
        hyperparameters['lr_max'] = args.lr_max
        logger.info(f"📐 Manual LR Max: {args.lr_max:.2e}")
    if args.weight_decay:
        hyperparameters['weight_decay'] = args.weight_decay
        logger.info(f"⚖️ Manual Weight Decay: {args.weight_decay:.2e}")
    
    # Log configuration
    logger.info("� SageMaker 7-Step ImageNet Training Configuration:")
    logger.info(f"   📝 Job Name: {args.job_name}")
    logger.info(f"   🖥️  Instance: {args.instance_type}")
    logger.info(f"   💰 Spot Training: {'Yes' if args.spot_training else 'No'}")
    logger.info(f"   📊 Epochs: {args.epochs}")
    
    logger.info("📋 7-Step Pipeline Control:")
    logger.info(f"   1️⃣ LR Range Test: {'Skip' if args.skip_lr_finder else 'Run'}")
    logger.info(f"   5️⃣ Weight Decay Search: {'Skip' if args.skip_wd_search else 'Run'}")
    logger.info(f"   🚀 Quick Mode: {'Yes' if args.quick_mode else 'No'}")
    
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
    try:
        estimator = PyTorch(
            entry_point='sagemaker_wrapper.py',
            source_dir='.',
            role=args.role_arn,
            framework_version='2.0.0', 
            py_version='py310',
            instance_count=1,
            instance_type=args.instance_type,
            hyperparameters=hyperparameters,
            use_spot_instances=args.spot_training,
            max_wait=3600 if args.spot_training else None,
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
    
    # Launch training
    try:
        logger.info("🚀 Launching SageMaker training job...")
        logger.info(f"📊 Data inputs: {data_inputs}")
        
        estimator.fit(
            inputs=data_inputs,
            job_name=args.job_name,
            wait=False
        )
        
        logger.info(f"✅ SageMaker job '{args.job_name}' launched successfully!")
        logger.info(f"📊 Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{args.job_name}")
        logger.info(f"📋 Track progress with: python monitor_training.py --job-name {args.job_name}")
        
        # Save job config
        logger.info("💾 Saving job configuration...")
        os.makedirs("job_configs", exist_ok=True)
        config = {
            'job_name': args.job_name,
            'launch_time': datetime.now().isoformat(),
            'hyperparameters': hyperparameters,
            'instance_type': args.instance_type,
            'spot_training': args.spot_training,
            'pipeline_config': {
                'lr_finder_enabled': not args.skip_lr_finder,
                'wd_search_enabled': not args.skip_wd_search,
                'quick_mode': args.quick_mode
            }
        }
        
        config_file = f"job_configs/{args.job_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Job configuration saved to: {config_file}")
        logger.info("🎉 SageMaker job launch completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ SageMaker job launch failed: {e}")
        logger.error("Please check your AWS credentials, role permissions, and S3 bucket access")
        raise

if __name__ == "__main__":
    main()