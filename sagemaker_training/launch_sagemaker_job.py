#!/usr/bin/env python3
"""
SageMaker Training Job Launcher with Spot Instance Support
This script launches your ImageNet ResNet50 training on SageMaker with spot instances.
"""

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import argparse
from datetime import datetime


def create_sagemaker_estimator(
    role,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    use_spot_instances=True,
    max_wait=3600,
    max_run=3600,
    hyperparameters=None,
    output_path=None,
    checkpoint_s3_uri=None
):
    """Create SageMaker PyTorch estimator with spot instance support"""
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'epochs': 90,
            'batch-size': 256,
            'lr-max': 0.4,
            'weight-decay': 1e-4,
            'momentum': 0.9,
            'num-workers': 4,
            'mixed-precision': True,
            'full-pipeline': True
        }
    
    # Spot instance configuration
    use_spot = use_spot_instances
    max_wait_time = max_wait if use_spot else None
    
    # Create estimator
    estimator = PyTorch(
        entry_point='sagemaker_train.py',
        source_dir='./sagemaker_training',  # Local directory with training script
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version='2.0.0',  # Compatible with your torch>=2.0.0
        py_version='py310',
        
        # Spot instance configuration
        use_spot_instances=use_spot,
        max_wait=max_wait_time,
        max_run=max_run,
        
        # Checkpointing (important for spot instances)
        checkpoint_s3_uri=checkpoint_s3_uri,
        
        # Output configuration
        output_path=output_path,
        
        # Training configuration
        hyperparameters=hyperparameters,
        
        # Resource configuration
        volume_size=100,  # GB - adjust based on dataset size
        
        # Environment variables
        environment={
            'PYTHONPATH': '/opt/ml/code',
        },
        
        # Enable network isolation for security (optional)
        enable_network_isolation=False,  # Set to True for production
        
        # Enable SageMaker debugger (optional)
        debugger_hook_config=False,  # Disable for faster training
        
        # Tags for cost tracking
        tags=[
            {'Key': 'Project', 'Value': 'TSAI-ERAv4-ImageNet'},
            {'Key': 'Model', 'Value': 'ResNet50'},
            {'Key': 'Dataset', 'Value': 'ImageNet-1K'}
        ]
    )
    
    return estimator


def setup_data_inputs(train_data_s3_uri, val_data_s3_uri=None, distribution='FullyReplicated'):
    """Setup S3 data inputs for training"""
    
    inputs = {}
    
    # Training data input
    inputs['training'] = TrainingInput(
        s3_data=train_data_s3_uri,
        distribution=distribution,  # FullyReplicated or ShardedByS3Key
        content_type='application/x-image',  # For image data
        s3_data_type='S3Prefix',
        input_mode='File'  # File mode for better performance with large datasets
    )
    
    # Validation data input (if separate)
    if val_data_s3_uri and val_data_s3_uri != train_data_s3_uri:
        inputs['validation'] = TrainingInput(
            s3_data=val_data_s3_uri,
            distribution=distribution,
            content_type='application/x-image',
            s3_data_type='S3Prefix',
            input_mode='File'
        )
    
    return inputs


def launch_training_job(
    role,
    train_data_s3_uri,
    job_name=None,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    use_spot_instances=True,
    val_data_s3_uri=None,
    output_s3_uri=None,
    checkpoint_s3_uri=None,
    hyperparameters=None,
    wait=True
):
    """Launch SageMaker training job"""
    
    # Generate job name if not provided
    if job_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f'imagenet-resnet50-{timestamp}'
    
    print(f"🚀 Launching SageMaker training job: {job_name}")
    print(f"📊 Instance: {instance_type} (count: {instance_count})")
    print(f"💰 Spot instances: {'Enabled' if use_spot_instances else 'Disabled'}")
    print(f"📂 Training data: {train_data_s3_uri}")
    
    # Create estimator
    estimator = create_sagemaker_estimator(
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        use_spot_instances=use_spot_instances,
        hyperparameters=hyperparameters,
        output_path=output_s3_uri,
        checkpoint_s3_uri=checkpoint_s3_uri
    )
    
    # Setup data inputs
    inputs = setup_data_inputs(train_data_s3_uri, val_data_s3_uri)
    
    # Start training
    print(f"🎯 Starting training with hyperparameters: {hyperparameters}")
    estimator.fit(inputs, job_name=job_name, wait=wait)
    
    if wait:
        print("✅ Training completed!")
        print(f"📦 Model artifacts: {estimator.model_data}")
        print(f"📊 Training job name: {estimator.latest_training_job.name}")
    else:
        print("🔄 Training job submitted (running asynchronously)")
        print(f"📊 Training job name: {job_name}")
    
    return estimator


def get_default_role():
    """Get default SageMaker execution role"""
    try:
        role = sagemaker.get_execution_role()
        return role
    except Exception:
        # If not running in SageMaker environment, try to get role from session
        session = sagemaker.Session()
        return session.get_caller_identity_arn().replace('user/', 'role/').replace('SageMaker', 'SageMakerExecutionRole')


def main():
    """Main function to launch SageMaker training"""
    
    parser = argparse.ArgumentParser(description='Launch SageMaker ImageNet Training')
    
    # Required arguments
    parser.add_argument('--train-data-s3', required=True, type=str,
                       help='S3 URI for training data (e.g., s3://bucket/imagenet/)')
    parser.add_argument('--role', type=str,
                       help='SageMaker execution role ARN (auto-detected if not provided)')
    
    # Optional S3 paths
    parser.add_argument('--val-data-s3', type=str,
                       help='S3 URI for validation data (uses train data if not provided)')
    parser.add_argument('--output-s3', type=str,
                       help='S3 URI for output artifacts (auto-generated if not provided)')
    parser.add_argument('--checkpoint-s3', type=str,
                       help='S3 URI for checkpoints (enables resuming from interruptions)')
    
    # Training configuration
    parser.add_argument('--job-name', type=str,
                       help='Training job name (auto-generated if not provided)')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                       choices=['ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.g4dn.4xlarge', 
                              'ml.p3.2xlarge', 'ml.p3.8xlarge', 'ml.p4d.24xlarge'],
                       help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    parser.add_argument('--no-spot', action='store_true',
                       help='Disable spot instances (use on-demand)')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr-max', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--quick-mode', action='store_true',
                       help='Run in quick validation mode')
    
    # Execution options
    parser.add_argument('--no-wait', action='store_true',
                       help='Submit job and return immediately (don\'t wait for completion)')
    
    args = parser.parse_args()
    
    # Get or auto-detect role
    role = args.role or get_default_role()
    print(f"🔐 Using SageMaker role: {role}")
    
    # Setup hyperparameters
    hyperparameters = {
        'epochs': args.epochs,
        'batch-size': args.batch_size,
        'lr-max': args.lr_max,
        'weight-decay': args.weight_decay,
        'num-workers': 4,
        'mixed-precision': True,
        'full-pipeline': not args.quick_mode,
        'quick-mode': args.quick_mode
    }
    
    # Generate default output path if not provided
    output_s3_uri = args.output_s3
    if not output_s3_uri:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        output_s3_uri = f's3://{bucket}/imagenet-training-output'
        print(f"📦 Using default output path: {output_s3_uri}")
    
    # Generate checkpoint path if not provided
    checkpoint_s3_uri = args.checkpoint_s3
    if not checkpoint_s3_uri and not args.no_spot:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        checkpoint_s3_uri = f's3://{bucket}/imagenet-training-checkpoints'
        print(f"💾 Using default checkpoint path: {checkpoint_s3_uri}")
    
    # Launch training
    try:
        estimator = launch_training_job(
            role=role,
            train_data_s3_uri=args.train_data_s3,
            job_name=args.job_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            use_spot_instances=not args.no_spot,
            val_data_s3_uri=args.val_data_s3,
            output_s3_uri=output_s3_uri,
            checkpoint_s3_uri=checkpoint_s3_uri,
            hyperparameters=hyperparameters,
            wait=not args.no_wait
        )
        
        return estimator
        
    except Exception as e:
        print(f"❌ Failed to launch training job: {str(e)}")
        raise


if __name__ == '__main__':
    estimator = main()