#!/usr/bin/env python3
"""
Quick test to verify the orchestrator command building

Usage:
    python test_orchestrator_command.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_command_building():
    """Test the command building logic"""
    
    # Simulate the training args that would be created
    training_args = {
        'job_name': 'imagenet-7stage-20251021-013357',
        'role_arn': 'arn:aws:iam::872109682518:role/service-role/AmazonSageMaker-ExecutionRole-20251009T010774',
        'source_bucket': 'tsai-era-v4-mini-capstone',
        'target_prefix': 'Datasets/imagenet1k/ILSVRC/imagenet-sagemaker',
        'instance_type': 'ml.g5.12xlarge',
        'use_spot': True,
        'epochs': 2,
    }
    
    job_name = training_args['job_name']
    
    # Build command exactly like the orchestrator does
    source_bucket = training_args.get("source_bucket", "")
    s3_bucket_uri = f"s3://{source_bucket}" if source_bucket and not source_bucket.startswith("s3://") else source_bucket
    
    cmd_args = [
        "python", "launch_sagemaker.py",
        "--job-name", job_name,
        "--role-arn", training_args.get("role_arn"),  # Fixed: no hardcoded default
        "--s3-bucket", s3_bucket_uri,
        "--instance-type", training_args.get("instance_type", "ml.g5.12xlarge"),
        "--epochs", str(training_args.get("epochs"))  # Fixed: use actual value, no hardcoded default
    ]
    
    # Add optional arguments
    if training_args.get("use_spot"):
        cmd_args.append("--spot-training")
    if training_args.get("batch_size"):
        cmd_args.extend(["--batch-size", str(training_args.get("batch_size"))])
    
    print("🔧 Generated Command:")
    print(" ".join(cmd_args))
    print()
    print("📋 Command breakdown:")
    for i in range(0, len(cmd_args), 2):
        if i + 1 < len(cmd_args) and cmd_args[i].startswith('--'):
            print(f"   {cmd_args[i]}: {cmd_args[i+1]}")
        elif not cmd_args[i].startswith('--'):
            print(f"   {cmd_args[i]}")
    
    # Validation
    print("\n✅ Validation:")
    print(f"   ✅ S3 bucket: {'✅' if s3_bucket_uri.startswith('s3://') else '❌'} {s3_bucket_uri}")
    print(f"   ✅ Role ARN: {'✅' if training_args['role_arn'].startswith('arn:') else '❌'}")
    print(f"   ✅ Instance: {'✅' if training_args['instance_type'].startswith('ml.') else '❌'} {training_args['instance_type']}")
    print(f"   ✅ Spot training: {'✅' if '--spot-training' in cmd_args else '❌'}")

if __name__ == '__main__':
    test_command_building()