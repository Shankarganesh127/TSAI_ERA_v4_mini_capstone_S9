#!/usr/bin/env python3
"""
Simple SageMaker Job Runner

Usage:
    python run_sagemaker_job.py --role-arn <your-role-arn> --bucket <your-bucket>
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run SageMaker ImageNet Training Job')
    
    # Required arguments
    parser.add_argument('--role-arn', required=True, 
                       help='SageMaker execution role ARN')
    parser.add_argument('--bucket', required=True,
                       help='S3 bucket name (without s3:// prefix)')
    
    # Optional arguments
    parser.add_argument('--job-name', 
                       default=f"imagenet-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}", 
                       help='SageMaker job name')
    parser.add_argument('--instance-type', default='ml.p3.2xlarge',
                       help='Instance type (default: ml.p3.2xlarge)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs (default: 1)')
    parser.add_argument('--use-spot', action='store_true',
                       help='Use spot instances')
    
    args = parser.parse_args()
    
    # Build command for sagemaker_orchestrator
    cmd = [
        sys.executable, 'sagemaker_orchestrator.py',
        '--role-arn', args.role_arn,
        '--source-bucket', args.bucket,
        '--epochs', str(args.epochs),
        '--instance-type', args.instance_type
    ]
    
    if args.use_spot:
        cmd.append('--use-spot')
    
    print("🚀 Running SageMaker Job with:")
    print(f"   Job Name: {args.job_name}")
    print(f"   Instance Type: {args.instance_type}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Spot Instances: {args.use_spot}")
    print(f"   Bucket: s3://{args.bucket}")
    print(f"   Command: {' '.join(cmd)}")
    
    # Run the orchestrator
    import subprocess
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()