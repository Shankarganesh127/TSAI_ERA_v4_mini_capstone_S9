#!/usr/bin/env python3
"""
Quick retry launcher for SageMaker training with fallback instance types

Usage:
    python retry_sagemaker_training.py --job-name <original-job-name> --retry-count 3
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Retry SageMaker training with fallback options')
    parser.add_argument('--job-name', required=True, help='Original job name to retry')
    parser.add_argument('--role-arn', required=True, help='IAM role ARN')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--retry-count', type=int, default=3, help='Number of retry attempts')
    
    args = parser.parse_args()
    
    # Instance type fallback chain (from largest to smallest)
    instance_types = [
        "ml.g5.12xlarge",  # Original (large, may timeout)
        "ml.g5.4xlarge",   # Medium GPU
        "ml.g5.2xlarge",   # Smaller GPU (faster provisioning)
        "ml.g4dn.4xlarge"  # Older generation (most reliable)
    ]
    
    # Try with spot first, then without spot
    spot_options = [True, False]
    
    success = False
    
    for retry in range(args.retry_count):
        for instance_type in instance_types:
            for use_spot in spot_options:
                job_name = f"{args.job_name}-retry{retry+1}-{instance_type.replace('.', '-')}"
                if use_spot:
                    job_name += "-spot"
                
                logger.info(f"🔄 Retry {retry+1}: Attempting with {instance_type} (spot: {use_spot})")
                
                # Build command
                cmd = [
                    sys.executable, "launch_sagemaker.py",
                    "--job-name", job_name,
                    "--role-arn", args.role_arn,
                    "--s3-bucket", args.s3_bucket,
                    "--instance-type", instance_type,
                    "--epochs", str(args.epochs)
                ]
                
                if use_spot:
                    cmd.append("--spot-training")
                
                try:
                    logger.info(f"🚀 Command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=Path(__file__).parent,
                        timeout=900  # 15 minutes timeout for smaller instances
                    )
                    
                    if result.returncode == 0:
                        logger.info("✅ Training job submitted successfully!")
                        logger.info(f"🎯 Job name: {job_name}")
                        logger.info(f"💻 Instance type: {instance_type}")
                        logger.info(f"💰 Spot training: {use_spot}")
                        success = True
                        break
                    else:
                        logger.warning(f"⚠️ Failed with {instance_type} (spot: {use_spot})")
                        logger.warning(f"   STDERR: {result.stderr[:200]}...")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"⏰ Timeout with {instance_type} (spot: {use_spot})")
                
                # Brief pause between attempts
                time.sleep(5)
            
            if success:
                break
        
        if success:
            break
        
        if retry < args.retry_count - 1:
            logger.info(f"😴 Waiting 30 seconds before retry {retry+2}...")
            time.sleep(30)
    
    if not success:
        logger.error("❌ All retry attempts failed!")
        logger.error("💡 Suggestions:")
        logger.error("   - Check AWS service health")
        logger.error("   - Verify IAM permissions")
        logger.error("   - Try again later")
        sys.exit(1)
    
    logger.info("🎉 Training successfully submitted!")

if __name__ == '__main__':
    main()