import os
import sys
import argparse
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.utils import unique_name_from_base
from datetime import datetime

# Import logger setup from your existing utilities
try:
    from logger_setup import get_unified_logger, setup_unified_logger
except ImportError:
    # Basic fallback if utility files are not in the path
    import logging
    logging.basicConfig(level=logging.INFO)
    setup_unified_logger = logging.getLogger

def launch_processing_job(role_arn, source_bucket, instance_type='ml.c5.9xlarge'):
    """
    Launches a SageMaker Processing job to convert ImageNet folders to WebDataset tar files.
    """
    logger = get_unified_logger("sagemaker_processing_launcher")
    
    # --- 1. Configuration ---
    
    # Bucket name (e.g., tsai-era-v4-mini-capstone)
    bucket_name = source_bucket 
    
    # S3 Prefixes for Source Data (Current ImageFolder structure)
    #s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/
    #s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/
    train_s3_prefix = f's3://{bucket_name}/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/'
    val_s3_prefix = f's3://{bucket_name}/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/'

    # S3 Prefix for Target Data (New WebDataset structure)
    # The training job will now point to this new S3 path
    output_s3_prefix = f's3://{bucket_name}/webdataset_shards/'

    # Job Name
    job_name = unique_name_from_base(f"imagenet-webdataset-conversion-{datetime.now().strftime('%Y%m%d')}")
    
    logger.info(f"🚀 Launching SageMaker Processing Job: {job_name}")
    logger.info(f"💻 Using instance type: {instance_type} (CPU optimized)")
    logger.info(f"⬇️ Reading train data from: {train_s3_prefix}")
    logger.info(f"⬆️ Writing tar files to: {output_s3_prefix}")
    
    # --- 2. Processor Setup ---
    # Using the scikit-learn container as a generic Python environment
    # Note: Ensure all required packages (webdataset, tqdm, pillow) are in the requirements.txt
    processor = ScriptProcessor(
        # Use a high-CPU, cost-effective instance for heavy data processing
        instance_type=instance_type,
        instance_count=1,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310",
        command=['python3'],
        role=role_arn,
        # Use spot instances to save cost (highly recommended for processing jobs)
        volume_kms_key=None,
    )
    
    # --- 3. Inputs and Outputs ---
    
    inputs = [
        # Mount the ImageNet train folders to a local directory named 'train'
        ProcessingInput(
            source=train_s3_prefix,
            destination='/opt/ml/processing/input/train',
            input_name='train',
            s3_data_distribution_type='FullyReplicated' # Download all data
        ),
        # Mount the ImageNet validation folders to a local directory named 'val'
        ProcessingInput(
            source=val_s3_prefix,
            destination='/opt/ml/processing/input/val',
            input_name='val',
            s3_data_distribution_type='FullyReplicated' # Download all data
        )
    ]

    outputs = [
        # Output the generated tar files to the specified S3 location
        ProcessingOutput(
            source='/opt/ml/processing/output/tars',
            destination=output_s3_prefix,
            output_name='tars',
            s3_upload_mode='EndOfJob'
        )
    ]
    
    # --- 4. Run the Job ---
    try:
        processor.run(
            code='webdataset_converter_sagemaker.py', # The script to execute
            inputs=inputs,
            outputs=outputs,
            job_name=job_name,
            wait=True, # Wait for the job to complete
            logs=True
        )
        logger.info(f"🎉 Conversion Job '{job_name}' completed successfully!")
        logger.info(f"✅ WebDataset shards are available at: {output_s3_prefix}")
        return output_s3_prefix
        
    except Exception as e:
        logger.error(f"❌ Processing job failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch SageMaker WebDataset Conversion Job")
    parser.add_argument("--role-arn", type=str, required=True,
                        help="AWS IAM Role ARN for SageMaker execution.")
    parser.add_argument("--source-bucket", type=str, required=True,
                        help="S3 bucket containing the source ImageNet folders (e.g., tsai-era-v4-mini-capstone).")
    parser.add_argument("--instance-type", type=str, default='ml.c5.9xlarge',
                        help="CPU instance type for conversion (default: ml.c5.9xlarge).")
    
    args = parser.parse_args()
    
    launch_processing_job(args.role_arn, args.source_bucket, args.instance_type)