#!/usr/bin/env python3
"""
Quick script to check SageMaker training job status

Usage:
    python check_job_status.py [job_name]
    python check_job_status.py  # Will find latest job
"""

import boto3
import sys
from datetime import datetime, timezone
from sagemaker_logging import setup_sagemaker_logger

def check_job_status(job_name=None):
    """Check the status of a SageMaker training job"""
    
    logger = setup_sagemaker_logger(__name__)
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        
        if job_name is None:
            # Find the latest job
            logger.info("🔍 Finding latest training job...")
            response = sagemaker_client.list_training_jobs(
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=10
            )
            
            if not response['TrainingJobSummaries']:
                logger.error("❌ No training jobs found")
                return False
                
            job_name = response['TrainingJobSummaries'][0]['TrainingJobName']
            logger.info(f"🎯 Latest job: {job_name}")
        
        # Get job details
        logger.info(f"📊 Checking status for: {job_name}")
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        status = response['TrainingJobStatus']
        creation_time = response['CreationTime']
        
        # Calculate elapsed time
        now = datetime.now(timezone.utc)
        elapsed = now - creation_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        logger.info(f"📅 Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"⏱️  Elapsed: {elapsed_str}")
        logger.info(f"📊 Status: {status}")
        
        # Status-specific information
        if status == 'InProgress':
            if 'TrainingStartTime' in response:
                start_time = response['TrainingStartTime']
                training_elapsed = now - start_time
                training_elapsed_str = str(training_elapsed).split('.')[0]
                logger.info(f"🚀 Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                logger.info(f"⏳ Training time: {training_elapsed_str}")
                logger.info("✅ Job is actively training!")
            else:
                logger.info("⏳ Job is still starting up (provisioning instances...)")
                
        elif status == 'Completed':
            end_time = response['TrainingEndTime']
            total_time = end_time - creation_time
            total_time_str = str(total_time).split('.')[0]
            logger.info(f"🎉 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"⏱️  Total time: {total_time_str}")
            
            if 'ModelArtifacts' in response:
                model_uri = response['ModelArtifacts']['S3ModelArtifacts']
                logger.info(f"💾 Model saved: {model_uri}")
                
        elif status == 'Failed':
            logger.error(f"❌ Job failed!")
            if 'FailureReason' in response:
                logger.error(f"   Reason: {response['FailureReason']}")
                
        elif status == 'Stopped':
            logger.warning(f"⏹️  Job was stopped")
            if 'TrainingEndTime' in response:
                end_time = response['TrainingEndTime']
                logger.info(f"🛑 Stopped: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Instance information
        if 'ResourceConfig' in response:
            instance_type = response['ResourceConfig']['InstanceType']
            instance_count = response['ResourceConfig']['InstanceCount']
            logger.info(f"💻 Instance: {instance_count}x {instance_type}")
            
        # Spot instance info
        if response.get('EnableManagedSpotTraining', False):
            logger.info("💰 Using managed spot training")
            if 'BillableTimeInSeconds' in response:
                billable_time = response['BillableTimeInSeconds']
                logger.info(f"💵 Billable time: {billable_time//3600}h {(billable_time%3600)//60}m")
        
        # Console link
        region = boto3.Session().region_name or 'us-east-1'
        console_url = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
        logger.info(f"🔗 Console: {console_url}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check job status: {e}")
        return False

def main():
    job_name = None
    if len(sys.argv) > 1:
        job_name = sys.argv[1]
    
    check_job_status(job_name)

if __name__ == '__main__':
    main()