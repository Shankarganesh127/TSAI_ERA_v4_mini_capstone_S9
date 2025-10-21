#!/usr/bin/env python3
"""
SageMaker Job Monitor - Check status of running/submitted jobs
"""

import boto3
import json
import time
from datetime import datetime, timedelta
import argparse

def check_sagemaker_jobs(job_name_pattern=None, max_jobs=10):
    """Check SageMaker training job status"""
    
    print("🔍 SageMaker Job Monitor")
    print("=" * 60)
    
    try:
        # Initialize SageMaker client
        sagemaker = boto3.client('sagemaker')
        print("✅ Connected to AWS SageMaker")
        
        # List training jobs
        if job_name_pattern:
            print(f"🔍 Searching for jobs matching: {job_name_pattern}")
            response = sagemaker.list_training_jobs(
                MaxResults=max_jobs,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
        else:
            print(f"📋 Listing recent {max_jobs} training jobs...")
            response = sagemaker.list_training_jobs(
                MaxResults=max_jobs,
                SortBy='CreationTime', 
                SortOrder='Descending'
            )
        
        jobs = response['TrainingJobSummaries']
        
        if not jobs:
            print("❌ No training jobs found")
            return
            
        print(f"\n📊 Found {len(jobs)} training jobs:")
        print("-" * 60)
        
        for i, job in enumerate(jobs, 1):
            job_name = job['TrainingJobName']
            status = job['TrainingJobStatus']
            creation_time = job['CreationTime']
            
            # Skip if pattern specified and doesn't match
            if job_name_pattern and job_name_pattern.lower() not in job_name.lower():
                continue
            
            # Format creation time
            time_str = creation_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            age = datetime.now(creation_time.tzinfo) - creation_time
            age_str = str(age).split('.')[0]  # Remove microseconds
            
            # Status emoji
            status_emoji = {
                'InProgress': '🚀',
                'Completed': '✅', 
                'Failed': '❌',
                'Stopping': '⏹️',
                'Stopped': '⏸️',
                'Starting': '⏳'
            }.get(status, '❓')
            
            print(f"{i}. {status_emoji} {job_name}")
            print(f"   Status: {status}")
            print(f"   Created: {time_str} (age: {age_str})")
            
            # Get detailed info for active jobs
            if status in ['InProgress', 'Starting']:
                try:
                    detail = sagemaker.describe_training_job(TrainingJobName=job_name)
                    
                    # Instance info
                    instance_type = detail['ResourceConfig']['InstanceType']
                    instance_count = detail['ResourceConfig']['InstanceCount']
                    print(f"   Instance: {instance_count}x {instance_type}")
                    
                    # Spot instance info
                    if detail.get('EnableManagedSpotTraining'):
                        print(f"   💰 Spot Training: Enabled")
                    
                    # Secondary status
                    secondary_status = detail.get('SecondaryStatus', 'Unknown')
                    print(f"   Stage: {secondary_status}")
                    
                    # Training time
                    if 'TrainingStartTime' in detail:
                        start_time = detail['TrainingStartTime']
                        training_duration = datetime.now(start_time.tzinfo) - start_time
                        print(f"   Runtime: {str(training_duration).split('.')[0]}")
                    
                    # Billable time (for spot instances)
                    if 'BillableTimeInSeconds' in detail:
                        billable_minutes = detail['BillableTimeInSeconds'] // 60
                        print(f"   💳 Billable time: {billable_minutes} minutes")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not get details: {e}")
            
            elif status == 'Failed':
                try:
                    detail = sagemaker.describe_training_job(TrainingJobName=job_name)
                    failure_reason = detail.get('FailureReason', 'Unknown')
                    print(f"   💥 Failure: {failure_reason}")
                except:
                    pass
            
            print()
        
        # Show how to get logs
        print("📋 To get logs for a specific job, use:")
        print("   aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name {job-name}/algo-1-{timestamp}")
        print("\n🔗 AWS Console: https://console.aws.amazon.com/sagemaker/home#/jobs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check AWS credentials: aws configure list")
        print("   - Verify region: aws configure get region")
        print("   - Test access: aws sts get-caller-identity")


def monitor_job_realtime(job_name, refresh_interval=30):
    """Monitor a specific job in real-time"""
    
    print(f"👁️ Monitoring job: {job_name}")
    print(f"🔄 Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    try:
        sagemaker = boto3.client('sagemaker')
        
        last_status = None
        last_secondary_status = None
        
        while True:
            try:
                response = sagemaker.describe_training_job(TrainingJobName=job_name)
                
                status = response['TrainingJobStatus']
                secondary_status = response.get('SecondaryStatus', 'Unknown')
                
                # Only print if status changed
                if status != last_status or secondary_status != last_secondary_status:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    status_emoji = {
                        'InProgress': '🚀',
                        'Completed': '✅', 
                        'Failed': '❌',
                        'Stopping': '⏹️',
                        'Stopped': '⏸️',
                        'Starting': '⏳'
                    }.get(status, '❓')
                    
                    print(f"[{timestamp}] {status_emoji} {status} - {secondary_status}")
                    
                    # Show additional info for certain statuses
                    if status == 'Starting' and secondary_status == 'Starting':
                        print("            ⏳ Launching instance...")
                    elif status == 'InProgress' and secondary_status == 'Downloading':
                        print("            📥 Downloading training data...")
                    elif status == 'InProgress' and secondary_status == 'Training':
                        print("            🏋️ Training model...")
                    elif status == 'InProgress' and secondary_status == 'Uploading':
                        print("            📤 Uploading results...")
                    elif status == 'Failed':
                        failure_reason = response.get('FailureReason', 'Unknown')
                        print(f"            💥 {failure_reason}")
                        break
                    elif status == 'Completed':
                        print("            🎉 Training completed successfully!")
                        break
                    
                    last_status = status
                    last_secondary_status = secondary_status
                
                # Exit if job is done
                if status in ['Completed', 'Failed', 'Stopped']:
                    break
                    
                time.sleep(refresh_interval)
                
            except KeyboardInterrupt:
                print("\n⏸️ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error monitoring job: {e}")
                time.sleep(refresh_interval)
                
    except Exception as e:
        print(f"❌ Failed to start monitoring: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monitor SageMaker training jobs')
    parser.add_argument('--job-name', type=str, help='Specific job name to monitor')
    parser.add_argument('--pattern', type=str, help='Job name pattern to search for')
    parser.add_argument('--monitor', action='store_true', help='Real-time monitoring mode')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum jobs to list')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval for monitoring (seconds)')
    
    args = parser.parse_args()
    
    if args.monitor and args.job_name:
        monitor_job_realtime(args.job_name, args.refresh)
    else:
        check_sagemaker_jobs(args.pattern or args.job_name, args.max_jobs)


if __name__ == "__main__":
    main()