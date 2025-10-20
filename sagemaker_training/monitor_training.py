#!/usr/bin/env python3
"""
SageMaker Training Job Monitor
Monitor and manage your ImageNet training jobs on SageMaker.
"""

import boto3
import argparse
import time
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for logger import
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from logger_setup import setup_logger

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


class SageMakerMonitor:
    def __init__(self, region='us-east-1'):
        """Initialize SageMaker monitor"""
        self.logger = setup_logger("sagemaker_monitor")
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.logger.info(f"🔧 SageMaker Monitor initialized for region: {region}")
    
    def list_training_jobs(self, status_filter=None, max_results=10):
        """List training jobs with optional status filter"""
        
        self.logger.info(f"📋 Listing training jobs (max: {max_results})")
        if status_filter:
            self.logger.info(f"   Filter: {status_filter}")
        
        params = {
            'MaxResults': max_results,
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending'
        }
        
        if status_filter:
            params['StatusEquals'] = status_filter
        
        response = self.sagemaker.list_training_jobs(**params)
        
        jobs = []
        for job in response['TrainingJobSummaries']:
            jobs.append({
                'Name': job['TrainingJobName'],
                'Status': job['TrainingJobStatus'],
                'Created': job['CreationTime'].strftime('%Y-%m-%d %H:%M'),
                'Duration': self._calculate_duration(job),
                'Instance': job.get('ResourceConfig', {}).get('InstanceType', 'N/A')
            })
        
        return jobs
    
    def get_job_details(self, job_name):
        """Get detailed information about a training job"""
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            return response
        except Exception as e:
            print(f"Error getting job details: {str(e)}")
            return None
    
    def get_job_logs(self, job_name, lines=50):
        """Get recent logs for a training job"""
        log_group = '/aws/sagemaker/TrainingJobs'
        
        try:
            # Get log streams for this job
            streams_response = self.logs.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                orderBy='LastEventTime',
                descending=True
            )
            
            if not streams_response['logStreams']:
                return "No logs found for this job."
            
            # Get logs from the most recent stream
            stream_name = streams_response['logStreams'][0]['logStreamName']
            
            logs_response = self.logs.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=lines,
                startFromHead=False
            )
            
            # Format logs
            log_lines = []
            for event in logs_response['events']:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000.0)
                log_lines.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} {event['message']}")
            
            return '\n'.join(log_lines)
            
        except Exception as e:
            return f"Error getting logs: {str(e)}"
    
    def stop_training_job(self, job_name):
        """Stop a running training job"""
        try:
            self.sagemaker.stop_training_job(TrainingJobName=job_name)
            print(f"✅ Stopped training job: {job_name}")
            return True
        except Exception as e:
            print(f"❌ Error stopping job: {str(e)}")
            return False
    
    def get_job_metrics(self, job_name):
        """Get training metrics for a job"""
        job_details = self.get_job_details(job_name)
        if not job_details:
            return None
        
        metrics = {
            'job_name': job_name,
            'status': job_details['TrainingJobStatus'],
            'instance_type': job_details['ResourceConfig']['InstanceType'],
            'instance_count': job_details['ResourceConfig']['InstanceCount'],
            'training_time': self._calculate_training_time(job_details),
            'billable_seconds': job_details.get('BillableTimeInSeconds', 0),
        }
        
        # Add hyperparameters
        if 'HyperParameters' in job_details:
            metrics['hyperparameters'] = job_details['HyperParameters']
        
        # Add final metrics if available
        if 'FinalMetricDataList' in job_details:
            final_metrics = {}
            for metric in job_details['FinalMetricDataList']:
                final_metrics[metric['MetricName']] = metric['Value']
            metrics['final_metrics'] = final_metrics
        
        return metrics
    
    def _calculate_duration(self, job):
        """Calculate job duration"""
        start_time = job['CreationTime']
        end_time = job.get('TrainingEndTime', datetime.now(start_time.tzinfo))
        
        duration = end_time - start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def _calculate_training_time(self, job_details):
        """Calculate total training time"""
        if 'TrainingStartTime' in job_details and 'TrainingEndTime' in job_details:
            duration = job_details['TrainingEndTime'] - job_details['TrainingStartTime']
            return duration.total_seconds() / 3600  # hours
        return 0


def main():
    parser = argparse.ArgumentParser(description='Monitor SageMaker Training Jobs')
    
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region (default: us-east-1)')
    
    # Commands
    parser.add_argument('--list', action='store_true',
                       help='List recent training jobs')
    parser.add_argument('--status', type=str, 
                       choices=['InProgress', 'Completed', 'Failed', 'Stopped'],
                       help='Filter jobs by status')
    parser.add_argument('--details', type=str,
                       help='Get details for specific job name')
    parser.add_argument('--logs', type=str,
                       help='Get logs for specific job name')
    parser.add_argument('--stop', type=str,
                       help='Stop specific training job')
    parser.add_argument('--metrics', type=str,
                       help='Get metrics for specific job name')
    
    # Options
    parser.add_argument('--max-results', type=int, default=10,
                       help='Maximum number of jobs to show (default: 10)')
    parser.add_argument('--log-lines', type=int, default=50,
                       help='Number of log lines to show (default: 50)')
    parser.add_argument('--watch', action='store_true',
                       help='Watch job status (refresh every 30 seconds)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SageMakerMonitor(args.region)
    
    try:
        if args.list or (not any([args.details, args.logs, args.stop, args.metrics])):
            # List training jobs
            print("📊 SageMaker Training Jobs")
            print("=" * 50)
            
            jobs = monitor.list_training_jobs(args.status, args.max_results)
            
            if jobs:
                headers = ['Name', 'Status', 'Created', 'Duration', 'Instance']
                table_data = [[job[h] for h in ['Name', 'Status', 'Created', 'Duration', 'Instance']] for job in jobs]
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
            else:
                print("No training jobs found.")
            
            if args.watch:
                print("\n🔄 Watching for updates (Ctrl+C to stop)...")
                while True:
                    time.sleep(30)
                    print(f"\n--- Refreshed at {datetime.now().strftime('%H:%M:%S')} ---")
                    jobs = monitor.list_training_jobs(args.status, args.max_results)
                    if jobs:
                        table_data = [[job[h] for h in ['Name', 'Status', 'Created', 'Duration', 'Instance']] for job in jobs]
                        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        elif args.details:
            # Get job details
            print(f"📋 Training Job Details: {args.details}")
            print("=" * 50)
            
            details = monitor.get_job_details(args.details)
            if details:
                print(f"Status: {details['TrainingJobStatus']}")
                print(f"Instance: {details['ResourceConfig']['InstanceType']} (x{details['ResourceConfig']['InstanceCount']})")
                print(f"Created: {details['CreationTime']}")
                
                if 'TrainingStartTime' in details:
                    print(f"Started: {details['TrainingStartTime']}")
                if 'TrainingEndTime' in details:
                    print(f"Ended: {details['TrainingEndTime']}")
                
                if 'HyperParameters' in details:
                    print("\nHyperparameters:")
                    for k, v in details['HyperParameters'].items():
                        print(f"  {k}: {v}")
                
                if 'ModelArtifacts' in details:
                    print(f"\nModel Artifacts: {details['ModelArtifacts']['S3ModelArtifacts']}")
        
        elif args.logs:
            # Get job logs
            print(f"📜 Training Logs: {args.logs}")
            print("=" * 50)
            
            logs = monitor.get_job_logs(args.logs, args.log_lines)
            print(logs)
        
        elif args.stop:
            # Stop training job
            print(f"🛑 Stopping Training Job: {args.stop}")
            print("=" * 50)
            
            success = monitor.stop_training_job(args.stop)
            if success:
                print("Job stop request submitted.")
        
        elif args.metrics:
            # Get job metrics
            print(f"📊 Training Metrics: {args.metrics}")
            print("=" * 50)
            
            metrics = monitor.get_job_metrics(args.metrics)
            if metrics:
                print(json.dumps(metrics, indent=2, default=str))
    
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    main()