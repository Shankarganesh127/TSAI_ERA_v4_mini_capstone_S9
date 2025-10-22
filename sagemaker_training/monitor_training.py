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

# Import logger from same directory
from logger_setup import setup_logger

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


class SageMakerMonitor:
    def __init__(self, region='eu-west-2'):
        """Initialize SageMaker monitor"""
        self.logger = setup_logger("sagemaker_monitor")
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.current_job_name = None
        self.logger.info(f"🔧 SageMaker Monitor initialized for region: {region}")
    
    def set_current_job(self, job_name):
        """Set the current job being monitored"""
        self.current_job_name = job_name
        self.logger.info(f"📌 Monitoring job: {job_name}")
    
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
    
    def generate_training_summary(self):
        """Generate comprehensive training summary"""
        self.logger.info("📊 Generating training summary...")
        
        # Get recent jobs for summary
        jobs = self.list_training_jobs(max_results=5)
        
        summary = {
            "generation_time": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "recent_jobs": []
        }
        
        # Process each job
        for job in jobs:
            job_info = {
                "name": job["Name"],
                "status": job["Status"],
                "instance_type": job["Instance"],
                "creation_time": job["Created"],
                "duration": job["Duration"]
            }
            
            # Get additional details
            try:
                details = self.get_job_details(job["Name"])
                if details:
                    job_info.update({
                        "training_start": details.get("TrainingStartTime"),
                        "training_end": details.get("TrainingEndTime"),
                        "instance_count": details.get("ResourceConfig", {}).get("InstanceCount", 1),
                        "volume_size": details.get("ResourceConfig", {}).get("VolumeSizeInGB", 0),
                        "hyperparameters": details.get("HyperParameters", {}),
                        "model_artifacts": details.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
                    })
                    
                    # Get metrics if available
                    metrics = self.get_job_metrics(job["Name"])
                    if metrics:
                        job_info["metrics"] = metrics
                        
            except Exception as e:
                self.logger.warning(f"Could not get details for job {job['Name']}: {e}")
            
            summary["recent_jobs"].append(job_info)
        
        # Calculate summary stats
        completed_jobs = [j for j in summary["recent_jobs"] if j["status"] == "Completed"]
        failed_jobs = [j for j in summary["recent_jobs"] if j["status"] == "Failed"]
        in_progress_jobs = [j for j in summary["recent_jobs"] if j["status"] == "InProgress"]
        
        summary.update({
            "completed_count": len(completed_jobs),
            "failed_count": len(failed_jobs),
            "in_progress_count": len(in_progress_jobs),
            "success_rate": f"{(len(completed_jobs) / len(jobs) * 100):.1f}%" if jobs else "0%"
        })
        
        self.logger.info("✅ Training summary generated")
        return summary
    
    def generate_cost_analysis(self):
        """Generate cost analysis for recent training jobs"""
        self.logger.info("💰 Generating cost analysis...")
        
        # Get recent jobs for cost analysis
        jobs = self.list_training_jobs(max_results=10)
        
        # AWS SageMaker pricing (approximate, varies by region)
        # These are rough estimates - actual pricing may vary
        instance_pricing = {
            "ml.p3.8xlarge": 14.688,  # per hour
            "ml.p3.16xlarge": 29.376,
            "ml.p3.2xlarge": 3.672,
            "ml.p2.xlarge": 1.686,
            "ml.g4dn.xlarge": 0.736,
            "ml.g4dn.2xlarge": 1.062,
            "ml.g4dn.4xlarge": 1.714,
            "ml.g4dn.8xlarge": 3.261,
            "ml.g4dn.12xlarge": 5.606,
            "ml.g4dn.16xlarge": 6.522,
            "ml.m5.large": 0.134,
            "ml.m5.xlarge": 0.269,
            "ml.m5.2xlarge": 0.538,
            "ml.m5.4xlarge": 1.075,
            "ml.c5.xlarge": 0.238,
            "ml.c5.2xlarge": 0.476,
            "ml.c5.4xlarge": 0.952
        }
        
        cost_analysis = {
            "generation_time": datetime.now().isoformat(),
            "currency": "USD",
            "note": "Costs are estimates based on standard on-demand pricing",
            "jobs": [],
            "total_estimated_cost": 0,
            "total_training_hours": 0
        }
        
        for job in jobs:
            job_cost = {
                "name": job["Name"],
                "status": job["Status"],
                "instance_type": job["Instance"],
                "estimated_cost": 0,
                "training_hours": 0,
                "spot_instance": False
            }
            
            try:
                # Get job details for more accurate cost calculation
                details = self.get_job_details(job["Name"])
                if details:
                    # Calculate training time
                    training_hours = self._calculate_training_time(details)
                    instance_count = details.get("ResourceConfig", {}).get("InstanceCount", 1)
                    instance_type = details.get("ResourceConfig", {}).get("InstanceType", job["Instance"])
                    
                    # Check if spot instance
                    managed_spot = details.get("EnableManagedSpotTraining", False)
                    spot_discount = 0.7 if managed_spot else 1.0  # Spot instances ~30% discount
                    
                    # Calculate cost
                    hourly_rate = instance_pricing.get(instance_type, 1.0)  # Default rate if unknown
                    estimated_cost = training_hours * hourly_rate * instance_count * spot_discount
                    
                    job_cost.update({
                        "training_hours": round(training_hours, 2),
                        "instance_count": instance_count,
                        "hourly_rate": hourly_rate,
                        "spot_instance": managed_spot,
                        "spot_discount": round((1 - spot_discount) * 100, 1) if managed_spot else 0,
                        "estimated_cost": round(estimated_cost, 2)
                    })
                    
                    cost_analysis["total_estimated_cost"] += estimated_cost
                    cost_analysis["total_training_hours"] += training_hours
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate cost for job {job['Name']}: {e}")
            
            cost_analysis["jobs"].append(job_cost)
        
        # Round totals
        cost_analysis["total_estimated_cost"] = round(cost_analysis["total_estimated_cost"], 2)
        cost_analysis["total_training_hours"] = round(cost_analysis["total_training_hours"], 2)
        
        # Add cost breakdown by instance type
        instance_costs = {}
        for job in cost_analysis["jobs"]:
            instance_type = job.get("instance_type", "unknown")
            if instance_type not in instance_costs:
                instance_costs[instance_type] = {"cost": 0, "hours": 0, "jobs": 0}
            
            instance_costs[instance_type]["cost"] += job.get("estimated_cost", 0)
            instance_costs[instance_type]["hours"] += job.get("training_hours", 0)
            instance_costs[instance_type]["jobs"] += 1
        
        cost_analysis["cost_by_instance_type"] = {
            k: {
                "total_cost": round(v["cost"], 2),
                "total_hours": round(v["hours"], 2),
                "job_count": v["jobs"],
                "avg_cost_per_job": round(v["cost"] / v["jobs"], 2) if v["jobs"] > 0 else 0
            }
            for k, v in instance_costs.items()
        }
        
        self.logger.info("✅ Cost analysis generated")
        return cost_analysis
    
    def generate_performance_graphs(self, output_dir):
        """Generate performance graphs and visualizations"""
        self.logger.info(f"📊 Generating performance graphs in {output_dir}...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import timedelta
            
            # Get recent jobs
            jobs = self.list_training_jobs(max_results=10)
            
            if not jobs:
                self.logger.warning("No jobs found for performance graphs")
                return
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Training Job Status Distribution
            statuses = [job["Status"] for job in jobs]
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                plt.figure(figsize=(10, 6))
                colors = {'Completed': 'green', 'Failed': 'red', 'InProgress': 'blue', 'Stopped': 'orange'}
                plot_colors = [colors.get(status, 'gray') for status in status_counts.keys()]
                
                plt.pie(status_counts.values(), labels=status_counts.keys(), colors=plot_colors, autopct='%1.1f%%')
                plt.title('Training Job Status Distribution')
                plt.savefig(output_dir / "job_status_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Instance Type Usage
            instances = [job["Instance"] for job in jobs]
            instance_counts = {}
            for instance in instances:
                instance_counts[instance] = instance_counts.get(instance, 0) + 1
            
            if instance_counts:
                plt.figure(figsize=(12, 6))
                plt.bar(instance_counts.keys(), instance_counts.values())
                plt.title('Instance Type Usage')
                plt.xlabel('Instance Type')
                plt.ylabel('Number of Jobs')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "instance_type_usage.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Training Timeline (for completed jobs)
            completed_jobs = []
            for job in jobs:
                try:
                    details = self.get_job_details(job["Name"])
                    if details and details.get("TrainingJobStatus") == "Completed":
                        if "TrainingStartTime" in details and "TrainingEndTime" in details:
                            completed_jobs.append({
                                "name": job["Name"],
                                "start": details["TrainingStartTime"],
                                "end": details["TrainingEndTime"],
                                "duration": (details["TrainingEndTime"] - details["TrainingStartTime"]).total_seconds() / 3600
                            })
                except:
                    continue
            
            if completed_jobs:
                plt.figure(figsize=(12, 8))
                for i, job in enumerate(completed_jobs):
                    plt.barh(i, job["duration"], left=0, height=0.6)
                    plt.text(job["duration"]/2, i, f"{job['duration']:.1f}h", 
                            ha='center', va='center', fontsize=8)
                
                plt.yticks(range(len(completed_jobs)), [job["name"] for job in completed_jobs])
                plt.xlabel('Duration (hours)')
                plt.title('Training Job Durations')
                plt.tight_layout()
                plt.savefig(output_dir / "training_durations.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Cost Analysis Chart (if cost data available)
            cost_analysis = self.generate_cost_analysis()
            if cost_analysis.get("cost_by_instance_type"):
                plt.figure(figsize=(12, 6))
                instance_types = list(cost_analysis["cost_by_instance_type"].keys())
                costs = [cost_analysis["cost_by_instance_type"][inst]["total_cost"] 
                        for inst in instance_types]
                
                plt.bar(instance_types, costs)
                plt.title('Estimated Costs by Instance Type')
                plt.xlabel('Instance Type')
                plt.ylabel('Estimated Cost (USD)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "cost_by_instance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"✅ Performance graphs saved to {output_dir}")
            
        except ImportError:
            self.logger.warning("matplotlib not available - skipping graph generation")
            self.logger.info("Install matplotlib with: pip install matplotlib")
        except Exception as e:
            self.logger.error(f"Error generating performance graphs: {e}")
    
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
    
    parser.add_argument('--region', type=str, default='eu-west-2',
                       help='AWS region (default: eu-west-2)')
    
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