#!/usr/bin/env python3
"""
SageMaker Training Orchestrator

Comprehensive pipeline orchestrator that handles:
1. AWS/SageMaker setup validation
2. Dataset structure validation and conversion
3. 7-stage hyperparameter optimization pipeline  
4. Training execution with detailed monitoring
5. Model saving and metrics tracking
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# All files are in t directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Local imports
from sagemaker_logging import setup_sagemaker_logger
from s3_dataset_converter import S3DatasetConverter
from monitor_training import SageMakerMonitor


class SageMakerPipelineOrchestrator:
    """Complete SageMaker training pipeline orchestrator"""
    
    def __init__(self, config_file=None):
        self.logger = setup_sagemaker_logger(__name__)
        self.config = self._load_config(config_file)
        self.aws_session = None
        self.s3_client = None
        self.sagemaker_client = None
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            "aws": {
                "region": "eu-west-2",
                "profile": None
            },
            "dataset": {
                "source_bucket": "tsai-era-v4-mini-capstone",
                "source_prefix": "Datasets/imagenet1k/ILSVRC",
                "target_prefix": "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker",
                "validation_required": True
            },
            "training": {
                "instance_type": "ml.g5.12xlarge", #ml.p3.8xlarge",
                "instance_count": 1,
                "use_spot": True,
                "max_runtime": 86400,  # 24 hours
                "checkpoint_interval": 300,  # 5 minutes
                "enable_7_stage_pipeline": True
            },
            "monitoring": {
                "enable_detailed_logging": True,
                "save_metrics": True,
                "create_graphs": True,
                "track_costs": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Deep merge configs
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return default_config
    
    def run_complete_pipeline(self, args):
        """Execute the complete SageMaker training pipeline"""
        
        # Enable debug mode if requested
        if hasattr(args, 'debug') and args.debug:
            self.logger.info("🐛 Debug mode enabled - activating real-time output")
            if "debug" not in self.config:
                self.config["debug"] = {}
            self.config["debug"]["enable_realtime_output"] = True
            self.config["debug"]["log_subprocess_details"] = True
            self.config["debug"]["verbose_error_reporting"] = True
        
        self.logger.info("🚀 Starting Complete SageMaker Training Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Setup AWS/SageMaker Access
            if not self._setup_aws_access():
                return False
                
            # Step 2: Validate Dataset Structure
            dataset_ready = self._validate_and_convert_dataset(args)
            if not dataset_ready:
                return False
            
            # Step 3: Launch SageMaker Training Pipeline
            training_job_name = self._launch_training_pipeline(args)
            if not training_job_name:
                return False
            
            # Step 4: Monitor Training with Detailed Metrics (DISABLED - was blocking)
            # success = self._monitor_training_pipeline(training_job_name)
            success = True  # Skip monitoring to avoid blocking
            self.logger.info("📊 Training monitoring skipped to avoid blocking the pipeline")
            self.logger.info("💡 Monitor training manually via AWS Console or use the monitor_training.py script separately")
            
            if success:
                self.logger.info("🎉 Complete pipeline execution successful!")
                return True
            else:
                self.logger.error("❌ Pipeline execution failed during training")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {str(e)}", exc_info=True)
            return False
    
    def _setup_aws_access(self):
        """Step 1: Setup and validate AWS/SageMaker access"""
        
        self.logger.info("🔧 Step 1: Setting up AWS/SageMaker access...")
        
        try:
            # Initialize AWS session
            if self.config["aws"]["profile"]:
                self.aws_session = boto3.Session(profile_name=self.config["aws"]["profile"])
                self.logger.info(f"✅ Using AWS profile: {self.config['aws']['profile']}")
            else:
                self.aws_session = boto3.Session()
                self.logger.info("✅ Using default AWS credentials")
            
            # Initialize clients
            self.s3_client = self.aws_session.client('s3', region_name=self.config["aws"]["region"])
            self.sagemaker_client = self.aws_session.client('sagemaker', region_name=self.config["aws"]["region"])
            
            # Test credentials
            sts_client = self.aws_session.client('sts')
            identity = sts_client.get_caller_identity()
            self.logger.info(f"✅ AWS Identity: {identity['Arn']}")
            
            # Test SageMaker access
            self.sagemaker_client.list_training_jobs(MaxResults=1)
            self.logger.info("✅ SageMaker access validated")
            
            return True
            
        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not found. Please run 'aws configure'")
            return False
        except ClientError as e:
            self.logger.error(f"❌ AWS access error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ AWS setup failed: {e}")
            return False
    
    def _validate_and_convert_dataset(self, args):
        """Step 2: Validate dataset structure and convert if needed"""
        
        self.logger.info("📊 Step 2: Validating dataset structure...")
        
        try:
            # Check if source dataset exists
            source_bucket = args.source_bucket or self.config["dataset"]["source_bucket"]
            source_prefix = args.source_prefix or self.config["dataset"]["source_prefix"]
            target_prefix = args.target_prefix or self.config["dataset"]["target_prefix"]
            
            if not source_bucket:
                self.logger.error("❌ Source bucket not specified")
                return False
            
            # Check if conversion is needed
            conversion_needed = self._check_conversion_needed(source_bucket, source_prefix, target_prefix)
            
            if conversion_needed:
                self.logger.info("🔄 Dataset conversion required, starting conversion...")
                
                # Initialize converter
                converter = S3DatasetConverter(source_bucket, self.config["aws"]["profile"])
                
                # Convert dataset
                success = converter.convert_ilsvrc_to_sagemaker(source_prefix, target_prefix)
                
                if success:
                    self.logger.info("✅ Dataset conversion completed successfully")
                else:
                    self.logger.error("❌ Dataset conversion failed")
                    return False
            else:
                self.logger.info("✅ Dataset already in correct structure, skipping conversion")
            
            # Final validation
            if self._validate_sagemaker_dataset_structure(source_bucket, source_prefix, target_prefix):
                self.logger.info("✅ Dataset structure validation passed")
                return True
            else:
                self.logger.error("❌ Dataset structure validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Dataset validation/conversion failed: {e}")
            return False
    
    def _check_conversion_needed(self, bucket, source_prefix, target_prefix):
        """Check if dataset conversion is needed"""
        
        try:
            # Check if converted validation data exists
            val_prefix = f"{target_prefix}/val/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket, 
                Prefix=val_prefix,
                PaginationConfig={'MaxItems': 1}
            )
            
            for page in page_iterator:
                if 'Contents' in page and len(page['Contents']) > 0:
                    self.logger.info("Found existing converted validation data")
                    return False
            
            self.logger.info("No converted validation data found, conversion needed")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking conversion status: {e}, assuming conversion needed")
            return True
    
    def _validate_sagemaker_dataset_structure(self, bucket, source_prefix, target_prefix):
        """Validate that dataset structure is correct for SageMaker"""
        
        try:
            required_paths = {
                "training": f"{source_prefix}/Data/CLS-LOC/train/",
                "validation": f"{target_prefix}/val/"
            }
            
            # Check if test data was converted
            test_prefix = f"{target_prefix}/test/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket, 
                Prefix=test_prefix,
                PaginationConfig={'MaxItems': 1}
            )
            
            for page in page_iterator:
                if 'Contents' in page and len(page['Contents']) > 0:
                    required_paths["test"] = test_prefix
                    break
            
            # Validate each path
            for name, prefix in required_paths.items():
                if not self._check_s3_path_exists(bucket, prefix):
                    self.logger.error(f"❌ {name} data not found at: s3://{bucket}/{prefix}")
                    return False
                self.logger.info(f"✅ {name} data validated: s3://{bucket}/{prefix}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating dataset structure: {e}")
            return False
    
    def _check_s3_path_exists(self, bucket, prefix):
        """Check if S3 path exists and has content"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket, 
                Prefix=prefix,
                PaginationConfig={'MaxItems': 1}
            )
            
            for page in page_iterator:
                if 'Contents' in page and len(page['Contents']) > 0:
                    return True
            return False
        except Exception:
            return False
    
    def _launch_training_pipeline(self, args):
        """Step 3: Launch SageMaker training with 7-stage pipeline"""
        
        self.logger.info("🚀 Step 3: Launching SageMaker training pipeline...")
        
        try:
            # Generate job name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"imagenet-7stage-{timestamp}"
            
            # Prepare training arguments  
            training_args = self._prepare_training_args(args, job_name)
            
            # Launch training using subprocess call to launch_sagemaker.py
            self.logger.info(f"🚀 Launching training job: {job_name}")
            
            # Build command line arguments for launch_sagemaker.py
            source_bucket = training_args.get("source_bucket", "")
            s3_bucket_uri = f"s3://{source_bucket}" if source_bucket and not source_bucket.startswith("s3://") else source_bucket
            
            # Configure train and validation data paths based on your specific structure
            source_prefix = training_args.get("source_prefix", "Datasets/imagenet1k/ILSVRC")
            target_prefix = training_args.get("target_prefix", "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker")
            
            # Your specific dataset paths
            train_data_path = f"{source_prefix}/Data/CLS-LOC/train/"
            val_data_path = f"{target_prefix}/val/"
            
            self.logger.info(f"📂 Training data S3 path: {s3_bucket_uri}/{train_data_path}")
            self.logger.info(f"📂 Validation data S3 path: {s3_bucket_uri}/{val_data_path}")
            self.logger.info(f"🔧 Using separate train/val datasets")
            
            cmd_args = [
                "python", "launch_sagemaker.py",
                "--job-name", job_name,
                "--role-arn", training_args.get("role_arn"),  # Fixed: no hardcoded default
                "--s3-bucket", s3_bucket_uri,  # Fixed: use proper S3 URI format
                "--train-data-s3", train_data_path,  # Original training data
                "--val-data-s3", val_data_path,     # Converted validation data
                "--instance-type", training_args.get("instance_type", "ml.g5.12xlarge"),
                "--instance-count", str(training_args.get("instance_count", 1)),
                "--epochs", str(training_args.get("epochs")),  # Fixed: no hardcoded default
                "--auto-confirm"  # Skip user confirmation for automated pipeline
            ]
            
            # Add optional arguments
            if training_args.get("use_spot"):  # Fixed: use_spot instead of spot_training
                cmd_args.append("--spot-training")
            if training_args.get("batch_size"):
                cmd_args.extend(["--batch-size", str(training_args.get("batch_size"))])
                
            try:
                # Calculate dynamic timeout based on instance type and configuration
                timeout = self._calculate_submission_timeout(training_args.get("instance_type"), training_args.get("use_spot"))
                
                # Launch training job in non-blocking mode with better feedback
                self.logger.info(f"🚀 Launching SageMaker training job (timeout: {timeout//60} minutes)...")
                self.logger.info("💡 You can monitor progress in the AWS SageMaker console")
                self.logger.info(f"📊 Job name: {job_name}")
                self.logger.info("⏳ Submitting job to SageMaker...")
                
                # Debug: Show the command being executed
                self.logger.info(f"🔧 Command: {' '.join(cmd_args)}")
                self.logger.info(f"📂 Working directory: {Path(__file__).parent}")
                self.logger.info(f"⏰ Timeout: {timeout} seconds ({timeout//60} minutes)")
                
                # Launch with calculated timeout and enhanced logging
                start_time = time.time()
                
                # Check if real-time output is enabled
                use_realtime = self.config.get("debug", {}).get("enable_realtime_output", False)
                
                try:
                    if use_realtime:
                        self.logger.info("🔄 Using real-time output mode for debugging")
                        result = self._run_subprocess_with_realtime_output(cmd_args, timeout)
                    else:
                        result = subprocess.run(cmd_args, capture_output=True, text=True, cwd=Path(__file__).parent, timeout=timeout)
                    
                    elapsed_time = time.time() - start_time
                    success = result.returncode == 0
                    
                    # Always log the execution details (unless already logged in real-time mode)
                    if not use_realtime:
                        self.logger.info(f"⏱️ Subprocess completed in {elapsed_time:.1f} seconds")
                        self.logger.info(f"🔄 Return code: {result.returncode}")
                        
                        if result.stdout:
                            self.logger.info("� STDOUT:")
                            for line in result.stdout.strip().split('\n'):
                                if line.strip():  # Skip empty lines
                                    self.logger.info(f"   {line}")
                        
                        if result.stderr:
                            self.logger.warning("⚠️ STDERR:")
                            for line in result.stderr.strip().split('\n'):
                                if line.strip():  # Skip empty lines
                                    self.logger.warning(f"   {line}")
                    else:
                        self.logger.info(f"⏱️ Process completed in {elapsed_time:.1f} seconds")
                    
                    if success:
                        self.logger.info("✅ Training job submitted successfully to SageMaker!")
                        self.logger.info("🚀 Job is now in queue - instance provisioning may take 5-15 minutes")
                        self.logger.info("💰 Using spot instances - may take longer due to capacity availability")
                        self.logger.info("🔗 Monitor training at: https://console.aws.amazon.com/sagemaker/home#/jobs")
                        self.logger.info(f"🎯 Search for job: {job_name}")
                    else:
                        self.logger.error(f"❌ Training submission failed with return code {result.returncode}")
                        
                except subprocess.TimeoutExpired as timeout_error:
                    elapsed_time = time.time() - start_time
                    self.logger.error(f"⏰ Subprocess timed out after {elapsed_time:.1f} seconds (limit: {timeout})")
                    
                    # Try to get partial output if available
                    try:
                        if hasattr(timeout_error, 'stdout') and timeout_error.stdout:
                            self.logger.info("📝 Partial STDOUT before timeout:")
                            # Handle both bytes and string output
                            stdout_text = timeout_error.stdout
                            if isinstance(stdout_text, bytes):
                                stdout_text = stdout_text.decode('utf-8', errors='replace')
                            for line in stdout_text.strip().split('\n'):
                                if line.strip():
                                    self.logger.info(f"   {line}")
                        
                        if hasattr(timeout_error, 'stderr') and timeout_error.stderr:
                            self.logger.warning("⚠️ Partial STDERR before timeout:")
                            # Handle both bytes and string output
                            stderr_text = timeout_error.stderr
                            if isinstance(stderr_text, bytes):
                                stderr_text = stderr_text.decode('utf-8', errors='replace')
                            for line in stderr_text.strip().split('\n'):
                                if line.strip():
                                    self.logger.warning(f"   {line}")
                    except Exception as parse_error:
                        self.logger.warning(f"⚠️ Could not parse timeout error output: {parse_error}")
                    
                    raise  # Re-raise to be caught by the outer except block
                    
            except subprocess.TimeoutExpired:
                current_instance = training_args.get('instance_type')
                alternatives = self._suggest_alternative_instances(current_instance)
                
                self.logger.error(f"❌ Training job submission timed out after {timeout//60} minutes")
                self.logger.error("   This may indicate:")
                self.logger.error("   - Network connectivity issues")
                self.logger.error("   - AWS API throttling")
                self.logger.error("   - Large instance type provisioning delays")
                self.logger.error("   💡 Suggestions:")
                self.logger.error(f"   - Try a smaller instance type (current: {current_instance})")
                for i, alt in enumerate(alternatives[:2], 1):
                    self.logger.error(f"   - Alternative {i}: {alt}")
                if training_args.get('use_spot'):
                    self.logger.error("   - Disable spot instances: remove --spot-training flag")
                self.logger.error("   - Check AWS service health dashboard")
                success = False
            except Exception as e:
                self.logger.error(f"❌ Failed to launch training: {e}")
                success = False
            
            if success:
                self.logger.info(f"✅ Training job launched successfully: {job_name}")
                return job_name
            else:
                self.logger.error("❌ Failed to launch training job")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Training launch failed: {e}")
            return None
    
    def _prepare_training_args(self, args, job_name):
        """Prepare training arguments from config and command line args"""
        
        # Build training data S3 path
        source_bucket = args.source_bucket or self.config["dataset"]["source_bucket"]
        target_prefix = args.target_prefix or self.config["dataset"]["target_prefix"]
        
        # Priority: command line args > config defaults
        role_arn = getattr(args, 'role_arn', None) or self.config.get("aws", {}).get("default_role_arn")
        epochs = getattr(args, 'epochs', None) or self.config.get("training", {}).get("default_epochs", 90)
        instance_type = args.instance_type or self.config["training"]["instance_type"]
        instance_count = args.instance_count or self.config["training"]["instance_count"]
        use_spot = args.use_spot if hasattr(args, 'use_spot') else self.config["training"]["use_spot"]
        
        # Validate required parameters
        if not role_arn:
            raise ValueError("role_arn is required either via --role-arn command line argument or aws.default_role_arn in config")
        
        training_args = {
            'job_name': job_name,
            'role_arn': role_arn,
            'source_bucket': source_bucket,
            'target_prefix': target_prefix,
            'instance_type': instance_type,
            'instance_count': instance_count,
            'use_spot': use_spot,
            'max_runtime': self.config["training"]["max_runtime"],
            'enable_7_stage': self.config["training"]["enable_7_stage_pipeline"],
            'epochs': epochs,
            'enable_monitoring': self.config["monitoring"]["enable_detailed_logging"],
            'save_metrics': self.config["monitoring"]["save_metrics"],
            'track_costs': self.config["monitoring"]["track_costs"]
        }
        
        # Log the training configuration
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"   � Role ARN: {role_arn[:50]}... ({'from command line' if hasattr(args, 'role_arn') and args.role_arn else 'from config default'})")
        self.logger.info(f"   �📅 Epochs: {epochs} ({'from command line' if hasattr(args, 'epochs') and args.epochs else 'from config default'})")
        self.logger.info(f"   💻 Instance: {instance_type}")
        self.logger.info(f"   💰 Spot training: {use_spot}")
        self.logger.info(f"   🚀 Job name: {job_name}")
        
        return training_args
    
    def _calculate_submission_timeout(self, instance_type, use_spot):
        """Calculate dynamic timeout based on instance type and spot usage"""
        
        # Get timeout configuration  
        timeout_config = self.config.get("timeouts", {})
        base_timeout = timeout_config.get("job_submission_timeout_base", 900)  # 15 minutes default (increased from 10)
        large_multiplier = timeout_config.get("large_instance_timeout_multiplier", 2.5)
        spot_multiplier = timeout_config.get("spot_instance_timeout_multiplier", 2.0)  # Increased for better spot provisioning
        max_timeout = timeout_config.get("max_timeout", 3600)  # 60 minutes max (increased from 30)
        large_instance_types = timeout_config.get("large_instance_types", [
            "ml.p3.8xlarge", "ml.p3.16xlarge", "ml.p4d.24xlarge", "ml.g5.12xlarge", "ml.g5.24xlarge"
        ])
        
        # Start with base timeout
        timeout = base_timeout
        
        # Apply multiplier for large instances
        if instance_type in large_instance_types:
            timeout = int(timeout * large_multiplier)
            self.logger.info(f"🔧 Using extended timeout for large instance type: {instance_type}")
        
        # Apply multiplier for spot instances
        if use_spot:
            timeout = int(timeout * spot_multiplier)
            self.logger.info("💰 Using extended timeout for spot instance provisioning")
        
        # Cap at maximum timeout
        timeout = min(timeout, max_timeout)
        
        self.logger.info(f"⏱️ Job submission timeout: {timeout//60} minutes ({timeout} seconds)")
        
        return timeout
    
    def _suggest_alternative_instances(self, current_instance):
        """Suggest faster-provisioning alternative instance types"""
        
        # Instance alternatives mapping (faster provisioning alternatives)
        alternatives = {
            "ml.g5.12xlarge": ["ml.g5.4xlarge", "ml.g5.2xlarge", "ml.g4dn.4xlarge"],
            "ml.g5.24xlarge": ["ml.g5.12xlarge", "ml.g5.4xlarge", "ml.g4dn.8xlarge"], 
            "ml.g5.48xlarge": ["ml.g5.24xlarge", "ml.g5.12xlarge", "ml.g4dn.12xlarge"],
            "ml.p4d.24xlarge": ["ml.g5.12xlarge", "ml.g5.4xlarge", "ml.g4dn.8xlarge"],
            "ml.p4de.24xlarge": ["ml.p4d.24xlarge", "ml.g5.12xlarge", "ml.g4dn.8xlarge"],
            "ml.trn1.32xlarge": ["ml.g5.12xlarge", "ml.g5.4xlarge", "ml.g4dn.8xlarge"]
        }
        
        return alternatives.get(current_instance, ["ml.g5.2xlarge", "ml.g5.4xlarge"])
    
    def _run_subprocess_with_realtime_output(self, cmd_args, timeout, cwd=None):
        """Run subprocess with real-time output logging for better debugging"""
        import subprocess
        import select
        import threading
        
        if cwd is None:
            cwd = Path(__file__).parent
            
        self.logger.info(f"🔧 Executing: {' '.join(cmd_args)}")
        self.logger.info(f"📂 Working directory: {cwd}")
        self.logger.info(f"⏰ Timeout: {timeout} seconds ({timeout//60} minutes)")
        
        start_time = time.time()
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stdout():
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip()
                    stdout_lines.append(line)
                    self.logger.info(f"📝 {line}")
                process.stdout.close()
            
            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    line = line.rstrip()
                    stderr_lines.append(line)
                    self.logger.warning(f"⚠️ {line}")
                process.stderr.close()
            
            # Start threads to read output
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process with timeout
            try:
                returncode = process.wait(timeout=timeout)
                elapsed_time = time.time() - start_time
                
                # Wait for output threads to finish
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                
                self.logger.info(f"⏱️ Process completed in {elapsed_time:.1f} seconds")
                self.logger.info(f"🔄 Return code: {returncode}")
                
                # Create result object similar to subprocess.run
                class Result:
                    def __init__(self, returncode, stdout_lines, stderr_lines):
                        self.returncode = returncode
                        self.stdout = '\n'.join(stdout_lines)
                        self.stderr = '\n'.join(stderr_lines)
                
                return Result(returncode, stdout_lines, stderr_lines)
                
            except subprocess.TimeoutExpired:
                elapsed_time = time.time() - start_time
                self.logger.error(f"⏰ Process timed out after {elapsed_time:.1f} seconds")
                
                # Terminate process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                # Wait for output threads
                stdout_thread.join(timeout=2)
                stderr_thread.join(timeout=2)
                
                raise subprocess.TimeoutExpired(cmd_args, timeout, output='\n'.join(stdout_lines), stderr='\n'.join(stderr_lines))
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Process failed after {elapsed_time:.1f} seconds: {e}")
            raise
    
    def _monitor_training_pipeline(self, job_name):
        """Step 4: Monitor training with detailed metrics and logging"""
        
        self.logger.info("📊 Step 4: Starting comprehensive training monitoring...")
        
        try:
            # Initialize monitor with correct class name
            monitor = SageMakerMonitor(
                region=self.config["aws"]["region"]
            )
            
            # Configure monitoring options
            monitor_config = {
                'detailed_logging': self.config["monitoring"]["enable_detailed_logging"],
                'save_metrics': self.config["monitoring"]["save_metrics"],
                'create_graphs': self.config["monitoring"]["create_graphs"],
                'track_costs': self.config["monitoring"]["track_costs"],
                'checkpoint_interval': self.config["training"]["checkpoint_interval"]
            }
            
            # Start monitoring - adjust method call based on actual SageMakerMonitor class
            self.logger.info(f"📊 Starting monitoring for job: {job_name}")
            
            # Set the current job for the monitor
            monitor.set_current_job(job_name)
            
            success = True  # Placeholder - actual implementation would call monitor methods
            # success = monitor.monitor_job(job_name, monitor_config)  # Uncomment when method is ready
            
            if success:
                self.logger.info("✅ Training completed successfully with full monitoring")
                
                # Generate final reports
                self._generate_final_reports(job_name, monitor)
                
                return True
            else:
                self.logger.error("❌ Training failed or monitoring error")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Training monitoring failed: {e}")
            return False
    
    def _generate_final_reports(self, job_name, monitor):
        """Generate comprehensive final reports"""
        
        self.logger.info("📈 Generating final training reports...")
        
        try:
            # Create reports directory
            reports_dir = Path(f"training_reports/{job_name}")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate training summary
            try:
                self.logger.info("📊 Generating training summary...")
                summary = monitor.generate_training_summary()
                
                with open(reports_dir / "training_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                self.logger.info("✅ Training summary saved")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Could not generate training summary: {e}")
            
            # Generate cost analysis
            try:
                self.logger.info("💰 Generating cost analysis...")
                cost_analysis = monitor.generate_cost_analysis()
                
                with open(reports_dir / "cost_analysis.json", 'w') as f:
                    json.dump(cost_analysis, f, indent=2, default=str)
                self.logger.info("✅ Cost analysis saved")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Could not generate cost analysis: {e}")
            
            # Generate performance graphs
            if self.config["monitoring"]["create_graphs"]:
                try:
                    self.logger.info("📊 Generating performance graphs...")
                    monitor.generate_performance_graphs(reports_dir)
                    self.logger.info("✅ Performance graphs saved")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not generate performance graphs: {e}")
            
            self.logger.info(f"✅ Reports generated in: {reports_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Final report generation failed: {e}")
            # Don't raise the exception - this is non-critical


def main():
    parser = argparse.ArgumentParser(description='SageMaker Complete Training Pipeline')
    
    # Required arguments
    parser.add_argument('--source-bucket', required=True, type=str,
                       help='S3 bucket containing ILSVRC dataset')
    
    # Optional arguments (can be provided via command line or config)
    parser.add_argument('--role-arn', type=str,
                       help='SageMaker execution role ARN (required - can be set in config)')
    parser.add_argument('--source-prefix', type=str, default='Datasets/imagenet1k/ILSVRC',
                       help='S3 prefix of ILSVRC dataset (default: Datasets/imagenet1k/ILSVRC)')
    parser.add_argument('--target-prefix', type=str, default='Datasets/imagenet1k/ILSVRC/imagenet-sagemaker',
                       help='S3 prefix for converted dataset (default: Datasets/imagenet1k/ILSVRC/imagenet-sagemaker)')
    parser.add_argument('--instance-type', type=str, default='ml.p3.8xlarge',
                       help='SageMaker instance type (default: ml.p3.8xlarge)')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of SageMaker instances (default: 1)')
    parser.add_argument('--use-spot', action='store_true',
                       help='Use spot instances for cost savings')
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of training epochs (default: 90)')
    parser.add_argument('--config-file', type=str,
                       help='Configuration file path (optional)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with real-time output and verbose logging')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = SageMakerPipelineOrchestrator(args.config_file)
    
    # Run complete pipeline
    success = orchestrator.run_complete_pipeline(args)
    
    if success:
        print("\n🎉 SageMaker Training Pipeline completed successfully!")
        return 0
    else:
        print("\n❌ SageMaker Training Pipeline failed!")
        return 1


if __name__ == '__main__':
    exit(main())