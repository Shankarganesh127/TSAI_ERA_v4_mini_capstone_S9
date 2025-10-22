#!/usr/bin/env python3
"""
SageMaker Training Environment Setup Script

Comprehensive setup for SageMaker training pipeline including:
- AWS credentials validation
- Environment setup
- Dependencies installation  
- Configuration validation
- Initial testing
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

def setup_logger():
    """Simple logging setup"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_aws_credentials(logger):
    """Check if AWS credentials are configured"""
    logger.info("🔐 Checking AWS credentials...")
    
    try:
        import boto3
        
        # Try to get caller identity
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        logger.info(f"✅ AWS credentials found for: {identity['Arn']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ AWS credentials not found or invalid: {e}")
        logger.info("💡 Run 'aws configure' to set up credentials")
        return False

def install_dependencies(logger):
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        requirements_file = Path(__file__).parent / "requirements.txt"
        
        if requirements_file.exists():
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.warning("⚠️ requirements.txt not found, installing core dependencies")
            
            # Install core dependencies
            core_deps = ["boto3", "sagemaker", "torch", "torchvision", "matplotlib"]
            for dep in core_deps:
                cmd = [sys.executable, "-m", "pip", "install", dep]
                subprocess.run(cmd, check=True)
            
            logger.info("✅ Core dependencies installed")
            return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def validate_configuration(logger):
    """Validate pipeline configuration"""
    logger.info("⚙️ Validating configuration...")
    
    config_file = Path(__file__).parent / "pipeline_config.json"
    
    if not config_file.exists():
        logger.warning("⚠️ pipeline_config.json not found, using defaults")
        return True
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['aws', 'dataset', 'training', 'monitoring']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing configuration section: {section}")
                return False
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def test_sagemaker_access(logger):
    """Test SageMaker access"""
    logger.info("🧪 Testing SageMaker access...")
    
    try:
        import boto3
        
        sagemaker = boto3.client('sagemaker')
        
        # Test by listing training jobs (limit 1)
        response = sagemaker.list_training_jobs(MaxResults=1)
        
        logger.info("✅ SageMaker access confirmed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SageMaker access test failed: {e}")
        logger.info("💡 Check your AWS permissions for SageMaker")
        return False

def test_s3_access(logger, bucket_name=None):
    """Test S3 access"""
    logger.info("🗄️ Testing S3 access...")
    
    try:
        import boto3
        
        s3 = boto3.client('s3')
        
        if bucket_name:
            # Test specific bucket
            s3.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ S3 bucket access confirmed: {bucket_name}")
        else:
            # Test general S3 access by listing buckets
            s3.list_buckets()
            logger.info("✅ S3 access confirmed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 access test failed: {e}")
        logger.info("💡 Check your AWS permissions for S3")
        return False

def create_sample_config(logger):
    """Create sample configuration file"""
    logger.info("📝 Creating sample configuration...")
    
    config_file = Path(__file__).parent / "sample_config.json"
    
    sample_config = {
        "aws": {
            "region": "eu-west-2",
            "profile": None
        },
        "dataset": {
            "source_bucket": "your-dataset-bucket",
            "source_prefix": "ILSVRC",
            "target_prefix": "imagenet-sagemaker"
        },
        "training": {
            "instance_type": "ml.p3.8xlarge",
            "use_spot": True,
            "max_runtime": 86400,
            "epochs": 90
        },
        "monitoring": {
            "enable_detailed_logging": True,
            "save_metrics": True,
            "create_graphs": True,
            "track_costs": True
        }
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"✅ Sample configuration created: {config_file}")
        logger.info("💡 Copy sample_config.json to pipeline_config.json and customize")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample config: {e}")
        return False

def run_quick_test(logger):
    """Run quick integration test"""
    logger.info("🚀 Running quick integration test...")
    
    try:
        # Import main modules
        from sagemaker_orchestrator import SageMakerPipelineOrchestrator
        from s3_dataset_converter import S3DatasetConverter
        
        logger.info("✅ Core modules imported successfully")
        
        # Test orchestrator initialization
        orchestrator = SageMakerPipelineOrchestrator()
        logger.info("✅ Pipeline orchestrator initialized")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup SageMaker training environment')
    parser.add_argument('--test-bucket', help='Test specific S3 bucket access')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--quick-test', action='store_true', help='Run quick integration test')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    logger.info("🚀 Starting SageMaker Training Environment Setup")
    logger.info("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # Step 1: Install dependencies
    if not args.skip_deps:
        total_checks += 1
        if install_dependencies(logger):
            success_count += 1
        logger.info("-" * 60)
    
    # Step 2: Check AWS credentials
    total_checks += 1
    if check_aws_credentials(logger):
        success_count += 1
    logger.info("-" * 60)
    
    # Step 3: Test SageMaker access
    total_checks += 1
    if test_sagemaker_access(logger):
        success_count += 1
    logger.info("-" * 60)
    
    # Step 4: Test S3 access
    total_checks += 1
    if test_s3_access(logger, args.test_bucket):
        success_count += 1
    logger.info("-" * 60)
    
    # Step 5: Validate configuration
    total_checks += 1
    if validate_configuration(logger):
        success_count += 1
    logger.info("-" * 60)
    
    # Step 6: Create sample config if needed
    if not (Path(__file__).parent / "pipeline_config.json").exists():
        total_checks += 1
        if create_sample_config(logger):
            success_count += 1
        logger.info("-" * 60)
    
    # Step 7: Quick integration test
    if args.quick_test:
        total_checks += 1
        if run_quick_test(logger):
            success_count += 1
        logger.info("-" * 60)
    
    # Final summary
    logger.info("📊 Setup Summary:")
    logger.info(f"   ✅ Successful checks: {success_count}/{total_checks}")
    
    if success_count == total_checks:
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("\n💡 Next steps:")
        logger.info("   1. Review and customize pipeline_config.json")
        logger.info("   2. Set your S3 bucket and IAM role in configuration")
        logger.info("   3. Run: python sagemaker_orchestrator.py --help")
        return 0
    else:
        logger.error("❌ Some setup checks failed. Please resolve the issues above.")
        return 1

if __name__ == '__main__':
    exit(main())