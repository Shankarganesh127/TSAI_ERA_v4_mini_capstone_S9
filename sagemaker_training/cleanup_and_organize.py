#!/usr/bin/env python3
"""
SageMaker Training Folder Cleanup and Organization Script

Organizes the sagemaker_training folder according to the user's requirements:
1. Sets up AWS/SageMaker access
2. Provides main orchestrator entry point
3. Implements 7-stage pipeline selection
4. Validates dataset structure and converts if needed
5. Executes complete SageMaker training with monitoring
6. Saves models, metrics, and generates detailed graphs
"""

import os
import sys
import shutil
from pathlib import Path
import json

def setup_logger():
    """Simple logging setup"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def organize_sagemaker_folder():
    """Organize the sagemaker_training folder structure"""
    
    logger = setup_logger()
    logger.info("🧹 Organizing SageMaker training folder...")
    
    current_dir = Path(__file__).parent
    
    # Create organized structure
    folders_to_create = [
        "configs",
        "scripts", 
        "logs",
        "outputs",
        "documentation"
    ]
    
    for folder in folders_to_create:
        (current_dir / folder).mkdir(exist_ok=True)
    
    # Move files to appropriate locations
    file_organization = {
        "configs": [
            "pipeline_config.json",
            "config_examples.json"
        ],
        "scripts": [
            "setup_environment.py",
            "setup.bat",
            "setup.sh",
            "run_sagemaker.bat"
        ],
        "documentation": [
            "README.md",
            "QUICK_REFERENCE.md",
            "S3_DATASET_CONVERTER_README.md",
            "LOGGING_INTEGRATION_SUMMARY.md",
            "SIMPLIFIED_STRUCTURE.md",
            "README_OLD.md"
        ]
    }
    
    # Move files
    for folder, files in file_organization.items():
        folder_path = current_dir / folder
        for file_name in files:
            source = current_dir / file_name
            destination = folder_path / file_name
            
            if source.exists() and not destination.exists():
                try:
                    shutil.move(str(source), str(destination))
                    logger.info(f"✅ Moved {file_name} to {folder}/")
                except Exception as e:
                    logger.warning(f"⚠️ Could not move {file_name}: {e}")
    
    # Create main entry point indicator
    create_main_entry_point(current_dir, logger)
    
    # Update requirements organization
    organize_requirements(current_dir, logger)
    
    logger.info("✅ Folder organization completed!")

def create_main_entry_point(base_dir, logger):
    """Create clear main entry point"""
    
    entry_point_content = '''# SageMaker Training Pipeline - Main Entry Points

## 🚀 Primary Usage (Recommended)

### Complete Automated Pipeline
```bash
python sagemaker_orchestrator.py \\
    --role-arn arn:aws:iam::123456789012:role/SageMakerRole \\
    --source-bucket your-imagenet-bucket \\
    --use-spot
```

This single command handles:
1. ✅ AWS/SageMaker access validation
2. ✅ Dataset structure validation and conversion  
3. ✅ 7-stage pipeline parameter optimization
4. ✅ SageMaker training job launch
5. ✅ Real-time monitoring with detailed graphs
6. ✅ Model saving every epoch
7. ✅ Cost tracking and comprehensive logging

## 🛠️ Setup and Validation

### First-time Setup
```bash
python scripts/setup_environment.py --test-bucket your-bucket --quick-test
```

### Manual Components (if needed)

```bash
# Dataset conversion only
python s3_dataset_converter.py --bucket your-bucket --source-prefix ILSVRC

# Launch training only (dataset already converted)
python launch_sagemaker.py --job-name my-job --role-arn arn:... --s3-bucket s3://...

# Monitor existing job
python monitor_training.py --job-name imagenet-7stage-20241020-123456
```

## 📁 File Organization

- **Core Pipeline**: `sagemaker_orchestrator.py` (main entry)
- **Components**: `sagemaker_wrapper.py`, `launch_sagemaker.py`, `monitor_training.py`  
- **Dataset**: `s3_dataset_converter.py`
- **Config**: `configs/pipeline_config.json`
- **Setup**: `scripts/setup_environment.py`
- **Docs**: `documentation/README.md`, `documentation/QUICK_REFERENCE.md`
'''
    
    entry_file = base_dir / "MAIN_ENTRY_POINTS.md"
    
    try:
        with open(entry_file, 'w') as f:
            f.write(entry_point_content)
        logger.info("✅ Created main entry point guide")
    except Exception as e:
        logger.warning(f"⚠️ Could not create entry point guide: {e}")

def organize_requirements(base_dir, logger):
    """Organize requirements files"""
    
    # Ensure we have comprehensive requirements
    main_requirements = base_dir / "requirements.txt"
    
    enhanced_requirements = """# SageMaker Training Pipeline - Complete Requirements

# Core AWS and SageMaker
boto3>=1.26.0
botocore>=1.29.0  
sagemaker>=2.175.0

# Deep Learning Framework
torch>=2.0.0
torchvision>=0.15.0

# Data Processing
numpy>=1.21.0
pandas>=1.5.0
pillow>=9.0.0
opencv-python>=4.5.0

# Visualization and Monitoring  
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities and Logging
tqdm>=4.64.0
tabulate>=0.9.0
psutil>=5.9.0
scikit-learn>=1.0.0
pyyaml>=6.0

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
"""
    
    try:
        with open(main_requirements, 'w') as f:
            f.write(enhanced_requirements)
        logger.info("✅ Enhanced requirements.txt")
    except Exception as e:
        logger.warning(f"⚠️ Could not update requirements: {e}")

def create_usage_summary(base_dir, logger):
    """Create comprehensive usage summary"""
    
    summary_content = '''# SageMaker Training Pipeline - Usage Flow

## 🎯 Complete Workflow

### Step 1: Environment Setup
```bash
cd sagemaker_training/
python scripts/setup_environment.py --test-bucket your-bucket
```

### Step 2: Configure Pipeline  
```bash
cp configs/config_examples.json configs/pipeline_config.json
# Edit configs/pipeline_config.json with your settings
```

### Step 3: Run Complete Pipeline
```bash
python sagemaker_orchestrator.py \\
    --role-arn arn:aws:iam::ACCOUNT:role/SageMakerRole \\
    --source-bucket your-imagenet-bucket \\
    --config-file configs/pipeline_config.json
```

## 🔄 What Happens Automatically

1. **AWS Access Validation** - Checks credentials and permissions
2. **Dataset Structure Check** - Validates S3 ILSVRC format
3. **Smart Conversion** - Converts only val/test data (skips train copy)
4. **SageMaker Job Launch** - Configures and starts training
5. **7-Stage Pipeline Execution**:
   - LR Range Test → Pick LR bounds → OneCycle LR
   - Choose batch size → Tune weight-decay → Full training  
   - Monitor & iterate with comprehensive logging
6. **Real-time Monitoring** - Live metrics, cost tracking, graphs
7. **Model Persistence** - Save models every epoch to S3
8. **Performance Analytics** - Generate detailed training reports

## 📊 Outputs and Results

All outputs automatically saved to S3:
```
s3://your-bucket/sagemaker-outputs/job-name/
├── model.tar.gz              # Final trained model
├── epoch-checkpoints/        # Model saved every epoch
├── training-metrics/         # Loss, accuracy, LR curves
├── performance-graphs/       # Automated visualizations  
├── cost-analysis/            # Detailed cost breakdowns
└── training-logs/            # Comprehensive logs
```

## 🎛️ Advanced Controls

### Custom Hyperparameters
Edit `configs/pipeline_config.json`:
```json
{
  "7_stage_pipeline": {
    "stage_2_lr_bounds": {
      "manual_lr_min": 1e-4,
      "manual_lr_max": 1e-1
    }
  }
}
```

### Development Mode
```bash
python sagemaker_orchestrator.py \\
    --role-arn arn:... \\
    --source-bucket bucket \\
    --epochs 5 \\
    --instance-type ml.p3.2xlarge
```

### Monitor Existing Job
```bash
python monitor_training.py --job-name imagenet-7stage-20241020-123456
```

## 🔧 Component Integration

This pipeline integrates seamlessly with existing code:
- Uses existing `main.py`, `imagenet_training_pipeline.py`
- Preserves all 7-stage methodology sophistication
- Leverages existing `logger_setup.py`, `utils.py`
- No modifications required to existing training logic

## 💡 Key Benefits

- **Zero Code Changes** - Works with existing training code
- **Complete Automation** - Single command for entire pipeline  
- **Cost Optimization** - Spot instances, smart checkpointing
- **Professional Monitoring** - Real-time metrics and visualization
- **Comprehensive Logging** - Detailed analytics and reporting
- **Scalable Architecture** - Multi-GPU, multi-instance support
'''
    
    summary_file = base_dir / "USAGE_FLOW.md"
    
    try:
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        logger.info("✅ Created comprehensive usage flow guide")
    except Exception as e:
        logger.warning(f"⚠️ Could not create usage flow: {e}")

def main():
    logger = setup_logger()
    
    logger.info("🚀 Starting SageMaker Training Folder Cleanup")
    logger.info("=" * 60)
    
    try:
        # Organize folder structure
        organize_sagemaker_folder()
        
        # Create usage documentation
        base_dir = Path(__file__).parent
        create_usage_summary(base_dir, logger)
        
        logger.info("=" * 60)
        logger.info("🎉 SageMaker Training folder successfully organized!")
        logger.info("")
        logger.info("📋 Next Steps:")
        logger.info("   1. Review MAIN_ENTRY_POINTS.md")
        logger.info("   2. Run: python scripts/setup_environment.py")
        logger.info("   3. Configure: configs/pipeline_config.json")  
        logger.info("   4. Execute: python sagemaker_orchestrator.py --help")
        logger.info("")
        logger.info("💡 Main Entry Point: sagemaker_orchestrator.py")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())