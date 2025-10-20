# 📁 Complete File Reference Guide

**Comprehensive documentation of every file in the SageMaker training pipeline with detailed usage instructions.**

## 🎯 Main Entry Points

### `sagemaker_orchestrator.py` ⭐ **PRIMARY ENTRY POINT**
**Purpose**: Complete 9-step automated pipeline orchestration with model replacement
**Usage**:
```bash
# Complete automated pipeline (recommended)
python sagemaker_orchestrator.py

# With custom configuration
python sagemaker_orchestrator.py --config configs/pipeline_config.json
```
**Features**:
- ✅ AWS environment validation
- ✅ Automatic dataset conversion (if needed)
- ✅ SageMaker training launch
- ✅ Model replacement system activation
- ✅ Real-time monitoring
- ✅ Spot instance handling
- ✅ Final model archiving
- ✅ Training summary generation
- ✅ Cleanup and organization

**Output**: Complete trained models with automatic replacement, logs, and deployment archives
**README**: [`MAIN_ENTRY_POINTS.md`](MAIN_ENTRY_POINTS.md)

---

### `launch_sagemaker.py` 
**Purpose**: Direct SageMaker training job launcher with model replacement
**Usage**:
```bash
# Standard training with model replacement
python launch_sagemaker.py \
    --job-name "imagenet-training" \
    --role-arn "arn:aws:iam::123:role/SageMakerRole" \
    --train-data-s3 "s3://bucket/data/" \
    --enable-model-replacement

# Quick testing
python launch_sagemaker.py \
    --job-name "test-run" \
    --instance-type "ml.p3.2xlarge" \
    --epochs 5 \
    --pipeline-stage "lr_range_test"
```
**Features**:
- ☁️ SageMaker training job creation
- 🔄 Model replacement integration  
- 💰 Spot instance support
- ⚙️ 7-stage pipeline selection
- 📊 Real-time job monitoring

**Output**: SageMaker training job with automatic model replacement
**README**: [`USAGE_FLOW.md`](USAGE_FLOW.md)

## 🔄 Model Management System

### `model_saver.py` 🏆 **CORE MODEL REPLACEMENT**
**Purpose**: Automatic model saving with epoch-based replacement logic
**Key Features**:
```python
# Automatic usage - no direct calls needed
# Integrated into training pipeline automatically

# What it does:
# ✅ Saves model every epoch
# ✅ Replaces previous epoch model (no accumulation)
# ✅ Tracks best model by accuracy
# ✅ Creates final training archive
# ✅ Generates training summary
```
**Classes**:
- `EpochModelSaver`: Main model management class
- `ModelReplacementConfig`: Configuration management

**Model Output Structure**:
```
models/
├── model_current.pth     # Latest epoch (replaced each time)
├── model_best.pth        # Best accuracy model
└── model_final.pth       # Final training state
```
**README**: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)

---

### `training_integration.py`
**Purpose**: Background monitoring thread for automatic model replacement
**Key Features**:
```python
# Automatic background operation
# Monitors for new model files during training
# Handles replacement without interrupting training

# Integration points:
monitor = TrainingMonitorThread(output_dir)
monitor.start_monitoring()  # Automatic in pipeline
```
**Classes**:
- `TrainingMonitorThread`: Background model monitoring
- Model file detection and processing
- Automatic replacement orchestration

**Integration**: Seamlessly works with existing training code (no modifications needed)
**README**: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)

## 🌐 Data Management

### `s3_dataset_converter.py`
**Purpose**: Convert S3 ILSVRC dataset to SageMaker-compatible format
**Usage**:
```bash
# Complete conversion (val + test)
python s3_dataset_converter.py \
    --bucket "my-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"

# Conversion options
python s3_dataset_converter.py \
    --bucket "my-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker" \
    --convert-val \
    --convert-test \
    --skip-training-copy \
    --dry-run
```
**Features**:
- 📁 Reorganizes val/test data by class
- 🚀 Skips training data copy (performance optimization)
- 📊 Creates SageMaker manifest files
- 🔍 Dry-run mode for testing
- 📈 Progress tracking and validation

**Input**: ILSVRC dataset in S3
**Output**: SageMaker-ready dataset structure
**README**: [`S3_DATASET_CONVERTER_README.md`](S3_DATASET_CONVERTER_README.md)

---

### `sagemaker_wrapper.py`
**Purpose**: Training wrapper integrating 7-stage pipeline with model replacement
**Key Features**:
- 🔧 Wraps existing training code
- 🔄 Integrates model replacement system
- ⚙️ Supports all 7 pipeline stages
- 📊 Professional logging integration
- 🎯 Preserves original training logic

**Pipeline Stages**:
1. `lr_range_test` - Learning rate range testing
2. `lr_bounds` - LR boundary determination  
3. `onecycle_lr` - OneCycle learning rate testing
4. `batch_size_test` - Batch size optimization
5. `weight_decay_tuning` - Weight decay optimization
6. `full_training` - Complete model training
7. `monitoring` - Results analysis and monitoring

**Integration**: Automatically called by orchestrator and launcher
**README**: [`SIMPLIFIED_STRUCTURE.md`](SIMPLIFIED_STRUCTURE.md)

## 🔍 Monitoring & Logging

### `monitor_training.py`
**Purpose**: Real-time SageMaker training job monitoring with model tracking
**Usage**:
```bash
# Basic monitoring
python monitor_training.py --job-name "your-job-name"

# Advanced monitoring with model tracking
python monitor_training.py \
    --job-name "your-job-name" \
    --show-models \
    --metrics \
    --auto-restart
```
**Features**:
- 📊 Real-time training metrics
- 🔄 Model replacement status tracking
- 💰 Cost tracking (spot vs on-demand)
- 🚨 Error detection and alerting
- 🔄 Automatic restart capability

**Output**: Live training progress, model status, and performance metrics
**README**: [`LOGGING_INTEGRATION_SUMMARY.md`](LOGGING_INTEGRATION_SUMMARY.md)

---

### `sagemaker_logging.py`
**Purpose**: Professional logging infrastructure for the entire pipeline
**Features**:
```python
# Automatic setup - used by all components
# Professional log formatting
# Multiple log levels and outputs
# Structured logging for analysis

# Integration:
from sagemaker_logging import setup_logging
logger = setup_logging("component_name")
```
**Log Structure**:
- 📝 Detailed log files in `logs/` directory
- 🎯 Component-specific logging
- 📊 Structured format for analysis
- 🔄 Automatic log rotation

**Integration**: Used by all pipeline components automatically
**README**: [`LOGGING_INTEGRATION_SUMMARY.md`](LOGGING_INTEGRATION_SUMMARY.md)

## ⚙️ Configuration & Setup

### `setup_environment.py`
**Purpose**: Complete environment setup and AWS validation
**Usage**:
```bash
# Complete setup
python setup_environment.py

# Validation only
python setup_environment.py --validate

# Detailed diagnostics
python setup_environment.py --diagnose
```
**Features**:
- ✅ AWS credentials validation
- ✅ Python dependencies check
- ✅ SageMaker permissions verification
- ✅ S3 access testing
- ✅ Directory structure creation
- ✅ Configuration file generation

**Output**: Validated environment ready for training
**README**: [`scripts/README.md`](../scripts/README.md)

---

### Configuration Files

#### `configs/pipeline_config.json`
**Purpose**: Complete pipeline configuration
**Contents**:
```json
{
    "model_replacement": {
        "enable": true,
        "replace_previous_model": true,
        "save_best_model": true,
        "create_sagemaker_archive": true
    },
    "training": {
        "instance_type": "ml.p3.8xlarge",
        "use_spot_instances": true,
        "max_runtime_seconds": 86400
    },
    "pipeline": {
        "stages": ["lr_range_test", "full_training"],
        "enable_checkpointing": true
    }
}
```
**Usage**: Referenced automatically by orchestrator and launcher
**README**: [`configs/README.md`](../configs/README.md)

#### `configs/config_examples.json`
**Purpose**: Example configurations for different use cases
**Examples**:
- Development/testing configuration
- Production training configuration
- Cost-optimized configuration
- Research/experimentation configuration

## 🧪 Testing & Validation

### `test_model_replacement.py`
**Purpose**: Comprehensive testing of model replacement functionality
**Usage**:
```bash
# Complete test suite
python test_model_replacement.py

# Tests performed:
# ✅ Basic model saving functionality
# ✅ Epoch-based model replacement
# ✅ Best model tracking
# ✅ Background monitoring thread
# ✅ SageMaker integration
# ✅ File organization and cleanup
```
**Test Coverage**:
- Model saving with replacement logic
- Background monitoring detection
- Training integration without code changes
- Model file organization
- Archive creation for deployment

**Output**: Comprehensive test results and validation
**README**: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)

## 🛠️ Utility & Maintenance

### `cleanup_and_organize.py`
**Purpose**: Project cleanup and organization
**Usage**:
```bash
# Complete project cleanup
python cleanup_and_organize.py

# Cleanup tasks:
# 🧹 Remove temporary files
# 📁 Organize output directories
# 📊 Archive old logs
# 🔄 Reset for new training runs
```
**Features**:
- Temporary file cleanup
- Log organization and archiving
- Output directory structuring
- Configuration reset options

**README**: [`CLEANUP_SUMMARY.md`](CLEANUP_SUMMARY.md)

---

### Scripts Directory (`scripts/`)

#### `setup.bat` / `setup.sh`
**Purpose**: Cross-platform environment setup scripts
**Usage**:
```bash
# Windows
.\scripts\setup.bat

# Linux/Mac  
./scripts/setup.sh
```

#### `run_sagemaker.bat`
**Purpose**: Windows batch script for easy SageMaker launch
**Usage**:
```cmd
.\scripts\run_sagemaker.bat
```

## 📊 Dependencies & Requirements

### `requirements.txt`
**Purpose**: Main Python dependencies for SageMaker training
**Key Packages**:
- `sagemaker>=2.140.0` - AWS SageMaker SDK
- `boto3>=1.26.0` - AWS Python SDK
- `torch>=1.13.0` - PyTorch framework
- `torchvision>=0.14.0` - Computer vision utilities
- `pytest>=7.0.0` - Testing framework

### `converter_requirements.txt`
**Purpose**: Dependencies for S3 dataset conversion
**Key Packages**:
- `boto3>=1.26.0` - AWS S3 operations
- `tqdm>=4.64.0` - Progress bars
- `pillow>=9.0.0` - Image processing

## 📖 Documentation Files

### Core Documentation
- **`README.md`** - This complete guide (main entry point)
- **`MAIN_ENTRY_POINTS.md`** - Entry point selection guide
- **`USAGE_FLOW.md`** - Step-by-step workflow
- **`COMPLETE_FILE_REFERENCE.md`** - This detailed file guide

### Component Documentation  
- **`IMPLEMENTATION_COMPLETE.md`** - Model replacement system details
- **`S3_DATASET_CONVERTER_README.md`** - Dataset conversion guide
- **`LOGGING_INTEGRATION_SUMMARY.md`** - Logging system documentation
- **`SIMPLIFIED_STRUCTURE.md`** - Architecture and design overview

### Setup & Maintenance
- **`CLEANUP_SUMMARY.md`** - Cleanup procedures and organization
- **`scripts/README.md`** - Setup scripts documentation
- **`configs/README.md`** - Configuration guide

## 🎯 Quick Reference by Use Case

### 🚀 **"I want to start training immediately"**
```bash
python sagemaker_orchestrator.py
```
**Files involved**: `sagemaker_orchestrator.py`, `model_saver.py`, `training_integration.py`

### 🔧 **"I need to convert my dataset first"**
```bash
python s3_dataset_converter.py --bucket "my-bucket"
```
**Files involved**: `s3_dataset_converter.py`

### 📊 **"I want to monitor my training"**
```bash
python monitor_training.py --job-name "my-job"
```
**Files involved**: `monitor_training.py`, `sagemaker_logging.py`

### 🧪 **"I want to test the model replacement"**
```bash
python test_model_replacement.py
```
**Files involved**: `test_model_replacement.py`, `model_saver.py`, `training_integration.py`

### ⚙️ **"I need to setup my environment"**
```bash
python setup_environment.py
```
**Files involved**: `setup_environment.py`, configuration files

---

## 🏆 Summary

This file reference provides complete documentation for every component in the SageMaker training pipeline. The system is designed for **automatic model replacement every epoch** while maintaining **professional cloud training capabilities**.

**Key Innovation**: Models are saved and replaced automatically every epoch (no accumulation), ensuring efficient storage while preserving training history and best models.

**🎯 Start with**: `python sagemaker_orchestrator.py` for complete automated training with model replacement!