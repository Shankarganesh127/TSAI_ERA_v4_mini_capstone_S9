# SageMaker ImageNet Training Pipeline

A complete, self-contained AWS SageMaker training pipeline for ImageNet classification using a systematic 7-step approach.

## 🚀 Quick Start

### Option 1: Simple Job Runner (Recommended)
```bash
cd sagemaker_training

# Basic training job (1 epoch, spot instances)
python run_sagemaker_job.py \
  --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX" \
  --bucket "your-s3-bucket" \
  --epochs 1 \
  --use-spot

# Full training job (30 epochs, regular instances)
python run_sagemaker_job.py \
  --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX" \
  --bucket "your-s3-bucket" \
  --epochs 30 \
  --instance-type ml.p3.8xlarge
```

### Option 2: Full Orchestrator (Advanced)
```bash
# Complete pipeline with dataset validation and conversion
python sagemaker_orchestrator.py \
  --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX" \
  --source-bucket "your-s3-bucket" \
  --epochs 30 \
  --instance-type ml.p3.8xlarge \
  --use-spot
```

## 📁 Directory Structure

### Core Training Components
| File | Size | Purpose |
|------|------|---------|
| `sagemaker_wrapper.py` | 22KB | **SageMaker Entry Point** - Main wrapper for SageMaker execution |
| `imagenet_training_pipeline.py` | 32KB | **7-Step Training Pipeline** - Complete systematic training approach |
| `imagenet_models.py` | 7KB | **Model Definitions** - ResNet, EfficientNet, Vision Transformer |
| `imagenet_dataset.py` | 5KB | **Dataset Classes** - ImageNet data loading and preprocessing |
| `ilsvrc_dataset.py` | 11KB | **ILSVRC Dataset** - Specialized ILSVRC format handling |
| `train_imagenet.py` | 8KB | **Training Logic** - Core training loop implementation |

### SageMaker Integration
| File | Size | Purpose |
|------|------|---------|
| `sagemaker_orchestrator.py` | 37KB | **Complete Orchestrator** - Full pipeline management |
| `launch_sagemaker.py` | 15KB | **Job Launcher** - Simple job submission |
| `run_sagemaker_job.py` | 2KB | **Quick Runner** - Easy command-line interface |
| `s3_dataset_converter.py` | 34KB | **Dataset Converter** - S3 data preparation |
| `monitor_training.py` | 24KB | **Job Monitor** - Real-time training monitoring |

### Utilities & Support
| File | Size | Purpose |
|------|------|---------|
| `training_integration.py` | 12KB | **Training Integration** - Model saving & monitoring |
| `model_saver.py` | 13KB | **Model Management** - Epoch-based model saving |
| `logger_setup.py` | 7KB | **Logging System** - Structured logging configuration |
| `sagemaker_logging.py` | 6KB | **SageMaker Logging** - Cloud-specific logging |
| `utils.py` | 4KB | **Helper Functions** - Common utilities |

### Configuration
| Directory/File | Contents |
|----------------|----------|
| `configs/` | Configuration files for pipeline settings |
| `├─ config_examples.json` | Example configurations |
| `├─ pipeline_config.json` | Pipeline-specific settings |
| `scripts/` | Support scripts |
| `├─ setup_environment.py` | Environment setup |
| `requirements.txt` | Python dependencies |
| `.sagemakerignore` | Files to exclude from upload |

### Runtime Directories
| Directory | Purpose |
|-----------|---------|
| `logs/` | Training logs (created during execution) |
| `outputs/` | Model outputs and artifacts |

## 🎯 7-Step Training Pipeline

The `imagenet_training_pipeline.py` implements a systematic approach:

1. **LR Range Test** - Find optimal learning rate bounds
2. **Learning Rate Selection** - Pick min/max LR from range test
3. **OneCycle Setup** - Configure cyclical learning rate + momentum
4. **Batch Size Optimization** - Find optimal batch size for memory/speed
5. **Weight Decay Tuning** - Optimize regularization parameters  
6. **Full Training** - Complete OneCycle training with all optimizations
7. **Monitoring & Iteration** - Track metrics and iterate as needed

## 📦 Dependencies

All required packages are specified in `requirements.txt`:
- **AWS & SageMaker**: `boto3>=1.26.0`, `sagemaker>=2.175.0`
- **Deep Learning**: `torch>=2.0.0`, `torchvision>=0.15.0`
- **Data Science**: `numpy>=1.21.0`, `pandas>=1.5.0`, `matplotlib>=3.5.0`
- **Utilities**: `tqdm>=4.64.0`, `pillow>=9.0.0`, `scikit-learn>=1.1.0`

## 🔧 Configuration Options

### Instance Types
- **Development**: `ml.p3.2xlarge` (1 GPU, 8 vCPU, 61GB RAM)
- **Production**: `ml.p3.8xlarge` (4 GPU, 32 vCPU, 244GB RAM)  
- **Large Scale**: `ml.p3.16xlarge` (8 GPU, 64 vCPU, 488GB RAM)

### Training Modes
- **Quick Mode**: Fast validation with reduced epochs
- **Full Mode**: Complete training with all optimization steps
- **Spot Instances**: Up to 90% cost savings with managed interruption

## 💡 Key Features

- **Self-Contained**: No external dependencies - all files included
- **Path Resolution Fixed**: No parent directory imports needed
- **Optimized Upload**: Only essential files uploaded to SageMaker
- **Real-time Monitoring**: Track training progress and metrics
- **Model Versioning**: Automatic epoch-based model saving
- **Spot Instance Support**: Cost-effective training with interruption handling
- **Multi-channel Data**: Separate train/validation data paths
- **Comprehensive Logging**: Detailed logs for debugging and analysis

## 🚨 Prerequisites

1. **AWS SageMaker Role**: Execution role with S3 and SageMaker permissions
2. **S3 Bucket**: For dataset storage and model artifacts  
3. **ImageNet Dataset**: Properly formatted and uploaded to S3
4. **AWS CLI**: Configured with appropriate credentials

## 📊 Expected Training Times

| Instance Type | Batch Size | Time/Epoch | Cost/Hour |
|---------------|------------|------------|-----------|
| ml.p3.2xlarge | 64-128 | ~45 min | $3.06 |
| ml.p3.8xlarge | 256-512 | ~15 min | $12.24 |
| ml.p3.16xlarge | 512-1024 | ~8 min | $24.48 |

*Note: Times vary based on model complexity and data preprocessing*

## 🔍 Monitoring & Debugging

- **CloudWatch Logs**: Real-time training logs
- **Model Artifacts**: Saved to S3 output location
- **Tensorboard**: Training metrics visualization  
- **Progress Tracking**: Step-by-step pipeline progress

## 📝 Usage Notes

- Directory is completely self-contained and ready for SageMaker deployment
- All paths resolved relative to this directory - no external dependencies
- Supports both single-channel and multi-channel data input
- Automatic batch size detection based on available GPU memory
- Built-in retry logic for spot instance interruptions

## 🛠️ File Sizes Summary

**Total Directory Size**: ~280KB (excluding cache files)
- **Core Training**: ~88KB (6 files)
- **SageMaker Integration**: ~104KB (5 files)  
- **Utilities**: ~47KB (5 files)
- **Configuration**: ~15KB (configs + requirements)
- **Support Scripts**: ~12KB (scripts + runner)

**Upload Efficiency**: Optimized for fast SageMaker deployment with `.sagemakerignore` filtering.