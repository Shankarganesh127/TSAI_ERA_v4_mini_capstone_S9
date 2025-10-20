# SageMaker Training for 7-Step ImageNet Pipeline

## Overview

Simplified SageMaker integration for your sophisticated 7-step ImageNet training pipeline:

1. **LR Range Test** → Find optimal learning rate bounds
2. **Pick LR bounds** → Extract min/max LR from range test
3. **OneCycle LR** → Configure advanced scheduler  
4. **Choose batch size** → Auto-detect optimal GPU memory usage
5. **Tune weight-decay** → Grid search with validation
6. **Full training** → Complete OneCycle training
7. **Monitor** → Comprehensive analysis and logging

## Files

- **`sagemaker_wrapper.py`** - Single training wrapper preserving 7-step methodology
- **`launch_sagemaker.py`** - Simple job launcher with hyperparameter control
- **`monitor_training.py`** - Job monitoring and progress tracking
- **`upload_imagenet_to_s3.py`** - Data upload utility

## Quick Start

### 1. Upload Data
```bash
python upload_imagenet_to_s3.py --s3-bucket s3://your-bucket
```

### 2. Launch Training

**Full Automated Pipeline:**
```bash
python launch_sagemaker.py \
  --job-name imagenet-auto \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --s3-bucket s3://your-bucket
```

**Quick Development:**
```bash
python launch_sagemaker.py \
  --job-name quick-test \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --s3-bucket s3://your-bucket \
  --quick-mode \
  --epochs 5
```

**Custom Hyperparameters:**
```bash
python launch_sagemaker.py \
  --job-name custom-hp \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --s3-bucket s3://your-bucket \
  --batch-size 64 \
  --skip-lr-finder \
  --weight-decay 1e-3
```

### 3. Monitor Progress
```bash
python monitor_training.py --job-name your-job-name
```

## Pipeline Control

| Flag | Description | Impact |
|------|-------------|---------|
| `--skip-lr-finder` | Skip LR Range Test (Step 1) | Uses default or manual LR bounds |
| `--skip-wd-search` | Skip Weight Decay Search (Step 5) | Uses default or manual weight decay |
| `--quick-mode` | Fast development iterations | Reduced epochs |
| `--batch-size <N>` | Override auto-detection (Step 4) | Manual batch size |
| `--lr-min <F>` | Manual minimum LR (Step 2) | Override LR bounds |
| `--lr-max <F>` | Manual maximum LR (Step 2) | Override LR bounds |
| `--weight-decay <F>` | Manual weight decay (Step 5) | Override search result |

## Cost Optimization

**Spot Instances (70% savings):**
```bash
python launch_sagemaker.py \
  --job-name cost-optimized \
  --role-arn <role> \
  --s3-bucket <bucket> \
  --spot-training \
  --instance-type ml.p3.2xlarge
```

## Key Features

- ✅ **Complete 7-step pipeline preservation** - No changes to original methodology
- ✅ **Flexible hyperparameter override** - Control each step individually  
- ✅ **Spot instance support** - Up to 70% cost savings
- ✅ **Professional logging** - CloudWatch integration
- ✅ **Simple interface** - Single wrapper, single launcher
- ✅ **No code changes** - Original `imagenet_training_pipeline.py` unchanged

## Architecture

```
SageMaker Job Launch
        ↓
   sagemaker_wrapper.py
        ↓
   imagenet_training_pipeline.py (Original 7-step methodology)
        ↓
   Complete training with all optimizations
```

The wrapper preserves your sophisticated 7-step approach while adding professional cloud deployment capabilities.