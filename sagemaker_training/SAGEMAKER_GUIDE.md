# 🚀 SageMaker Training Setup for ImageNet ResNet50

This directory contains everything you need to run your existing ImageNet ResNet50 training pipeline on Amazon SageMaker with spot instances - **without modifying any of your original code**.

## 📋 Complete Setup Guide

### 1. Prerequisites Setup

```bash
# Install required packages
pip install -r requirements.txt

# Configure AWS credentials (one-time setup)
aws configure
# Enter your AWS Access Key ID, Secret, Region (e.g., us-east-1), and output format (json)
```

### 2. Upload ImageNet Dataset to S3

#### Option A: Automated Upload (Recommended)
```bash
# Upload your local ImageNet dataset to S3
python upload_imagenet_to_s3.py \
  --local-data "/path/to/your/imagenet" \
  --bucket "your-imagenet-bucket" \
  --s3-prefix "imagenet-1k" \
  --create-bucket

# Example for Windows
python upload_imagenet_to_s3.py ^
  --local-data "D:\datasets\imagenet" ^
  --bucket "mycompany-imagenet-data" ^
  --s3-prefix "imagenet-1k" ^
  --create-bucket
```

#### Option B: Manual AWS CLI Upload
```bash
# Create bucket
aws s3 mb s3://your-imagenet-bucket

# Upload dataset (maintaining train/val structure)
aws s3 sync /path/to/imagenet/ s3://your-imagenet-bucket/imagenet-1k/
```

### 3. Launch SageMaker Training

#### 🎯 Quick Start - Default Training
```bash
# Standard ImageNet training with spot instances (recommended)
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --epochs 90
```

#### 🏃 Quick Test Mode (Development)
```bash
# Fast test with 1 epoch for development
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --epochs 1 \
  --batch-size 64 \
  --quick-mode \
  --job-name imagenet-test
```

#### 🚀 Production Training (High Performance)
```bash
# High-performance training with larger instance
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.p3.2xlarge \
  --epochs 90 \
  --batch-size 512 \
  --lr-max 0.6 \
  --job-name imagenet-production
```

#### 🌐 Distributed Training (Multi-Instance)
```bash
# Distributed training across 4 instances
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --instance-count 4 \
  --epochs 90 \
  --batch-size 128 \
  --job-name imagenet-distributed
```

### 4. Monitor Training Progress

#### Check Job Status
```bash
# List all recent training jobs
python monitor_training.py --list

# Watch jobs in real-time (refreshes every 30 seconds)
python monitor_training.py --list --watch

# Filter by status
python monitor_training.py --list --status InProgress
```

#### View Job Details and Logs
```bash
# Get detailed information about a specific job
python monitor_training.py --details your-job-name

# View recent training logs
python monitor_training.py --logs your-job-name --log-lines 100

# Get training metrics
python monitor_training.py --metrics your-job-name
```

#### Stop Training Job
```bash
# Stop a running job
python monitor_training.py --stop your-job-name
```

### 5. Windows Users - Easy GUI Interface

```cmd
# Double-click or run from command prompt
run_sagemaker.bat
```

This provides an interactive menu for:
- Launching training jobs
- Uploading datasets
- Quick test training

## 💰 Cost Optimization

### Spot Instance Savings (50-70% cost reduction)

All training uses spot instances by default. Your training will:
- ✅ Automatically save checkpoints every 5 epochs
- ✅ Resume from interruptions seamlessly  
- ✅ Use your existing 7-step pipeline without changes

### Instance Type Cost Comparison

| Instance | GPU | Memory | Spot Cost/hr | Use Case |
|----------|-----|--------|--------------|----------|
| ml.g4dn.xlarge | T4 16GB | 16GB | ~$0.20 | Development/Testing |
| ml.g4dn.2xlarge | T4 16GB | 32GB | ~$0.40 | Standard Training |
| ml.p3.2xlarge | V100 16GB | 61GB | ~$0.90 | High Performance |
| ml.p3.8xlarge | 4x V100 | 244GB | ~$3.60 | Multi-GPU Training |

### Cost Estimation Example
```
Standard Training (90 epochs, ml.g4dn.2xlarge):
- On-demand: ~$1.20/hr × 8 hours = $9.60
- Spot instance: ~$0.40/hr × 8 hours = $3.20
- Savings: 67% ($6.40 saved)
```

## 🔧 Advanced Configuration

### Custom Hyperparameters
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://bucket/data/ \
  --epochs 90 \
  --batch-size 256 \
  --lr-max 0.4 \
  --weight-decay 1e-4 \
  --instance-type ml.g4dn.2xlarge
```

### Separate Train/Validation Data
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://bucket/imagenet-train/ \
  --val-data-s3 s3://bucket/imagenet-val/ \
  --epochs 90
```

### Custom Output and Checkpoint Paths
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://bucket/data/ \
  --output-s3 s3://bucket/training-results/ \
  --checkpoint-s3 s3://bucket/checkpoints/ \
  --epochs 90
```

## 📊 What Happens During Training

Your SageMaker training automatically uses your existing:

1. **`imagenet_training_pipeline.py`** - Complete 7-step training process
   - LR Range Test → Pick LR bounds → OneCycle LR → Batch size → Weight decay tuning → Full training → Monitoring

2. **`imagenet_models.py`** - Your ResNet50 implementation

3. **`imagenet_dataset.py`** - Your DataLoader and preprocessing logic

4. **`logger_setup.py`** - Your logging system

**No code changes needed!** The SageMaker wrapper (`sagemaker_train.py`) acts as a bridge that:
- ✅ Handles SageMaker environment setup
- ✅ Maps S3 data paths to your existing code
- ✅ Saves results in SageMaker format
- ✅ Preserves all your hyperparameter optimization logic

## 📥 Retrieving Results

### Download Trained Model
```bash
# Models are automatically saved to S3
aws s3 sync s3://bucket/output/your-job-name/model/ ./trained_model/
```

### Download Training Logs and Metrics  
```bash
# Training history, loss curves, metrics
aws s3 sync s3://bucket/output/your-job-name/output/data/ ./training_logs/
```

### View in AWS Console
- Go to AWS SageMaker Console → Training Jobs
- Click on your job name to see metrics, logs, and model artifacts

## 🛠️ Troubleshooting

### Common Issues

1. **"Access Denied" Error**
   ```bash
   # Check IAM role permissions
   aws sts get-caller-identity
   # Ensure SageMaker execution role has S3 access
   ```

2. **"Dataset Not Found"**
   ```bash
   # Verify S3 structure
   aws s3 ls s3://your-bucket/imagenet-1k/train/ --recursive
   # Should show class folders: n01440764/, n01443537/, etc.
   ```

3. **Training Fails Quickly**
   ```bash
   # Check logs for detailed error
   python monitor_training.py --logs your-job-name --log-lines 200
   ```

### Debug Mode
```bash
# Run minimal training for debugging
python launch_sagemaker_job.py \
  --train-data-s3 s3://bucket/small-subset/ \
  --instance-type ml.g4dn.xlarge \
  --epochs 1 \
  --batch-size 8 \
  --no-spot \
  --job-name debug-test
```

## 🎯 Next Steps

1. **Start with quick test** to verify everything works
2. **Upload your ImageNet dataset** to S3 
3. **Launch standard training** with spot instances
4. **Monitor progress** and download results
5. **Scale up** to distributed training if needed

## 📞 Support

- Check `config_examples.json` for configuration templates
- View AWS SageMaker console for job details
- Use `monitor_training.py` for real-time status
- All your existing training features work unchanged!

---

### Summary: Zero Code Changes Required! 

Your existing ResNet50 ImageNet pipeline runs on SageMaker with:
- ✅ 50-70% cost savings with spot instances
- ✅ Automatic scaling and checkpointing  
- ✅ Complete 7-step training process preserved
- ✅ Professional MLOps integration
- ✅ Easy monitoring and management tools