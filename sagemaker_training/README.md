# SageMaker Training Configuration for ImageNet ResNet50

## Quick Setup Guide

### 1. Prerequisites
- AWS Account with SageMaker permissions
- ImageNet dataset uploaded to S3
- SageMaker execution role configured

### 2. Installation
```bash
cd sagemaker_training
pip install -r requirements.txt
```

### 3. Upload ImageNet Dataset to S3

#### Option A: Upload entire ImageNet-1K dataset
```bash
# Create S3 bucket (replace with your bucket name)
aws s3 mb s3://your-imagenet-bucket

# Upload ImageNet dataset (maintaining train/val structure)
aws s3 sync /path/to/imagenet/ s3://your-imagenet-bucket/imagenet-1k/ --exclude "*.tar" --exclude "*.zip"

# Expected S3 structure:
# s3://your-imagenet-bucket/imagenet-1k/
# ├── train/
# │   ├── n01440764/  # class folders
# │   ├── n01443537/
# │   └── ...
# └── val/
#     ├── n01440764/
#     ├── n01443537/
#     └── ...
```

#### Option B: Upload separate train/validation to different S3 paths
```bash
aws s3 sync /path/to/imagenet/train/ s3://your-imagenet-bucket/imagenet-train/
aws s3 sync /path/to/imagenet/val/ s3://your-imagenet-bucket/imagenet-val/
```

### 4. Launch SageMaker Training

#### Basic Training with Spot Instances (Recommended)
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --epochs 90 \
  --batch-size 256
```

#### Advanced Training with Custom Configuration
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --val-data-s3 s3://your-imagenet-bucket/imagenet-val/ \
  --instance-type ml.g4dn.2xlarge \
  --instance-count 1 \
  --epochs 90 \
  --batch-size 512 \
  --lr-max 0.4 \
  --weight-decay 1e-4 \
  --checkpoint-s3 s3://your-imagenet-bucket/checkpoints/ \
  --output-s3 s3://your-imagenet-bucket/output/ \
  --job-name imagenet-resnet50-production
```

#### Quick Test Mode (For Development)
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --epochs 1 \
  --batch-size 64 \
  --quick-mode \
  --job-name imagenet-test
```

#### Multi-Instance Training (Distributed)
```bash
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-imagenet-bucket/imagenet-1k/ \
  --instance-type ml.g4dn.xlarge \
  --instance-count 4 \
  --epochs 90 \
  --batch-size 128 \
  --job-name imagenet-distributed
```

### 5. Instance Type Recommendations

| Instance Type | GPU | Memory | Cost (Spot) | Use Case |
|---------------|-----|--------|-------------|----------|
| ml.g4dn.xlarge | 1x T4 | 16GB | ~$0.20/hr | Development, small batch |
| ml.g4dn.2xlarge | 1x T4 | 32GB | ~$0.40/hr | Standard training |
| ml.g4dn.4xlarge | 1x T4 | 64GB | ~$0.80/hr | Large batch sizes |
| ml.p3.2xlarge | 1x V100 | 61GB | ~$0.90/hr | Faster training |
| ml.p3.8xlarge | 4x V100 | 244GB | ~$3.60/hr | Multi-GPU training |

### 6. Cost Optimization Tips

1. **Use Spot Instances**: 50-70% cost savings
2. **Enable Checkpointing**: Resume from interruptions
3. **Right-size instances**: Don't over-provision
4. **Use S3 Intelligent Tiering**: For dataset storage
5. **Monitor training**: Stop early if not converging

### 7. Monitoring Training

#### Check Job Status
```bash
aws sagemaker describe-training-job --training-job-name your-job-name
```

#### View Logs
```bash
aws logs filter-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix your-job-name
```

#### Download Results
```bash
# Download model artifacts
aws s3 sync s3://your-bucket/output/your-job-name/output/ ./results/

# Download training logs and metrics
aws s3 sync s3://your-bucket/output/your-job-name/output/data/ ./logs/
```

### 8. Integration with Your Existing Pipeline

The SageMaker training automatically uses your existing:
- ✅ `imagenet_training_pipeline.py` (7-step process)
- ✅ `imagenet_models.py` (ResNet50 implementation)
- ✅ `imagenet_dataset.py` (DataLoader logic)
- ✅ `logger_setup.py` (Logging system)

No modifications needed to your existing code!

### 9. Troubleshooting

#### Common Issues:
1. **Role Permission**: Ensure SageMaker role has S3 access
2. **Data Format**: Check S3 data structure matches ImageNet format
3. **Instance Limits**: Check AWS service quotas
4. **Spot Interruption**: Enable checkpointing for recovery

#### Debug Mode:
```bash
# Run with minimal resources for debugging
python launch_sagemaker_job.py \
  --train-data-s3 s3://your-bucket/small-dataset/ \
  --instance-type ml.m5.large \
  --epochs 1 \
  --batch-size 8 \
  --no-spot
```