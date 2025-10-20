# SageMaker Training for ImageNet with S3 Dataset Conversion

Streamlined SageMaker integration for ImageNet training with automated S3 dataset conversion.

## 🎯 Overview

This tool provides:
- **S3 Dataset Converter**: Convert existing S3 ILSVRC data to SageMaker format
- **SageMaker Training**: Launch distributed training with 7-step pipeline
- **Cost Optimization**: Spot instances with 70% savings
- **Professional Logging**: Comprehensive monitoring and tracking

## 📁 Core Files

| File | Purpose |
|------|---------|
| `s3_dataset_converter.py` | Convert S3 ILSVRC → SageMaker format |
| `launch_sagemaker.py` | Launch SageMaker training jobs |
| `sagemaker_wrapper.py` | Training wrapper for 7-step pipeline |
| `monitor_training.py` | Monitor training progress |

## 🚀 Quick Start

### Step 1: Convert Your S3 Dataset
```bash
# Install converter dependencies
pip install -r converter_requirements.txt

# Convert existing S3 ILSVRC data to SageMaker format
python s3_dataset_converter.py \
    --bucket "your-s3-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### Step 2: Launch SageMaker Training
```bash
# Install SageMaker dependencies
pip install -r requirements.txt

# Launch training with converted dataset
python launch_sagemaker.py \
    --job-name "imagenet-training-$(date +%Y%m%d-%H%M)" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge" \
    --spot-training \
    --epochs 90
```

### Step 3: Monitor Training
```bash
python monitor_training.py --job-name "your-job-name"
```

## 📊 Dataset Conversion Details

### Input Format (Your Existing ILSVRC)
```
s3://your-bucket/ILSVRC/
├── Data/CLS-LOC/train/     # 1000 class folders
├── Data/CLS-LOC/val/       # 50,000 flat validation images
└── ImageSets/CLS-LOC/val.txt # Validation labels
```

### Output Format (SageMaker-Ready)
```
s3://your-bucket/imagenet-sagemaker/
├── train/                  # 1000 class folders (copied)
├── val/                    # 1000 class folders (reorganized)
├── metadata/               # Dataset metadata
└── manifest.json           # SageMaker manifest
```

## 🔧 Usage Examples

### Basic Conversion
```bash
python s3_dataset_converter.py \
    --bucket "my-imagenet-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### Training with Custom Parameters
```bash
python launch_sagemaker.py \
    --job-name "resnet50-training" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --train-data-s3 "s3://my-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.16xlarge" \
    --spot-training \
    --epochs 90 \
    --batch-size 256 \
    --pipeline-stage "full_training"
```

### Development/Testing
```bash
python launch_sagemaker.py \
    --job-name "quick-test" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --train-data-s3 "s3://my-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.2xlarge" \
    --epochs 5 \
    --pipeline-stage "lr_range_test"
```

## 📋 Requirements

### AWS Setup
1. **AWS Account** with SageMaker access
2. **S3 Bucket** with your ILSVRC dataset
3. **IAM Role** for SageMaker with S3 permissions
4. **AWS CLI** configured with credentials

### Python Dependencies
```bash
# For dataset conversion
pip install -r converter_requirements.txt

# For SageMaker training
pip install -r requirements.txt
```

## 🔍 Verification

### Check Conversion Results
```bash
# Verify converted structure
aws s3 ls s3://your-bucket/imagenet-sagemaker/

# Check training classes
aws s3 ls s3://your-bucket/imagenet-sagemaker/train/ | head -10

# Check validation classes  
aws s3 ls s3://your-bucket/imagenet-sagemaker/val/ | head -10

# View manifest
aws s3 cp s3://your-bucket/imagenet-sagemaker/manifest.json - | jq
```

### Monitor Training Job
```bash
# Real-time monitoring
python monitor_training.py --job-name "your-job-name"

# AWS Console
https://console.aws.amazon.com/sagemaker/home#/jobs
```

## 🏷️ 7-Step Pipeline Stages

The training supports the complete 7-step ImageNet methodology:

1. **`lr_range_test`** - Find optimal learning rate range
2. **`lr_bounds`** - Determine LR boundaries  
3. **`onecycle_lr`** - Test OneCycle learning rate
4. **`batch_size_test`** - Find optimal batch size
5. **`weight_decay_tuning`** - Optimize weight decay
6. **`full_training`** - Complete model training
7. **`monitoring`** - Track and analyze results

Use `--pipeline-stage` to run specific stages or omit for full pipeline.

## 💰 Cost Optimization

### Spot Instances
- Use `--spot-training` for 70% cost savings
- Automatic checkpointing for fault tolerance
- Recommended for non-urgent training jobs

### Instance Selection
- **Development**: `ml.p3.2xlarge` (1 GPU)
- **Production**: `ml.p3.8xlarge` (4 GPUs)
- **Large Scale**: `ml.p3.16xlarge` (8 GPUs)

## 🆘 Troubleshooting

### Common Issues

1. **AWS Permissions**
   ```bash
   # Ensure your role has these S3 permissions:
   # s3:GetObject, s3:PutObject, s3:ListBucket
   ```

2. **Data Not Found**
   ```bash
   # Verify your ILSVRC data exists
   aws s3 ls s3://your-bucket/ILSVRC/Data/CLS-LOC/
   ```

3. **Training Fails**
   ```bash
   # Check CloudWatch logs in SageMaker console
   # Verify converted dataset structure
   ```

## 📖 Additional Documentation

- **`S3_DATASET_CONVERTER_README.md`** - Detailed converter documentation
- **`LOGGING_INTEGRATION_SUMMARY.md`** - Logging system details
- **`SIMPLIFIED_STRUCTURE.md`** - Architecture overview

## 🎉 Success Path

1. ✅ Convert your S3 ILSVRC dataset
2. ✅ Launch SageMaker training job  
3. ✅ Monitor progress and metrics
4. ✅ Scale with spot instances for cost optimization
5. ✅ Achieve production ImageNet training in the cloud!

Your ImageNet training pipeline is now cloud-ready with professional monitoring and cost optimization!