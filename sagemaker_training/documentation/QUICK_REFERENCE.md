# Quick Reference: S3 Dataset Converter

## Essential Commands

### 1. Setup (One-time)
```bash
# Windows
setup.bat

# Linux/Mac  
chmod +x setup.sh && ./setup.sh

# Manual
pip install -r converter_requirements.txt
pip install -r requirements.txt
aws configure
```

### 2. Convert Dataset
```bash
python s3_dataset_converter.py \
    --bucket "your-s3-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### 3. Launch Training
```bash
python launch_sagemaker.py \
    --job-name "imagenet-training" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge" \
    --spot-training
```

### 4. Monitor
```bash
python monitor_training.py --job-name "imagenet-training"
```

## Key Files
- `s3_dataset_converter.py` - S3 ILSVRC → SageMaker converter
- `launch_sagemaker.py` - SageMaker job launcher  
- `S3_DATASET_CONVERTER_README.md` - Detailed documentation
- `README.md` - Complete guide

## Result
✅ S3 ILSVRC dataset → SageMaker-ready format → Distributed training