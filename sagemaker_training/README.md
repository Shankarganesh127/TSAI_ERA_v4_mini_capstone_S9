# SageMaker ImageNet Training

Self-contained AWS SageMaker training pipeline for ImageNet classification.

## Quick Start

### Basic Launch
```bash
python launch_sagemaker.py \
  --role-arn "arn:aws:iam::ACCOUNT:role/SageMakerRole" \
  --bucket "your-s3-bucket" \
  --epochs 30

python sagemaker_orchestrator.py   --role-arn "arn:aws:iam::872109682518:role/service-role/AmazonSageMaker-ExecutionRole-20251009T010774" --source-bucket "tsai-era-v4-mini-capstone" --use-spot --epochs 100 --instance-type "ml.g6.12xlarge" --instance-count 1

# For full pipeline with optimization
python sagemaker_orchestrator.py \
  --role-arn "arn:aws:iam::ACCOUNT:role/SageMakerRole" \
  --source-bucket "your-s3-bucket" \
  --epochs 30
```

## Core Files

- `sagemaker_wrapper.py` - Main entry point for SageMaker
- `imagenet_training_pipeline.py` - 7-step training pipeline
- `launch_sagemaker.py` - Job launcher
- `sagemaker_orchestrator.py` - Full pipeline orchestrator (optional)
- `requirements.txt` - Dependencies

## Dependencies

Key packages: `torch>=2.0.0`, `sagemaker>=2.175.0`, `boto3>=1.26.0`

## Features

- Self-contained training pipeline
- Optimized uploads (~160KB)
- Spot instance support
- Automatic hyperparameter optimization
- Real-time monitoring