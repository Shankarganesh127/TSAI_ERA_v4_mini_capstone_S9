# How to Set Separate S3 Paths for Train and Validation Data

## Your Dataset Paths
- **Training data**: `s3://<bucket>/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/`
- **Validation data**: `s3://<bucket>/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/`

## Method 1: Command Line (Recommended)

### Basic Command
```bash
python sagemaker_training/launch_sagemaker.py \
    --job-name "imagenet-train-$(date +%Y%m%d-%H%M%S)" \
    --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole" \
    --s3-bucket "s3://YOUR_BUCKET_NAME" \
    --train-data-s3 "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/" \
    --val-data-s3 "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30 \
    --spot-training
```

### Example with Actual Values
```bash
# Replace with your actual bucket name
python sagemaker_training/launch_sagemaker.py \
    --job-name "imagenet-multi-channel-20251021" \
    --role-arn "arn:aws:iam::123456789012:role/SageMakerExecutionRole" \
    --s3-bucket "s3://my-imagenet-bucket" \
    --train-data-s3 "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/" \
    --val-data-s3 "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30 \
    --spot-training
```

## Method 2: Set Defaults in Code

### Edit `launch_sagemaker.py` (around line 46):
```python
# Default S3 paths configuration - modify these for your datasets
DEFAULT_CONFIG = {
    'train_data_path': "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/",
    'val_data_path': "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/",
    'bucket': "s3://YOUR_BUCKET_NAME"
}
```

Then use defaults in argument parsing:
```python
parser.add_argument('--train-data-s3', type=str, 
                   default=DEFAULT_CONFIG['train_data_path'],
                   help='S3 path to training data')
parser.add_argument('--val-data-s3', type=str, 
                   default=DEFAULT_CONFIG['val_data_path'],
                   help='S3 path to validation data')
parser.add_argument('--s3-bucket', 
                   default=DEFAULT_CONFIG['bucket'],
                   help='S3 bucket (s3://bucket-name)')
```

## What Happens When You Run This

### SageMaker Environment Variables Created:
```bash
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet        # Your training data
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation    # Your validation data
```

### Container Directory Structure:
```
/opt/ml/input/data/imagenet/          # Training data from s3://bucket/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/
├── n01440764/
├── n01443537/
└── ...

/opt/ml/input/data/validation/        # Validation data from s3://bucket/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/
├── n01440764/
├── n01443537/
└── ...
```

### Your Training Script Receives:
```python
# In sagemaker_wrapper.py
args.data_dir = "/opt/ml/input/data/imagenet"      # Training data
args.val_dir = "/opt/ml/input/data/validation"     # Validation data

# Passed to imagenet_training_pipeline.py as:
--data /opt/ml/input/data/imagenet --val-data /opt/ml/input/data/validation
```

## Benefits of Multi-Channel Setup

✅ **Separate S3 Paths**: Use different S3 locations for train/val
✅ **Flexible Sources**: Can use different buckets if needed
✅ **Efficient Downloads**: Only downloads what you need
✅ **Clear Organization**: Train and val data are clearly separated

## Configuration Examples

### Example 1: Different Buckets
```bash
--train-data-s3 "s3://training-bucket/imagenet-train/"
--val-data-s3 "s3://validation-bucket/imagenet-val/"
```

### Example 2: Same Bucket, Different Paths
```bash
--s3-bucket "s3://my-bucket"
--train-data-s3 "datasets/imagenet/train/"
--val-data-s3 "datasets/imagenet/val/"
```

### Example 3: Your Specific Paths
```bash
--s3-bucket "s3://YOUR_BUCKET"
--train-data-s3 "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/"
--val-data-s3 "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/"
```

## Quick Test Command

Replace `YOUR_BUCKET` with your actual bucket name:
```bash
python sagemaker_training/launch_sagemaker.py \
    --job-name "test-multi-channel" \
    --role-arn "YOUR_ROLE_ARN" \
    --s3-bucket "s3://YOUR_BUCKET" \
    --train-data-s3 "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/" \
    --val-data-s3 "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/" \
    --instance-type "ml.p3.2xlarge" \
    --epochs 1 \
    --quick-mode
```

This will create two separate data channels and your training script will receive both paths!