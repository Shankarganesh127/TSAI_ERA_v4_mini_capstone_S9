# S3 Storage Handles in SageMaker for ImageNet Dataset

This guide explains how to properly configure S3 data inputs for your SageMaker ImageNet training jobs.

## 1. S3 Data Path Configuration

### Basic S3 Path Structure
Your converted ILSVRC dataset should be structured like this in S3:

```
s3://your-bucket/imagenet-sagemaker/
├── train/                  # Training data
│   ├── n01440764/         # Class folders
│   ├── n01443537/
│   └── ... (1000 classes)
├── val/                    # Validation data  
│   ├── n01440764/         # Class folders
│   ├── n01443537/
│   └── ... (1000 classes)
├── metadata/              # Dataset metadata
└── manifest.json          # SageMaker manifest
```

## 2. SageMaker Data Input Methods

### Method 1: Simple S3 Path (Recommended for most cases)
```bash
python launch_sagemaker.py \
    --job-name "imagenet-training" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --s3-bucket "s3://your-bucket" \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge"
```

### Method 2: Using Data Prefix (Default structure)
```bash
python launch_sagemaker.py \
    --job-name "imagenet-training" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --s3-bucket "s3://your-bucket" \
    --data-prefix "imagenet-sagemaker" \
    --instance-type "ml.p3.8xlarge"
```

### Method 3: Advanced TrainingInput Configuration
The enhanced `launch_sagemaker.py` automatically configures optimal S3 data inputs:

```python
# In launch_sagemaker.py - automatic configuration
train_input = TrainingInput(
    s3_data=data_s3_path,
    distribution='FullyReplicated',  # Replicate to all instances
    s3_data_type='S3Prefix',        # Directory prefix
    input_mode='FastFile',          # High-performance file access
    compression=None                # No compression for images
)
```

## 3. Data Distribution Modes

### FastFile Mode (Default - Recommended)
- **Best for**: Large datasets like ImageNet
- **Benefits**: Fastest data access, intelligent caching
- **Usage**: `--distribution-mode FastFile`

### File Mode
- **Best for**: Smaller datasets or debugging
- **Benefits**: Full S3 download to local storage
- **Usage**: `--distribution-mode File`

### Pipe Mode
- **Best for**: Streaming large datasets
- **Benefits**: Reduced storage requirements
- **Usage**: `--distribution-mode Pipe`

## 4. S3 Data Paths in SageMaker Training Container

### Inside Training Container
When your training job runs, SageMaker automatically maps your S3 data:

```
S3 Path: s3://bucket/imagenet-sagemaker/
Container Path: /opt/ml/input/data/imagenet/

Structure inside container:
/opt/ml/input/data/imagenet/
├── train/
│   ├── n01440764/
│   └── ... (class folders)
├── val/  
│   ├── n01440764/
│   └── ... (class folders)
└── metadata/
```

### Data Path Configuration in Code
The `sagemaker_wrapper.py` automatically handles this:

```python
# Hyperparameter passed to training
--data_dir /opt/ml/input/data/imagenet

# Your training code sees:
data_dir = "/opt/ml/input/data/imagenet"
train_path = os.path.join(data_dir, "train")      # /opt/ml/input/data/imagenet/train
val_path = os.path.join(data_dir, "val")          # /opt/ml/input/data/imagenet/val
```

## 5. Complete Example Workflow

### Step 1: Convert Your Existing S3 ILSVRC Data
```bash
python upload_imagenet_to_s3.py convert \
    --bucket "your-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### Step 2: Launch Training with Proper S3 Configuration
```bash
python launch_sagemaker.py \
    --job-name "imagenet-resnet50-$(date +%Y%m%d-%H%M)" \
    --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole" \
    --s3-bucket "s3://your-bucket" \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge" \
    --distribution-mode "FastFile" \
    --spot-training \
    --epochs 90
```

### Step 3: Monitor Training
```bash
python monitor_training.py --job-name "your-job-name"
```

## 6. S3 Data Input Options Explained

### Required Parameters:
- `--s3-bucket`: Your S3 bucket (e.g., "s3://my-bucket" or "my-bucket")
- `--role-arn`: SageMaker execution role with S3 access

### Data Path Options (choose one):
- `--train-data-s3`: Direct S3 path to your dataset
- `--data-prefix`: Prefix within the specified bucket

### Performance Options:
- `--distribution-mode`: FastFile (default), File, or Pipe
- `--spot-training`: Use spot instances for cost savings

## 7. Common S3 Data Issues and Solutions

### Issue 1: "S3 Access Denied"
**Solution**: Ensure your SageMaker role has these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket",
                "arn:aws:s3:::your-bucket/*"
            ]
        }
    ]
}
```

### Issue 2: "Data not found in container"
**Solution**: Check S3 path format and ensure data exists:
```bash
# Verify your data exists
aws s3 ls s3://your-bucket/imagenet-sagemaker/ --recursive | head -20

# Check converted structure
aws s3 ls s3://your-bucket/imagenet-sagemaker/train/ | head -10
aws s3 ls s3://your-bucket/imagenet-sagemaker/val/ | head -10
```

### Issue 3: "Slow data loading"
**Solution**: Use FastFile mode and ensure proper class folder structure:
```bash
# Use FastFile for best performance
--distribution-mode FastFile

# Verify class folder structure (required for fast loading)
aws s3 ls s3://your-bucket/imagenet-sagemaker/train/n01440764/ | head -5
```

## 8. Advanced S3 Configuration

### Multi-Instance Training Data Distribution
For multi-GPU training, SageMaker automatically:
- Replicates data to all instances (FullyReplicated)
- Coordinates data loading across GPUs
- Handles data sharding for distributed training

### Custom S3 Data Input (if needed)
If you need custom data input configuration, modify `launch_sagemaker.py`:

```python
# Custom TrainingInput for special cases
from sagemaker.inputs import TrainingInput

custom_input = TrainingInput(
    s3_data="s3://your-bucket/custom-path/",
    distribution='ShardedByS3Key',  # For custom sharding
    s3_data_type='S3Prefix',
    input_mode='File',
    compression='Gzip'  # If your data is compressed
)

# Use in estimator.fit()
estimator.fit(inputs={'imagenet': custom_input})
```

## 9. Best Practices

### Performance Optimization:
1. **Use FastFile mode** for large datasets
2. **Organize by class folders** for optimal loading
3. **Use spot instances** for cost savings
4. **Choose appropriate instance types** based on dataset size

### Cost Optimization:
1. **Spot instances**: 70% cost savings with fault tolerance
2. **Right-size storage**: Match EBS volume to dataset size
3. **Monitor usage**: Use CloudWatch for cost tracking

### Data Organization:
1. **Convert ILSVRC structure** using the conversion tool
2. **Validate structure** before launching training
3. **Use manifest files** for dataset metadata
4. **Test with small jobs** before full training

Your S3 data is now properly configured for SageMaker training with optimal performance and cost efficiency!