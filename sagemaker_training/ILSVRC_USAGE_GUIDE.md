# ILSVRC ImageNet Dataset Usage Guide for SageMaker

This guide explains how to use your ILSVRC ImageNet dataset structure with SageMaker training, including **conversion from existing S3 ILSVRC data**.

## Your Dataset Structure

Based on the structure you provided:

```
/your/path/ILSVRC/
├── Data/
│   └── CLS-LOC/
│       ├── train/          # 1000 class folders (n01440764/, n01443537/, etc.)
│       ├── val/            # 50,000 validation images (flat structure)
│       └── test/           # Test images (optional)
├── Annotations/
│   └── CLS-LOC/
│       ├── train/          # XML annotation files
│       └── val/            # Validation annotations
└── ImageSets/
    └── CLS-LOC/
        ├── train_cls.txt   # Training image list
        ├── val.txt         # Validation image list
        └── test.txt        # Test image list (optional)
```

## Option 1: Upload Local ILSVRC Dataset to S3

If you have the dataset locally and want to upload it:

```bash
python upload_imagenet_to_s3.py upload \
    --ilsvrc-path "/your/path/ILSVRC" \
    --bucket "your-s3-bucket-name" \
    --s3-prefix "imagenet-data" \
    --create-bucket \
    --max-workers 8
```

## Option 2: Convert Existing S3 ILSVRC to SageMaker Format ⭐

**This is what you need!** Since you already have ILSVRC data in S3, use the conversion mode:

```bash
python upload_imagenet_to_s3.py convert \
    --bucket "your-s3-bucket-name" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker" \
    --aws-profile "your-aws-profile"
```

### Conversion Parameters:
- `--bucket`: Your S3 bucket containing the ILSVRC dataset
- `--source-prefix`: S3 prefix of your existing ILSVRC data (e.g., "ILSVRC")
- `--target-prefix`: New prefix for SageMaker-compatible structure
- `--aws-profile`: AWS profile to use (optional)

### What the Conversion Does:

1. **Training Data**: Copies `ILSVRC/Data/CLS-LOC/train/` → `imagenet-sagemaker/train/`
   - Preserves class folder structure (already SageMaker-compatible)

2. **Validation Data**: Reorganizes `ILSVRC/Data/CLS-LOC/val/` → `imagenet-sagemaker/val/`
   - Converts flat structure to class folders using `ImageSets/CLS-LOC/val.txt`
   - Creates: `val/n01440764/`, `val/n01443537/`, etc.

3. **Metadata**: Creates SageMaker-compatible metadata
   - Class mappings and statistics
   - Dataset manifest for training
   - S3 path configurations

## Step 2: Verify Converted Structure

After conversion, your S3 structure will be SageMaker-ready:

```
s3://your-bucket/imagenet-sagemaker/
├── train/                  # SageMaker-compatible training data
│   ├── n01440764/         # Class folders with images (copied from source)
│   ├── n01443537/
│   └── ... (1000 classes)
├── val/                    # Validation data (reorganized)
│   ├── n01440764/         # Images organized by class (converted from flat)
│   ├── n01443537/
│   └── ... (1000 classes)
├── metadata/              # Generated SageMaker metadata
│   └── dataset_metadata.json
└── manifest.json          # SageMaker dataset manifest
```

### Conversion Benefits:
- ✅ **No Data Downloads**: Pure S3-to-S3 operations (fast and cost-effective)
- ✅ **Validation Reorganization**: Flat validation images sorted into class folders
- ✅ **SageMaker Compatibility**: Perfect structure for distributed training
- ✅ **Metadata Generation**: Automatic class mappings and statistics
- ✅ **Preserve Original**: Source ILSVRC data remains untouched

## Step 3: Launch SageMaker Training

Use your converted dataset for training:

```bash
python launch_sagemaker.py \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.2xlarge" \
    --use-spot \
    --pipeline-stage "full_training" \
    --epochs 90
```

## Example: Complete Workflow for Existing S3 Data

Since you already have ILSVRC data in S3, here's your complete workflow:

### 1. Convert Your Existing S3 ILSVRC Data
```bash
# Convert your existing S3 ILSVRC to SageMaker format
python upload_imagenet_to_s3.py convert \
    --bucket "your-bucket-name" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### 2. Verify the Conversion
The script will show progress and confirm:
- ✅ Training data copied (preserving class structure)
- ✅ Validation data reorganized into class folders  
- ✅ Metadata and manifest created

### 3. Start Training
```bash
# Launch training with your converted dataset
python launch_sagemaker.py \
    --train-data-s3 "s3://your-bucket-name/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge" \
    --use-spot \
    --pipeline-stage "full_training"
```

### Training Options:
- `--train-data-s3`: S3 path to your dataset
- `--instance-type`: EC2 instance type for training
- `--use-spot`: Use spot instances (70% cost savings)
- `--pipeline-stage`: Training stage (see 7-stage pipeline below)
- `--epochs`: Number of training epochs

## 7-Stage Pipeline Support

Your SageMaker integration supports the complete 7-stage ImageNet training pipeline:

1. **lr_range_test**: Find optimal learning rate range
2. **lr_bounds**: Determine lower and upper LR bounds  
3. **onecycle_lr**: Test OneCycle learning rate policy
4. **batch_size_test**: Find optimal batch size
5. **weight_decay_tuning**: Optimize weight decay parameter
6. **full_training**: Complete model training
7. **monitoring**: Track and analyze training metrics

### Example: Run LR Range Test
```bash
python launch_sagemaker.py \
    --train-data-s3 "s3://your-bucket/imagenet-data/" \
    --pipeline-stage "lr_range_test" \
    --instance-type "ml.p3.2xlarge"
```

### Example: Full Training with Custom Parameters
```bash
python launch_sagemaker.py \
    --train-data-s3 "s3://your-bucket/imagenet-data/" \
    --pipeline-stage "full_training" \
    --instance-type "ml.p3.8xlarge" \
    --use-spot \
    --epochs 90 \
    --batch-size 256 \
    --learning-rate 0.1 \
    --weight-decay 1e-4
```

## Advanced Configuration

### Custom Hyperparameters
Override any hyperparameter from the original pipeline:

```bash
python launch_sagemaker.py \
    --train-data-s3 "s3://your-bucket/imagenet-data/" \
    --pipeline-stage "full_training" \
    --custom-param "optimizer=SGD" \
    --custom-param "momentum=0.9" \
    --custom-param "lr_scheduler=cosine"
```

### Multi-GPU Training
Use larger instances for faster training:

```bash
# 4 GPUs (32GB GPU memory)
--instance-type "ml.p3.8xlarge"

# 8 GPUs (64GB GPU memory)  
--instance-type "ml.p3.16xlarge"

# Latest generation (A100)
--instance-type "ml.p4d.24xlarge"
```

## Monitoring Training

Monitor your training job:

```bash
python monitor_training.py --job-name "your-training-job-name"
```

This provides:
- Real-time training metrics
- GPU/CPU utilization
- Cost tracking
- Error alerts

## Cost Optimization

### Spot Instances
- Use `--use-spot` for 70% cost savings
- Automatic checkpointing for fault tolerance
- Recommended for non-urgent training

### Instance Selection
- **Development/Testing**: `ml.p3.2xlarge` (1 GPU)
- **Production Training**: `ml.p3.8xlarge` (4 GPUs)  
- **Large Scale**: `ml.p3.16xlarge` (8 GPUs)

## Troubleshooting

### Common Issues:

1. **Upload Fails**: Check AWS credentials and bucket permissions
2. **Training Fails**: Verify S3 data structure and paths
3. **Out of Memory**: Reduce batch size or use larger instance
4. **Slow Training**: Use multi-GPU instances or increase batch size

### Debug Mode:
```bash
python launch_sagemaker.py --debug --train-data-s3 "s3://your-bucket/imagenet-data/"
```

## Next Steps

1. **Upload your ILSVRC dataset** using the upload script
2. **Start with LR range test** to find optimal parameters
3. **Run full training** with optimized hyperparameters
4. **Monitor and analyze** results using the monitoring tools

Your existing ImageNet training pipeline code remains unchanged - the SageMaker integration preserves all functionality while adding cloud scalability and cost optimization.