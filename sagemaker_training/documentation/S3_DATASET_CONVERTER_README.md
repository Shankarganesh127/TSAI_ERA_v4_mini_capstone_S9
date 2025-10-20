# S3 Dataset Converter for SageMaker

Simple tool to convert existing S3 ILSVRC ImageNet dataset to SageMaker-compatible format.

## Purpose

Converts your existing S3 ILSVRC dataset structure to the format required by SageMaker for distributed training.

### Input Structure (ILSVRC)
```
s3://your-bucket/ILSVRC/
├── Data/CLS-LOC/train/     # 1000 class folders
├── Data/CLS-LOC/val/       # 50,000 flat validation images
├── Data/CLS-LOC/test/      # 100,000 flat test images (optional)
└── ImageSets/CLS-LOC/
    ├── val.txt             # Validation labels
    └── test.txt            # Test labels (optional)
```

### Output Structure (SageMaker-compatible)
```
Original ILSVRC training data (unchanged):
s3://your-bucket/ILSVRC/Data/CLS-LOC/train/  # Used directly (already organized)

New converted data:
s3://your-bucket/imagenet-sagemaker/
├── val/                    # 1000 class folders (reorganized from flat)
├── test/                   # 1000 class folders (reorganized from flat, if exists)
├── metadata/               # Dataset metadata (points to original train + converted val/test)
└── manifest.json           # SageMaker manifest (mixed paths)
```

## Installation

1. Install required dependencies:
```bash
pip install -r converter_requirements.txt
```

2. Configure AWS credentials:
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option 3: AWS profile
aws configure --profile your-profile
```

## Usage

### Basic Conversion
```bash
python s3_dataset_converter.py \
    --bucket "your-s3-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker"
```

### With AWS Profile
```bash
python s3_dataset_converter.py \
    --bucket "your-s3-bucket" \
    --source-prefix "ILSVRC" \
    --target-prefix "imagenet-sagemaker" \
    --aws-profile "your-aws-profile"
```

## Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--bucket` | Yes | S3 bucket containing ILSVRC dataset | `my-imagenet-bucket` |
| `--source-prefix` | Yes | S3 prefix of existing ILSVRC data | `ILSVRC` |
| `--target-prefix` | Yes | S3 prefix for SageMaker format | `imagenet-sagemaker` |
| `--aws-profile` | No | AWS profile to use | `my-aws-profile` |

## Conversion Process

The converter performs these optimized steps:

### 1. Training Data (Skipped) ⚡
- **Source**: `ILSVRC/Data/CLS-LOC/train/` (already organized)
- **Action**: **No copy needed** - training data used directly from original location
- **Benefit**: Saves time and storage space

### 2. Validation Data Reorganization
- **Source**: `ILSVRC/Data/CLS-LOC/val/` (flat structure)
- **Target**: `imagenet-sagemaker/val/` (class folders)
- **Action**: Uses `ImageSets/CLS-LOC/val.txt` to organize images into class folders

### 3. Test Data Reorganization (Optional)
- **Source**: `ILSVRC/Data/CLS-LOC/test/` (flat structure)
- **Target**: `imagenet-sagemaker/test/` (class folders)
- **Action**: Uses `ImageSets/CLS-LOC/test.txt` if available, or distributes evenly across classes

### 4. Metadata Creation
- **Creates**: `metadata/dataset_metadata.json`
- **Contains**: Mixed paths - original training location + converted val/test locations
- **Purpose**: SageMaker training configuration with optimized data paths

### 5. Manifest Generation
- **Creates**: `manifest.json`
- **Contains**: 
  - `train_data`: Points to original ILSVRC location
  - `val_data`: Points to converted location
  - `test_data`: Points to converted location (if exists)
- **Purpose**: Training job data input specification

## Output Verification

After conversion, verify the structure:

```bash
# Check converted structure
aws s3 ls s3://your-bucket/imagenet-sagemaker/

# Verify training classes
aws s3 ls s3://your-bucket/imagenet-sagemaker/train/ | head -10

# Verify validation classes
aws s3 ls s3://your-bucket/imagenet-sagemaker/val/ | head -10

# Verify test classes (if converted)
aws s3 ls s3://your-bucket/imagenet-sagemaker/test/ | head -10

# Check metadata
aws s3 cp s3://your-bucket/imagenet-sagemaker/manifest.json - | jq
```

## Usage with SageMaker Training

After conversion, use with SageMaker:

```bash
python launch_sagemaker.py \
    --job-name "imagenet-training" \
    --role-arn "arn:aws:iam::123456789:role/SageMakerRole" \
    --train-data-s3 "s3://your-bucket/imagenet-sagemaker/" \
    --instance-type "ml.p3.8xlarge" \
    --spot-training
```

## Error Handling

### Common Issues

1. **AWS Permissions Error**
   ```
   Solution: Ensure your AWS credentials have S3 read/write access:
   - s3:GetObject
   - s3:PutObject
   - s3:ListBucket
   ```

2. **Source Data Not Found**
   ```
   Solution: Verify your ILSVRC data exists:
   aws s3 ls s3://your-bucket/ILSVRC/Data/CLS-LOC/
   ```

3. **Validation Mapping Missing**
   ```
   Warning: If val.txt is missing, images will be distributed evenly across classes.
   This still works for training but may not preserve original validation split.
   ```

## Performance

- **Training Data**: Direct S3 copy (fast)
- **Validation Data**: Reorganization by class (slower, but one-time)
- **Typical Time**: 30-60 minutes for full ImageNet dataset
- **Cost**: Only S3 request costs (no data transfer within same region)

## Logs

The converter creates detailed logs showing:
- Progress for each conversion step
- Number of files processed
- Any errors or warnings
- Final S3 paths for training

Log example:
```
🔄 Converting S3 ILSVRC dataset to SageMaker format
📂 Source: s3://my-bucket/ILSVRC/
🎯 Target: s3://my-bucket/imagenet-sagemaker/
📁 Step 1: Copying training data...
   Copied 1000 files...
   Total files copied: 1281167
📁 Step 2: Reorganizing validation data by class...
   Reorganized 1000 validation images...
   Total validation images reorganized: 50000
📁 Step 3: Creating SageMaker metadata...
📋 Metadata created: s3://my-bucket/imagenet-sagemaker/metadata/dataset_metadata.json
📋 Manifest created: s3://my-bucket/imagenet-sagemaker/manifest.json
🎉 S3 ILSVRC to SageMaker conversion completed!
```

## Next Steps

After successful conversion:

1. **Verify structure** using AWS CLI commands above
2. **Launch SageMaker training** using the converted dataset
3. **Monitor training** using SageMaker console or monitoring tools
4. **Clean up** original ILSVRC data if no longer needed (optional)

Your dataset is now ready for high-performance SageMaker training with optimal data loading and distributed processing!