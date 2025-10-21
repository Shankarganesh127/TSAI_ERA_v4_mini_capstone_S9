# Multi-Channel Data Configuration for SageMaker

## Option 1: Single Channel with train/val Subdirectories (Current Default)

### S3 Structure:
```
s3://your-bucket/imagenet-data/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### Launch Command:
```bash
python launch_sagemaker.py \
    --s3-bucket "your-bucket" \
    --train-data-s3 "imagenet-data" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30
```

### SageMaker Environment:
```
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
```

### In Container:
```
/opt/ml/input/data/imagenet/
├── train/
└── val/
```

---

## Option 2: Separate Channels for Train and Validation

### S3 Structure:
```
s3://your-bucket/
├── imagenet-train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── imagenet-val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### Launch Command:
```bash
python launch_sagemaker.py \
    --s3-bucket "your-bucket" \
    --train-data-s3 "imagenet-train" \
    --val-data-s3 "imagenet-val" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30
```

### SageMaker Environment:
```
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
```

### In Container:
```
/opt/ml/input/data/imagenet/      # Training data
├── n01440764/
├── n01443537/
└── ...

/opt/ml/input/data/validation/    # Validation data
├── n01440764/
├── n01443537/
└── ...
```

---

## Option 3: Different S3 Buckets

### Launch Command:
```bash
python launch_sagemaker.py \
    --s3-bucket "main-bucket" \
    --train-data-s3 "s3://train-bucket/imagenet-train" \
    --val-data-s3 "s3://val-bucket/imagenet-val" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30
```

---

## How Your Training Script Receives the Data

### Single Channel (Option 1):
```python
# In imagenet_training_pipeline.py
train_dir = args.data + "/train"     # /opt/ml/input/data/imagenet/train
val_dir = args.data + "/val"         # /opt/ml/input/data/imagenet/val
```

### Multi-Channel (Options 2 & 3):
```python
# In imagenet_training_pipeline.py
train_dir = args.data                # /opt/ml/input/data/imagenet
val_dir = args.val_data              # /opt/ml/input/data/validation
```

---

## Benefits of Each Approach

### Single Channel (Option 1):
✅ Simpler configuration
✅ Standard ImageNet structure
✅ Works with existing code
❌ Must download both train and val even if you only need one

### Multi-Channel (Options 2 & 3):
✅ Flexible data sources
✅ Can use different S3 buckets
✅ Download only what you need
✅ Better for large datasets
❌ Requires code updates to handle separate paths

---

## Implementation Status

✅ **Completed**: 
- Added `--val-data-s3` argument to `launch_sagemaker.py`
- Updated data input configuration to create separate channels
- Modified `sagemaker_wrapper.py` to handle `SM_CHANNEL_VALIDATION`
- Added validation directory parameter to training command

🔧 **Next Step**: 
Update your `imagenet_training_pipeline.py` to accept `--val-data` argument if using multi-channel setup.