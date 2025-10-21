# How SageMaker Environment Variables Are Set for Training

## The Complete Flow

### 1. **Data Channel Configuration → Environment Variables**

When you configure data channels in `launch_sagemaker.py`, SageMaker **automatically** creates environment variables following this pattern:

```python
# Channel name in data_inputs dict → Environment variable
data_inputs = {
    'imagenet': train_input,      # → SM_CHANNEL_IMAGENET
    'validation': val_input       # → SM_CHANNEL_VALIDATION
}
```

### 2. **Single Channel Example**

#### Command:
```bash
python launch_sagemaker.py \
    --s3-bucket "my-bucket" \
    --train-data-s3 "imagenet-dataset"
```

#### What happens in `launch_sagemaker.py`:
```python
data_inputs = {'imagenet': train_input}
estimator.fit(inputs=data_inputs, ...)
```

#### SageMaker automatically sets:
```bash
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
```

#### In your container:
```
/opt/ml/input/data/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### 3. **Multi-Channel Example**

#### Command:
```bash
python launch_sagemaker.py \
    --s3-bucket "my-bucket" \
    --train-data-s3 "imagenet-train" \
    --val-data-s3 "imagenet-val"
```

#### What happens in `launch_sagemaker.py`:
```python
data_inputs = {
    'imagenet': train_input,      # From --train-data-s3
    'validation': val_input       # From --val-data-s3
}
estimator.fit(inputs=data_inputs, ...)
```

#### SageMaker automatically sets:
```bash
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
```

#### In your container:
```
/opt/ml/input/data/imagenet/       # Training data
├── n01440764/
├── n01443537/
└── ...

/opt/ml/input/data/validation/     # Validation data
├── n01440764/
├── n01443537/
└── ...
```

## 4. **How Your Code Reads Environment Variables**

### In `sagemaker_wrapper.py`:
```python
def parse_hyperparameters(self):
    parser = argparse.ArgumentParser()
    
    # SageMaker automatically sets these environment variables
    parser.add_argument('--data_dir', type=str, 
                       default=os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet'))
    parser.add_argument('--val_dir', type=str, 
                       default=os.environ.get('SM_CHANNEL_VALIDATION', None))
```

### Environment Variable Naming Rule:
```
Channel name: 'imagenet'     → SM_CHANNEL_IMAGENET
Channel name: 'validation'   → SM_CHANNEL_VALIDATION  
Channel name: 'test'         → SM_CHANNEL_TEST
Channel name: 'my_data'      → SM_CHANNEL_MY_DATA
```

## 5. **Real SageMaker Environment Variables**

### Standard SageMaker Variables (Always Available):
```bash
SM_MODEL_DIR=/opt/ml/model
SM_OUTPUT_DATA_DIR=/opt/ml/output/data
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation  # Only if you configure it
SM_NUM_GPUS=4
SM_CURRENT_INSTANCE_TYPE=ml.p3.8xlarge
SM_TRAINING_JOB_NAME=imagenet-7stage-20251021-143629
```

### Hyperparameters (From your launch command):
```bash
SM_HP_EPOCHS=30
SM_HP_NUM_WORKERS=4
SM_HP_QUICK_MODE=false
SM_HP_RUN_LR_FINDER=true
SM_HP_RUN_WD_SEARCH=true
```

## 6. **Testing Environment Variables Locally**

You can test your environment variable handling locally:

```python
# test_env_vars.py
import os

# Mock SageMaker environment
os.environ['SM_CHANNEL_IMAGENET'] = '/path/to/local/train/data'
os.environ['SM_CHANNEL_VALIDATION'] = '/path/to/local/val/data'
os.environ['SM_MODEL_DIR'] = '/path/to/local/output'

# Test your wrapper
from sagemaker_training.sagemaker_wrapper import ImageNetSageMakerTrainer
trainer = ImageNetSageMakerTrainer()
args = trainer.parse_hyperparameters()
print(f"Train data: {args.data_dir}")
print(f"Val data: {args.val_dir}")
```

## 7. **Key Points**

### ✅ **Automatic**: 
- You don't manually set environment variables
- SageMaker creates them from your `data_inputs` dictionary

### ✅ **Naming Convention**: 
- Channel name `'imagenet'` → `SM_CHANNEL_IMAGENET`
- Always uppercase, prefixed with `SM_CHANNEL_`

### ✅ **Path Format**: 
- Always `/opt/ml/input/data/{channel_name}`
- Data is downloaded to these paths automatically

### ✅ **Flexible**: 
- Add any number of channels
- Use different S3 buckets/paths
- Channel names can be anything (`train`, `val`, `test`, `augmented`, etc.)

## 8. **Your Current Setup Status**

✅ **Single Channel** (works now):
```python
data_inputs = {'imagenet': train_input}
# → SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
```

✅ **Multi-Channel** (ready to use):
```python
data_inputs = {
    'imagenet': train_input,
    'validation': val_input  
}
# → SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
# → SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
```

The environment variables are set **automatically by SageMaker** based on your channel configuration!