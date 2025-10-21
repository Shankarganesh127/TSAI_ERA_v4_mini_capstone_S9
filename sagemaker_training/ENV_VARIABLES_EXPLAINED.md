# SageMaker Environment Variables: Complete Example

## 🎯 **THE KEY POINT**: You DON'T manually set environment variables - SageMaker does it automatically!

---

## **Step-by-Step Example**

### **Step 1: You Configure Data Channels**
```python
# In launch_sagemaker.py
data_inputs = {
    'imagenet': TrainingInput(s3_data='s3://my-bucket/train-data/'),
    'validation': TrainingInput(s3_data='s3://my-bucket/val-data/')
}

estimator.fit(inputs=data_inputs, ...)
```

### **Step 2: SageMaker Automatically Sets Environment Variables**
```bash
# SageMaker creates these AUTOMATICALLY in the training container:
SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
```

### **Step 3: Your Code Reads the Environment Variables**
```python
# In sagemaker_wrapper.py
def parse_hyperparameters(self):
    parser.add_argument('--data_dir', 
                       default=os.environ.get('SM_CHANNEL_IMAGENET'))
    parser.add_argument('--val_dir', 
                       default=os.environ.get('SM_CHANNEL_VALIDATION'))
```

---

## **Real Example Commands**

### **Single Channel Setup:**
```bash
# Your command:
python launch_sagemaker.py \
    --s3-bucket "my-bucket" \
    --train-data-s3 "imagenet-dataset"

# SageMaker automatically creates:
# SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
```

### **Multi-Channel Setup:**
```bash
# Your command:
python launch_sagemaker.py \
    --s3-bucket "my-bucket" \
    --train-data-s3 "imagenet-train" \
    --val-data-s3 "imagenet-val"

# SageMaker automatically creates:
# SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
# SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
```

---

## **How This Appears in Your SageMaker Logs**

When your training job runs, you'll see these environment variables in the logs:
```
2025-10-21T14:42:37.249Z SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
2025-10-21T14:42:37.249Z SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
2025-10-21T14:42:37.249Z SM_MODEL_DIR=/opt/ml/model
2025-10-21T14:42:37.249Z SM_NUM_GPUS=4
2025-10-21T14:42:37.249Z SM_CURRENT_INSTANCE_TYPE=ml.p3.8xlarge
```

---

## **The Complete Flow**

```
1. You configure data_inputs = {'imagenet': ..., 'validation': ...}
   ↓
2. SageMaker downloads your S3 data to container paths
   ↓  
3. SageMaker sets environment variables pointing to those paths
   ↓
4. Your training script reads os.environ.get('SM_CHANNEL_...')
   ↓
5. Your script finds the data and trains the model
```

---

## **Environment Variable Rules**

| Channel Name | Environment Variable | Container Path |
|--------------|---------------------|----------------|
| `'imagenet'` | `SM_CHANNEL_IMAGENET` | `/opt/ml/input/data/imagenet` |
| `'validation'` | `SM_CHANNEL_VALIDATION` | `/opt/ml/input/data/validation` |
| `'test'` | `SM_CHANNEL_TEST` | `/opt/ml/input/data/test` |
| `'my_data'` | `SM_CHANNEL_MY_DATA` | `/opt/ml/input/data/my_data` |

---

## **Your Current Configuration Status**

✅ **Ready for Single Channel** (default):
```python
data_inputs = {'imagenet': train_input}
# → SM_CHANNEL_IMAGENET=/opt/ml/input/data/imagenet
```

✅ **Ready for Multi-Channel** (when you use --val-data-s3):
```python  
data_inputs = {
    'imagenet': train_input,      # → SM_CHANNEL_IMAGENET
    'validation': val_input       # → SM_CHANNEL_VALIDATION
}
```

✅ **Your wrapper already handles both**:
```python
args.data_dir = os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet')
args.val_dir = os.environ.get('SM_CHANNEL_VALIDATION', None)
```

---

## **Key Takeaway**

🎯 **You never manually set SM_CHANNEL_* variables!**

- ✅ Configure `data_inputs` in `launch_sagemaker.py` 
- ✅ SageMaker automatically sets environment variables
- ✅ Your training code reads the environment variables
- ✅ Everything works automatically!

The environment variables are **SageMaker's way of telling your training script where it put your data** after downloading it from S3!