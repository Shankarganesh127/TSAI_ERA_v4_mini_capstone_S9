# 📋 Quick Reference Card

**SageMaker ImageNet Training with Automatic Model Replacement**

## ⚡ Quick Commands

### 🚀 Start Training (Complete Automation)
```bash
python sagemaker_orchestrator.py
```
**Result**: Complete ImageNet training with automatic model replacement every epoch

### 🔧 Manual Training Setup
```bash
# 1. Convert dataset (if needed)
python s3_dataset_converter.py --bucket "my-bucket" --source-prefix "ILSVRC"

# 2. Launch training
python launch_sagemaker.py --job-name "my-job" --enable-model-replacement

# 3. Monitor progress  
python monitor_training.py --job-name "my-job"
```

### 🧪 Test System
```bash
python test_model_replacement.py
```

---

## 📁 Key Files & Their Purpose

| File | What It Does | When to Use |
|------|--------------|-------------|
| `sagemaker_orchestrator.py` | **Complete automated pipeline** | Want everything done automatically |
| `launch_sagemaker.py` | Direct SageMaker job launcher | Need custom control over training |
| `s3_dataset_converter.py` | Convert ILSVRC → SageMaker format | Have raw ILSVRC data in S3 |
| `monitor_training.py` | Real-time training monitoring | Track running training jobs |
| `model_saver.py` | **Automatic model replacement** | Used automatically by system |
| `training_integration.py` | Background model monitoring | Used automatically by system |

---

## 🔄 Model Replacement System

### **What Happens Automatically**
- ✅ **Saves model every epoch** 
- ✅ **Replaces previous epoch model** (no accumulation)
- ✅ **Tracks best model by accuracy**
- ✅ **Creates deployment archives**
- ✅ **No code changes needed**

### **Model Output Structure**
```
models/
├── model_current.pth    # Latest epoch (REPLACES each time)
├── model_best.pth       # Best accuracy model
└── model_final.pth      # Final training state
```

---

## 🎯 Pipeline Stages (7-Step Methodology)

| Stage | Purpose | Usage |
|-------|---------|-------|
| `lr_range_test` | Find optimal learning rate | `--pipeline-stage "lr_range_test"` |
| `lr_bounds` | Determine LR boundaries | `--pipeline-stage "lr_bounds"` |
| `onecycle_lr` | Test OneCycle learning rate | `--pipeline-stage "onecycle_lr"` |
| `batch_size_test` | Find optimal batch size | `--pipeline-stage "batch_size_test"` |
| `weight_decay_tuning` | Optimize weight decay | `--pipeline-stage "weight_decay_tuning"` |
| `full_training` | Complete model training | `--pipeline-stage "full_training"` |
| `monitoring` | Results analysis | `--pipeline-stage "monitoring"` |

---

## 💰 Cost Optimization

### **Instance Types**
- **Development**: `ml.p3.2xlarge` (1 GPU, ~$3/hour)
- **Production**: `ml.p3.8xlarge` (4 GPUs, ~$12/hour)  
- **Maximum**: `ml.p3.16xlarge` (8 GPUs, ~$24/hour)
- **Budget**: `ml.g4dn.2xlarge` (1 T4, ~$0.75/hour)

### **Spot Instances (70% Savings)**
```bash
# Add to any training command
--spot-training
```

---

## 🔍 Monitoring & Verification

### **Check Training Progress**
```bash
python monitor_training.py --job-name "your-job"
```

### **Verify Model Replacement**
```bash
# Check models during training (should only see these 3 files)
aws s3 ls s3://your-bucket/sagemaker-output/models/
# Expected: model_current.pth, model_best.pth, model_final.pth
```

### **View Training Summary**
```bash
aws s3 cp s3://your-bucket/sagemaker-output/model_training_summary.json - | jq
```

---

## 🆘 Common Issues & Solutions

### **AWS Permission Errors**
```bash
python setup_environment.py --validate
```

### **Dataset Not Found**
```bash
aws s3 ls s3://your-bucket/ILSVRC/Data/CLS-LOC/
python s3_dataset_converter.py --dry-run --bucket "your-bucket"
```

### **Model Replacement Not Working**
```bash
python test_model_replacement.py
```

### **Training Job Fails**
```bash
python monitor_training.py --job-name "your-job" --debug
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

---

## ⚙️ Configuration Files

### **Main Config**: `configs/pipeline_config.json`
```json
{
    "model_replacement": {"enable": true, "replace_previous_model": true},
    "training": {"instance_type": "ml.p3.8xlarge", "use_spot_instances": true},
    "pipeline": {"stages": ["full_training"], "enable_checkpointing": true}
}
```

### **Dependencies**
```bash
pip install -r requirements.txt           # Main dependencies
pip install -r converter_requirements.txt # Dataset conversion
```

---

## 📖 Documentation Quick Links

- **Complete Setup Guide**: [`README.md`](documentation/README.md)
- **Entry Point Selection**: [`MAIN_ENTRY_POINTS.md`](MAIN_ENTRY_POINTS.md)  
- **Detailed File Reference**: [`documentation/COMPLETE_FILE_REFERENCE.md`](documentation/COMPLETE_FILE_REFERENCE.md)
- **Model Replacement Details**: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)
- **Dataset Conversion**: [`documentation/S3_DATASET_CONVERTER_README.md`](documentation/S3_DATASET_CONVERTER_README.md)

---

## 🎉 Success Checklist

### ✅ **Environment Ready**
- [ ] AWS credentials configured
- [ ] Python dependencies installed  
- [ ] SageMaker permissions verified
- [ ] S3 bucket accessible

### ✅ **Dataset Ready**  
- [ ] ILSVRC data in S3
- [ ] Dataset converted (if needed)
- [ ] Manifest files created

### ✅ **Training Ready**
- [ ] Model replacement enabled
- [ ] Instance type selected
- [ ] Spot instances configured (optional)
- [ ] Monitoring setup

### ✅ **Training Success**
- [ ] Training job started successfully
- [ ] Model replacement working (3 files only)
- [ ] Real-time monitoring active
- [ ] Best model being tracked

---

## 🚀 **One-Line Complete Setup**
```bash
python sagemaker_orchestrator.py
```
**🎯 This single command provides complete ImageNet training with automatic model replacement every epoch!**

---

*Print this card for quick reference during training* 📋