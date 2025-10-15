# Learning Rate Optimization Tools

This folder contains tools for finding and optimizing learning rates for your ResNet50 training experiments.

## 📁 Contents

- `learning_rate_finder.ipynb` - Interactive Learning Rate Range Test implementation
- `README.md` - This documentation file

## 🎯 What is Learning Rate Finding?

The Learning Rate Finder implements the **Learning Rate Range Test** technique:

- **Systematically tests** different learning rates during training
- **Plots loss vs learning rate** to visualize the relationship  
- **Identifies optimal range** where loss decreases most rapidly
- **Prevents poor convergence** due to suboptimal learning rates

## 🚀 Quick Start

1. **Open the notebook:**
   ```bash
   cd lr_optimization
   jupyter notebook learning_rate_finder.ipynb
   ```

2. **Configure your dataset** (automatically detects Tiny ImageNet, Imagenette, etc.)

3. **Run the learning rate finder** using the interactive cells

4. **Analyze the results** and get learning rate recommendations

5. **Use the suggested LR** in your training:
   ```bash
   cd ..
   python train_imagenet.py --lr YOUR_OPTIMAL_LR
   ```

## 📊 Features

### 🔍 Smart Dataset Detection
- **Auto-configures** for Tiny ImageNet (200 classes, 64x64)
- **Auto-configures** for Imagenette/ImageWoof (10 classes, 224x224)
- **Flexible setup** for custom datasets

### 📈 Advanced Analysis
- **Steepest descent detection** - where loss decreases fastest
- **Minimum loss point** - lowest loss achieved
- **Conservative recommendations** - safer starting points
- **Gradient analysis** - visual guidance for optimal LR

### 🎯 Smart Recommendations
- **Conservative start** - 1/10 of steepest descent point
- **Optimal range** - steepest descent learning rate
- **Aggressive option** - for experienced users
- **Dataset-specific advice** - tailored to your data

### 📊 Visual Analysis
- **Loss vs LR plot** with marked recommendations
- **Gradient plot** showing steepest descent regions
- **Progress tracking** during LR search
- **Results export** for further analysis

## 🔧 Configuration Options

### LR Finder Settings
```python
LR_FINDER_CONFIG = {
    'min_lr': 1e-7,           # Start learning rate
    'max_lr': 10.0,           # End learning rate  
    'num_iterations': 100,    # Number of LR tests
    'beta': 0.98,             # Loss smoothing factor
    'stop_div_factor': 5,     # Stop if loss explodes
    'batch_size': 128,        # Batch size for testing
    'subset_size': 5000,      # Dataset subset size
}
```

### Model Auto-Configuration
- **Tiny ImageNet**: 200 classes, no pretrained weights
- **Imagenette/ImageWoof**: 10 classes, pretrained recommended
- **Custom datasets**: Configurable classes and pretrained options

## 💡 How to Interpret Results

### 📊 Key Metrics
- **Steepest Descent LR**: Where loss decreases fastest (recommended)
- **Minimum Loss LR**: Where lowest loss is achieved
- **Conservative LR**: Safe starting point (1/10 of steepest)

### 🎯 Choosing Learning Rate
1. **Start Conservative**: Use conservative LR for initial training
2. **Monitor Training**: Watch loss curves for stability
3. **Increase if Stable**: Move toward steepest descent LR
4. **Use Scheduling**: Implement LR decay during training

### ⚠️ Warning Signs
- **Loss explodes**: LR too high, use more conservative
- **Training too slow**: LR too low, increase gradually
- **No improvement**: Check model architecture or data

## 🔗 Integration with Training

The LR finder automatically works with:
- `../train_imagenet.py` - Main training script
- `../imagenet_models.py` - ResNet50 implementation
- `../imagenet_dataset.py` - Dataset loading utilities

## 📊 Dataset-Specific Guidance

### 🔹 Tiny ImageNet (200 classes, 64x64)
- **Typical LR range**: 1e-3 to 1e-1
- **Recommended epochs**: 50-100
- **Batch size**: 256-512 (smaller images)
- **Strategy**: Start with found LR, use step scheduling

### 🔸 Imagenette/ImageWoof (10 classes, 224x224)
- **Typical LR range**: 1e-4 to 1e-2  
- **Recommended epochs**: 20-50
- **Batch size**: 128-256
- **Strategy**: Use pretrained weights + found LR

### 🔹 Custom Datasets
- **Adjust subset_size** based on dataset size
- **Consider class imbalance** in LR selection
- **Monitor validation loss** during LR finding

## 🚀 Advanced Tips

### 📈 Learning Rate Scheduling
After finding optimal LR:
```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Warmup + decay
# Start with LR/10, warmup to found LR, then decay
```

### 🎯 Training Strategy
1. **Warmup**: Start with LR/10 for first 5 epochs
2. **Main training**: Use found LR for most epochs
3. **Fine-tuning**: Reduce LR by 10x for final epochs

### 📊 Monitoring
- **Save LR finder results** for future reference
- **Compare results** across different datasets
- **Track training curves** to validate LR choice

## 🎉 Success Indicators

✅ **Good LR Finding Results:**
- Clear decrease in loss during LR sweep
- Obvious steepest descent region
- Smooth loss curves without explosions
- Reasonable LR range (not too extreme)

❌ **Poor Results (Re-run with adjustments):**
- Loss doesn't decrease significantly
- Very noisy loss curves
- Immediate loss explosion
- No clear optimal region

## 🔄 Next Steps

After finding your optimal learning rate:

1. **Test with short training** (5 epochs) to verify stability
2. **Use learning rate scheduling** for longer training
3. **Monitor validation metrics** to ensure generalization
4. **Fine-tune based on results** and adjust if needed

Happy optimizing! 🚀