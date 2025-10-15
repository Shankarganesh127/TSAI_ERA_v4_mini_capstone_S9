# 🔍 Universal Learning Rate Finder Toolkit

A comprehensive, production-ready learning rate optimization toolkit that works with any PyTorch model. Compare multiple LR finding methods, get automated recommendations, and visualize results professionally.

## 🌟 Features

- **🔧 Universal Compatibility**: Works with ANY PyTorch model (CNN, RNN, Transformer, Custom)
- **📊 Multiple Methods**: Linear, Exponential, and Cyclical LR range tests
- **🎯 Smart Recommendations**: Automated optimal LR detection with explanations
- **📈 Beautiful Visualizations**: Comprehensive comparison plots and analysis
- **💾 Professional Organization**: Structured results, models, and documentation
- **🔄 Complete Reproducibility**: All experiments fully documented and reusable

## 🚀 Quick Start

### Basic Usage (any model)
```python
from universal_lr_finder import LinearLRFinder
import torch.nn as nn

# Your model and data
model = YourPyTorchModel()
train_loader = YourDataLoader()
criterion = nn.CrossEntropyLoss()

# Find optimal learning rate
finder = LinearLRFinder(model, criterion)
results = finder.find_lr(train_loader)

# Use the recommendation
optimal_lr = results['optimal_lr']
optimizer = torch.optim.Adam(model.parameters(), lr=optimal_lr)
```

### Comprehensive Analysis
```python
# Compare all methods for best results
methods = {
    'Linear': LinearLRFinder(model, criterion),
    'Exponential': ExponentialLRFinder(model, criterion),
    'Cyclical': CyclicalLRFinder(model, criterion)
}

all_results = {}
for name, finder in methods.items():
    all_results[name] = finder.find_lr(train_loader)
    print(f"{name} optimal LR: {all_results[name]['optimal_lr']:.2e}")
```

## 📁 Project Structure

```
lr_optimization/
├── universal_lr_finder.ipynb          # Main notebook with all implementations
├── README.md                          # This documentation
├── experiments/                       # Experimental results and logs
│   ├── lr_finder_experiment_summary_*.json
│   ├── detailed_results_*/
│   ├── lr_finder_usage_guide_*.md
│   └── lr_finder_quick_reference_*.csv
├── models/                           # Saved model checkpoints
│   └── lr_finder_models_*/
├── plots/                            # Generated visualizations
│   ├── comprehensive_lr_comparison_*.png
│   ├── linear_lr_test_*.png
│   ├── exponential_lr_test_*.png
│   └── cyclical_lr_test_*.png
├── lr_finder_results/               # Raw LR finder data
│   └── method_comparison_*.csv
└── configs/                         # Configuration files
    └── session_config.json
```

## 🛠️ Available Methods

### 1. Linear LR Range Test
- **Best for**: New models, detailed analysis, conservative estimates
- **How it works**: Increases LR linearly from min to max
- **Pros**: Thorough exploration, stable results
- **Cons**: Slower for large LR ranges

```python
finder = LinearLRFinder(model, criterion)
results = finder.find_lr(train_loader, min_lr=1e-7, max_lr=1.0)
```

### 2. Exponential LR Range Test  
- **Best for**: Quick tests, unknown LR ranges, large models
- **How it works**: Increases LR exponentially (multiplicative steps)
- **Pros**: Fast exploration, covers wide ranges efficiently  
- **Cons**: May miss fine-grained details

```python
finder = ExponentialLRFinder(model, criterion)
results = finder.find_lr(train_loader, min_lr=1e-7, max_lr=10.0)
```

### 3. Cyclical LR Range Test
- **Best for**: Noisy datasets, robust estimates, cyclical LR planning
- **How it works**: Uses cyclical patterns (triangular, cosine, exponential)
- **Pros**: Multiple confirmations, robust to outliers
- **Cons**: More complex analysis required

```python
finder = CyclicalLRFinder(model, criterion)
results = finder.find_lr(train_loader, min_lr=1e-6, max_lr=1.0, 
                         num_cycles=2, cycle_pattern='triangular')
```

## 📊 Comparison & Analysis

The toolkit automatically generates comprehensive comparison visualizations:

- **Loss vs Learning Rate**: Side-by-side method comparison
- **Optimal LR Analysis**: Statistical consensus and recommendations
- **Gradient Behavior**: Gradient norms across different LRs
- **Method Statistics**: Detailed performance metrics
- **Usage Recommendations**: When to use each method

## 🎯 Method Selection Guide

| Scenario | Recommended Method | Reasoning |
|----------|-------------------|-----------|
| 🆕 New model/dataset | Linear | Conservative, detailed exploration |
| ⚡ Quick assessment | Exponential | Fast, efficient range coverage |
| 📊 Noisy/unstable data | Cyclical | Robust, multiple confirmations |
| 🔬 Research/comparison | All methods | Complete analysis |
| 🏭 Production deployment | Linear + Exponential | Balance of speed and accuracy |

## 💡 Use Cases

### 1. **Research & Development**
```python
# Compare LR sensitivity across different architectures
models = {'ResNet': resnet_model, 'Transformer': transformer_model}
for name, model in models.items():
    finder = LinearLRFinder(model, criterion)
    results = finder.find_lr(train_loader)
    print(f"{name} optimal LR: {results['optimal_lr']:.2e}")
```

### 2. **Production Model Training**
```python
# Establish optimal LR baseline for production
finder = ExponentialLRFinder(production_model, criterion)
results = finder.find_lr(production_data, min_lr=1e-6, max_lr=1.0)
baseline_lr = results['optimal_lr']

# Use in production training
optimizer = torch.optim.AdamW(production_model.parameters(), lr=baseline_lr)
```

### 3. **Hyperparameter Optimization**
```python
# Integrate with hyperparameter search
def objective(trial):
    # Use LR finder to suggest starting point
    finder = LinearLRFinder(model, criterion)
    lr_results = finder.find_lr(train_loader)
    suggested_lr = lr_results['optimal_lr']
    
    # Use as starting point for optimization
    lr = trial.suggest_float('lr', suggested_lr/10, suggested_lr*10, log=True)
    # ... rest of training
```

### 4. **Educational/Debugging**
```python
# Understand model behavior across LR ranges
finder = CyclicalLRFinder(model, criterion)
results = finder.find_lr(train_loader, num_cycles=3)
finder.plot_results()  # Visualize learning behavior
```

## 🔧 Advanced Configuration

### Custom Models
```python
class YourCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return output

# Works seamlessly with any custom model
finder = LinearLRFinder(YourCustomModel(), criterion)
```

### Custom Optimizers
```python
# Use any optimizer class
finder = LinearLRFinder(model, criterion, optimizer_class=torch.optim.SGD)
finder = ExponentialLRFinder(model, criterion, optimizer_class=torch.optim.AdamW)
```

### Advanced Parameters
```python
results = finder.find_lr(
    train_loader,
    min_lr=1e-8,              # Lower bound
    max_lr=10.0,              # Upper bound  
    num_batches=200,          # More thorough search
    stop_div_factor=5.0,      # More aggressive early stopping
    smooth_factor=0.95        # Less smoothing
)
```

## 📋 Requirements

### Core Requirements
- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- Seaborn
- Pandas
- tqdm

### Optional (for enhanced features)
- Plotly (interactive visualizations)
- Torchvision (for CIFAR-10 demo)

### Installation
```bash
pip install torch torchvision numpy matplotlib seaborn pandas tqdm plotly
```

## 📚 Output Files

The toolkit generates comprehensive documentation:

- **`lr_finder_experiment_summary_*.json`**: Complete experiment metadata
- **`lr_finder_usage_guide_*.md`**: Detailed usage instructions
- **`lr_finder_quick_reference_*.csv`**: Quick comparison table
- **Individual method data**: CSV and JSON formats for analysis
- **Model checkpoints**: For complete reproducibility
- **Visualization plots**: High-quality comparison charts

## 🎓 Educational Value

Perfect for:
- **Teaching LR optimization concepts**
- **Demonstrating different search strategies**
- **Understanding model sensitivity to LR**
- **Comparing optimization approaches**
- **Learning best practices in ML experiments**

## 🤝 Contributing

This toolkit is designed to be extensible. Ideas for contributions:

- Additional LR finding methods (cosine restarts, warm restarts)
- Multi-GPU support for large models
- Integration with popular frameworks (PyTorch Lightning, Transformers)
- Automatic hyperparameter optimization integration
- Time series and NLP specific optimizations

## 📞 Support

For issues or questions:
1. Check the generated usage guide in `experiments/`
2. Review the comprehensive plots in `plots/`
3. Examine the example notebook cells
4. Verify your PyTorch and data loader compatibility

## 🏆 Success Stories

This toolkit enables:
- **Faster model convergence** through optimal LR selection
- **Reduced training time** by avoiding LR trial-and-error
- **Better model performance** through proper LR optimization
- **Standardized workflows** across different projects
- **Educational insights** into learning rate dynamics

---

**🎉 Universal Learning Rate Finder Toolkit - Making LR optimization accessible to everyone!**

*Created with ❤️ for the deep learning community*

## 📄 License

This project is open source and available under the MIT License.

---

*Last updated: October 2024*

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