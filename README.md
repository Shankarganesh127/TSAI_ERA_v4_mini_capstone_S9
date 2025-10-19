# TSAI ERAv4 Mini Capstone S9 🚀

Advanced ImageNet training with systematic hyperparameter optimization and simplified Tiny-ImageNet training.

## 🌟 **NEW: Complete 7-Step ImageNet Training Pipeline**

A systematic approach to ImageNet training with automated hyperparameter optimization:

### 🎯 **Quick Start - ImageNet Pipeline**

```bash
# Full production pipeline (recommended)
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode full

# Quick testing mode (faster iterations)
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode quick

# Minimal validation (very fast)
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode test

# Direct execution with uv
uv run python imagenet_training_pipeline.py --data /path/to/imagenet --quick-mode
```

### 🔬 **The 7-Step Process**
1. **LR Range Test** → Finds optimal learning rate bounds
2. **Pick LR bounds** → Extracts min/max LR from range test  
3. **OneCycle LR + cyclical momentum** → Configures advanced scheduler
4. **Choose batch size** → Auto-detects optimal GPU memory usage
5. **Tune weight-decay & regularizers** → Grid search with validation
6. **Full OneCycle training** → Complete training with all optimizations
7. **Monitor & iterate** → Comprehensive analysis and visualization

---

## 🎯 **Simplified Tiny-ImageNet Training**

Single-file implementation with all features:

```bash
# Navigate to tiny-imagenet training
cd tiny_imagenet_training

# Full training (recommended)
uv run python train.py --data ./datasets/tiny-imagenet-200 --out ./runs --epochs 20 --batch-size 128

# Quick test (1 epoch)
uv run python train.py --data ./datasets/tiny-imagenet-200 --out ./runs_test --epochs 1 --batch-size 64

# Custom configuration
uv run python train.py --data ./datasets/tiny-imagenet-200 --out ./runs_custom \
  --epochs 50 --batch-size 256 --lr-max 0.2 --wd 5e-4
```

**Features**: Mixed precision (AMP), OneCycleLR, gradient accumulation, label smoothing, clean progress bars, no warnings.

---

## 📁 **Complete Project Structure**

```
TSAI_ERAv4_mini_capstone_S9/
├── 🚀 IMAGENET TRAINING PIPELINE (NEW)
│   ├── imagenet_training_pipeline.py    # Complete 7-step pipeline
│   ├── run_imagenet_pipeline.sh         # Easy launcher script  
│   ├── test_imagenet_pipeline.py        # Pipeline validation tests
│   ├── IMAGENET_PIPELINE_README.md      # Detailed pipeline docs
│   └── IMPLEMENTATION_SUMMARY.md        # Complete overview
│
├── �� TINY-IMAGENET SIMPLIFIED (NEW)
│   └── tiny_imagenet_training/
│       ├── train.py                     # Single-file complete training
│       ├── pyproject.toml              # Simplified dependencies
│       ├── README.md                   # Tiny-ImageNet docs
│       ├── datasets/                   # Dataset location
│       ├── runs*/                      # Training outputs
│       └── logs/                       # Training logs
│
├── 📊 IMAGENET CORE MODULES
│   ├── main.py                         # User-friendly training launcher (recommended)
│   ├── train_imagenet.py               # Core ImageNet training engine
│   ├── imagenet_models.py              # ResNet50 for ImageNet
│   ├── imagenet_dataset.py             # Dataset loaders
│   ├── logger_setup.py                 # Logging system
│   └── utils.py                        # Training utilities
│
├── 🔬 LR OPTIMIZATION SUITE
│   └── lr_optimization/
│       ├── learning_rate_finder.ipynb         # Interactive LR finder
│       ├── imagenet_lr_finder_*.ipynb         # ImageNet-specific LR tools
│       ├── imagenet_one_cycle.ipynb           # OneCycle analysis
│       ├── universal_lr_finder.ipynb          # Universal LR finder
│       ├── paper_project_step_by_step.ipynb   # Research methodology
│       ├── configs/                           # LR configurations
│       ├── experiments/                       # LR experiments
│       ├── results/                           # LR optimization results
│       └── README.md                          # LR optimization docs
│
├── 🗂️ DATA ANALYSIS & TOOLS
│   ├── dataset_analysis/                      # Dataset statistics
│   ├── imagenet_data_exploration/             # Data exploration tools
│   ├── imagenet_dataset_tools/                # Dataset utilities
│   └── image_process_lib/                     # Image processing
│
├── 📚 DOCUMENTATION & SETUP
│   ├── docs/                                  # Project documentation
│   ├── setup/                                 # Setup scripts
│   ├── README_ImageNet_Setup.md               # ImageNet setup guide
│   └── pyproject.toml                         # Main dependencies
│
└── 📁 OUTPUTS & LOGS
    ├── datasets/                              # Dataset storage
    ├── logs/                                  # Training logs
    └── dist/                                  # Build artifacts
```


---

## 🚀 **Quick Start Guide**

### **Option 1: ImageNet Pipeline (Production)**
```bash
# Validate pipeline works
uv run python test_imagenet_pipeline.py

# Run complete pipeline
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode full --epochs 90

# Quick testing
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode quick --epochs 20
```

### **Option 2: Tiny-ImageNet Training (Learning)**
```bash
cd tiny_imagenet_training

# Quick start (1 epoch test)
uv run python train.py --data ./datasets/tiny-imagenet-200 --out ./runs_test --epochs 1

# Full training (20 epochs)
uv run python train.py --data ./datasets/tiny-imagenet-200 --out ./runs --epochs 20
```

### **Option 3: Traditional ImageNet Training (main.py)**
```bash
# Basic training with auto-detected dataset
uv run python main.py

# Specify dataset path
uv run python main.py --data-dir /path/to/imagenet

# Quick mode (10 epochs, batch size 128)
uv run python main.py --quick

# Test mode (1 epoch, batch size 32)
uv run python main.py --test

# Custom configuration
uv run python main.py --data-dir /path/to/imagenet --epochs 50 --batch-size 256 --lr 0.05

# Fine-tuning with pretrained weights
uv run python main.py --pretrained --epochs 20 --lr 0.01

# Show project info and system details
uv run python main.py --info
```

**Key Features of main.py:**
- 🔍 **Auto-dataset detection** - Searches common locations for ImageNet datasets
- 🎛️ **User-friendly interface** - Simplified command-line arguments
- 📊 **Comprehensive logging** - Detailed progress and configuration logging  
- 🔧 **Convenience modes** - Built-in `--quick` and `--test` modes
- ⚙️ **Error handling** - Validates dataset structure and dependencies
- 🚀 **Training launcher** - Internally calls `train_imagenet.py` with optimized parameters

#### **Complete main.py Arguments:**
```bash
# Dataset options
--data-dir PATH       # ImageNet dataset path (auto-detected if not specified)

# Training parameters
--epochs N           # Number of training epochs (default: 90)
--batch-size N       # Batch size for training (default: 256)
--lr FLOAT          # Initial learning rate (default: 0.1)
--momentum FLOAT    # SGD momentum (default: 0.9)
--weight-decay FLOAT # Weight decay (default: 1e-4)

# Model options
--pretrained         # Use pretrained weights for fine-tuning

# System options
--num-workers N      # Number of data loading workers (default: 4)
--save-dir PATH     # Directory to save checkpoints (default: ./checkpoints)

# Convenience modes
--quick             # Quick training mode (10 epochs, batch size 128)
--test              # Test mode (1 epoch, batch size 32)
--info              # Show project information and exit
```

#### **Common Use Cases:**

**1. First-time Setup & Testing:**
```bash
# Check if everything is working
uv run python main.py --info

# Quick test with minimal resources
uv run python main.py --test
```

**2. Educational/Learning:**
```bash
# Quick training for learning
uv run python main.py --quick

# Custom short training
uv run python main.py --epochs 5 --batch-size 64
```

**3. Baseline Training:**
```bash
# Standard training with auto-detected dataset
uv run python main.py

# Full training with specific dataset
uv run python main.py --data-dir /datasets/imagenet --epochs 90
```

**4. Fine-tuning:**
```bash
# Fine-tune pretrained model
uv run python main.py --pretrained --epochs 20 --lr 0.01 --batch-size 128
```

---

## 🎛️ **Pipeline Modes & Commands**

### **ImageNet Pipeline Modes**
| Mode | Purpose | Time | Command |
|------|---------|------|---------|
| `full` | Production training with all 7 steps | ~2-3 days | `--mode full --epochs 90` |
| `quick` | Fast testing with reduced iterations | ~4-6 hours | `--mode quick --epochs 20` |
| `test` | Minimal validation | ~30-60 min | `--mode test --epochs 5` |
| `custom` | Skip LR/WD search for experts | ~1-2 days | `--mode custom --skip-lr-test` |

### **Tiny-ImageNet Training Options**
```bash
# Basic options
--data PATH           # Dataset path (required)
--out DIR            # Output directory
--epochs N           # Number of epochs (default: 20)
--batch-size N       # Batch size (default: 64)

# Advanced options  
--lr-max FLOAT       # Maximum learning rate (default: 0.1)
--lr-min FLOAT       # Minimum learning rate (default: 0.001)
--wd FLOAT          # Weight decay (default: 1e-4)
```

---

## 🔄 **Script Comparison: main.py vs imagenet_training_pipeline.py**

| Aspect | main.py | imagenet_training_pipeline.py |
|--------|---------|-------------------------------|
| **Purpose** | User-friendly training launcher | Advanced 7-step optimization pipeline |
| **Complexity** | Simple, beginner-friendly | Advanced, research-oriented |
| **Automation** | Basic dataset auto-detection | Full hyperparameter optimization |
| **Time to Results** | Minutes to hours | Hours to days |
| **Configuration** | Manual parameter setting | Automated optimization |
| **Use Case** | Quick training, education | Production, research |

### **When to Use main.py:**
✅ **Quick training experiments**  
✅ **Learning and education**  
✅ **Simple baseline training**  
✅ **Testing with known hyperparameters**  
✅ **Fast iteration and debugging**  

### **When to Use imagenet_training_pipeline.py:**
✅ **Production model training**  
✅ **Research and benchmarking**  
✅ **Optimal hyperparameter discovery**  
✅ **Systematic optimization**  
✅ **Publication-quality results**  

### **Technical Differences:**

**main.py Features:**
- Direct training with `train_imagenet.py`
- Manual hyperparameter specification  
- Auto-dataset detection
- Built-in convenience modes (`--quick`, `--test`)
- Immediate training start
- Simple logging and progress tracking

**imagenet_training_pipeline.py Features:**
- 7-step systematic optimization process
- LR Range Test → Batch Size Optimization → Weight Decay Search
- OneCycle LR scheduler with momentum cycling
- Memory-aware batch size detection
- Comprehensive result analysis and visualization
- Professional experiment tracking
- Automated checkpoint management

---

## 🎯 **Key Features**

### **🔬 ImageNet Pipeline Features**
- **Automated LR Range Testing** with smart stopping
- **Memory Optimization** - auto-detects optimal batch size
- **Hyperparameter Search** - grid search for weight decay
- **Advanced Training** - OneCycle + cyclical momentum + label smoothing
- **Real-time Monitoring** - live progress, plots, comprehensive logging
- **Multiple Modes** - full/quick/test/custom for different use cases

### **🎯 Tiny-ImageNet Features**  
- **Single-file Implementation** - simplified from complex multi-file structure
- **Advanced Training** - Mixed precision, gradient accumulation, OneCycleLR
- **Clean Progress Bars** - tqdm with no warning interference
- **Proper Accuracy Calculation** - fixed CIFAR-10 style accuracy tracking
- **Professional Logging** - comprehensive logging with timestamps

### **📊 Core ImageNet Features**
- **ResNet50** optimized for ImageNet-1K (224x224, 1000 classes)
- **UV-based** dependency management
- **Multi-GPU ready** with minimal modifications
- **Comprehensive logging** with automatic file creation

---

## 📊 **Expected Results**

### **ImageNet (ResNet50)**
- **Top-1 Accuracy**: 75-78% (systematic optimization)
- **Training Time**: 24-48 hours (pipeline), 8-16 hours (basic)
- **Model Size**: ~25.6M parameters
- **Optimal Settings**: Auto-discovered and saved

### **Tiny-ImageNet (ResNet50 adapted)**
- **Top-1 Accuracy**: 60-70% (20 epochs)
- **Training Time**: 2-4 hours (depending on GPU)
- **Model Size**: ~23.9M parameters
- **Input Size**: 64x64, 200 classes

---

## 🔧 **Requirements & Setup**

### **System Requirements**
- Python 3.8+
- CUDA-capable GPU (8GB+ recommended)
- 16GB+ RAM
- UV package manager

### **Dataset Requirements**
- **ImageNet-1K**: Full dataset (~150GB) in standard structure
- **Tiny-ImageNet**: Subset dataset (~250MB) in `/tiny_imagenet_training/datasets/`

### **Quick Setup**
```bash
# Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository>
cd TSAI_ERAv4_mini_capstone_S9

# Test pipeline
uv run python test_imagenet_pipeline.py

# Start training
./run_imagenet_pipeline.sh --data /path/to/imagenet --mode test
```

---

## 📚 **Documentation**

- **[🚀 ImageNet Pipeline Guide](IMAGENET_PIPELINE_README.md)** - Complete pipeline documentation
- **[📋 Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Detailed overview of all components
- **[🎯 Tiny-ImageNet Guide](tiny_imagenet_training/README.md)** - Simplified training documentation
- **[🔬 LR Optimization](lr_optimization/README.md)** - Learning rate optimization tools
- **[📊 Dataset Analysis](dataset_analysis/README.md)** - Data exploration and analysis
- **[⚙️ Setup Guide](setup/README.md)** - Complete setup documentation

---

## 🎉 **What's New**

### **✨ Latest Updates (October 2025)**
- **🚀 Complete 7-Step ImageNet Pipeline** - Automated hyperparameter optimization
- **🎯 Simplified Tiny-ImageNet Training** - Single-file implementation
- **🔧 Fixed Accuracy Calculation** - CIFAR-10 style accuracy tracking
- **📊 Enhanced Progress Bars** - Clean tqdm with no warnings
- **⚡ UV Integration** - All commands updated to use `uv run`
- **📝 Comprehensive Documentation** - Updated guides and examples

### **🏗️ Architecture Improvements**
- **Modular Design** - Pipeline components can be used independently
- **Error Handling** - Robust error detection and recovery
- **Memory Management** - Smart GPU memory optimization
- **Professional Logging** - Structured logging throughout

---

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`  
3. Test changes: `uv run python test_imagenet_pipeline.py`
4. Submit pull request

---

## 📄 **License**

MIT License - see project files for details.

---

**TSAI ERAv4 - Building the future of AI education with systematic optimization** 🚀✨
