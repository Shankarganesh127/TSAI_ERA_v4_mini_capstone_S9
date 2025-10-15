# TSAI ERAv4 Mini Capstone S9

ImageNet-1K training with ResNet50 using modern Python tooling.

## 🚀 Quick Start with UV

### One-Command Setup
```bash
cd setup
python setup_uv.py
```

### Start Training (Recommended Method)
```bash
# Auto-detect dataset and start training
python main.py

# Or specify dataset location
python main.py --data-dir /path/to/imagenet

# Quick training (10 epochs)
python main.py --quick

# Test mode (1 epoch, small batch)
python main.py --test
```

### Manual Setup
```bash
# Install UV (if not already installed)
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv venv --python 3.11
uv pip install -e .

# Activate environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Test setup
cd setup
python test_setup.py

# Start training (requires ImageNet dataset)
python train_imagenet.py --data-dir /path/to/imagenet

# OR use the main.py entry point (recommended)
python main.py --data-dir /path/to/imagenet

# Quick start with main.py (auto-detects dataset)
python main.py

# Quick training mode (10 epochs, smaller batch)
python main.py --quick

# Test mode (1 epoch, tiny batch)
python main.py --test

# Show project information
python main.py --info
```

## 📁 Project Structure

```
TSAI_ERAv4_mini_capstone_S9/
├── main.py                   # 🚀 MAIN ENTRY POINT - Use this to start training
├── imagenet_models.py         # ResNet50 for ImageNet-1K
├── imagenet_dataset.py        # Dataset loader with transforms
├── train_imagenet.py          # Core training script (called by main.py)
├── utils.py                  # Training utilities
├── logger_setup.py           # Comprehensive logging system
├── pyproject.toml           # Dependencies and project config
├── logs/                    # Automatic log files (script_name_timestamp.log)
├── setup/                   # Setup and configuration tools
│   ├── setup_uv.py          # Automated UV environment setup
│   ├── test_setup.py        # Setup verification
│   ├── quick_setup.py       # Dataset configuration
│   └── README.md           # Setup documentation
├── dataset_tools/            # Dataset download and management
│   ├── imagenet_subset_downloader.ipynb  # Interactive dataset downloader
│   └── README.md            # Dataset tools documentation
├── lr_optimization/          # Learning rate optimization tools
│   ├── learning_rate_finder.ipynb        # LR Range Test implementation
│   └── README.md            # LR optimization documentation
├── data_exploration/         # Dataset visualization tools
│   ├── imagenet_dataset_explorer.ipynb   # Interactive dataset explorer
│   └── README.md            # Data exploration documentation
├── docs/                    # Project documentation
│   ├── CLEANUP_SUMMARY.md   # Project cleanup history
│   ├── project_requirements.md  # Project specifications
│   ├── EC2_instance.md      # AWS EC2 setup guide
│   ├── learning_rate_finder.md  # LR optimization docs
│   └── README.md            # Documentation index
└── README.md               # Main project documentation
```

## 🎯 Features

- **ResNet50** optimized for ImageNet-1K (224x224, 1000 classes)
- **UV-based** dependency management for fast, reliable setup
- **Simple training loop** with standard ImageNet augmentations
- **Pretrained weights** support for transfer learning
- **Comprehensive logging** system with automatic log file creation
- **Professional logging** - All print statements replaced with proper logging
- **Multi-GPU** ready (with minimal modifications)

## 🔧 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- ImageNet-1K dataset
- UV package manager

## 📝 Logging System

The project uses a comprehensive logging system that automatically:
- Creates a `logs/` directory for all log files
- Names log files as `{script_name}_{timestamp}.log`
- Logs to both console (INFO level) and file (DEBUG level)
- Includes system information, training configuration, and progress
- Provides detailed timestamps and structured formatting

**Example log file**: `logs/train_imagenet_20251014_225009.log`

All print statements have been replaced with proper logging for professional development.

## 📊 Expected Results

With full ImageNet-1K training:
- **Top-1 Accuracy**: ~76% (90 epochs from scratch)
- **Training Time**: ~24-48 hours on modern GPU
- **Model Size**: ~25.6M parameters

## 📚 Documentation

- [Project Documentation](docs/README.md) - Complete documentation index
- [Setup Guide](setup/README.md) - Complete setup and UV documentation
- [Dataset Tools](dataset_tools/README.md) - ImageNet dataset download and management
- [LR Optimization](lr_optimization/README.md) - Learning rate finding tools
- [Data Exploration](data_exploration/README.md) - Dataset visualization and analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `cd setup && python test_setup.py`
4. Submit pull request

## 📄 License

MIT License - see project files for details.

---

**TSAI ERAv4 - Building the future of AI education** 🚀