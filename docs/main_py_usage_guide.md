# Main.py Usage Guide

The `main.py` script serves as the main entry point for the TSAI ERAv4 Mini Capstone S9 ImageNet training project. It provides a user-friendly interface with automatic dataset detection, comprehensive logging, and convenient training modes.

## 🚀 Quick Start

### Basic Usage
```bash
# Auto-detect dataset and start training
python main.py

# Show project information
python main.py --info

# Get help with all options
python main.py --help
```

### Training Modes

#### 1. Quick Training Mode
```bash
python main.py --quick
```
- **Epochs**: 10 (instead of default 90)
- **Batch size**: 128 (instead of default 256)
- **Purpose**: Fast testing and development

#### 2. Test Mode
```bash
python main.py --test
```
- **Epochs**: 1
- **Batch size**: 32
- **Purpose**: Quick validation that everything works

#### 3. Custom Training
```bash
python main.py --epochs 50 --batch-size 128 --lr 0.05
```

#### 4. Fine-tuning Mode
```bash
python main.py --pretrained --epochs 20 --lr 0.01
```

## 📁 Dataset Auto-Detection

The `main.py` script automatically searches for ImageNet datasets in common locations:

- `./imagenet`
- `./datasets/imagenet`
- `./data/imagenet`
- `../imagenet`
- `../datasets/imagenet`
- `C:/datasets/imagenet`
- `D:/datasets/imagenet`

### Manual Dataset Specification
```bash
python main.py --data-dir /path/to/your/imagenet
```

## 📋 Command Line Options

### Essential Options
- `--data-dir PATH`: Specify dataset location
- `--epochs N`: Number of training epochs (default: 90)
- `--batch-size N`: Batch size (default: 256)
- `--lr FLOAT`: Learning rate (default: 0.1)

### Model Options
- `--pretrained`: Use pretrained weights for fine-tuning
- `--save-dir PATH`: Directory for checkpoints (default: ./checkpoints)

### System Options
- `--num-workers N`: Data loading workers (default: 4)

### Convenience Modes
- `--quick`: Quick training (10 epochs, batch 128)
- `--test`: Test mode (1 epoch, batch 32)
- `--info`: Show project info and exit

## 🔧 Dependency Handling

The script intelligently handles missing dependencies:

- **If dependencies are available**: Launches training normally
- **If dependencies are missing**: Shows helpful error message with setup instructions
- **Info mode**: Always works regardless of dependencies

### Setup Dependencies
```bash
# Method 1: Use setup script
cd setup
python setup_uv.py

# Method 2: Manual installation
uv pip install torch torchvision

# Method 3: Standard pip
pip install torch torchvision
```

## 📊 Logging

Every run of `main.py` creates detailed log files:

- **Log location**: `logs/main_YYYYMMDD_HHMMSS.log`
- **Content**: System info, configuration, progress, errors
- **Format**: Timestamped, structured logging

### Example Log Output
```
2025-10-14 23:00:27 - main - INFO - Starting TSAI ERAv4 Mini Capstone S9 - ImageNet Training
2025-10-14 23:00:27 - main - INFO - Quick training mode enabled
2025-10-14 23:00:27 - main - INFO - Using auto-detected dataset: ./imagenet
2025-10-14 23:00:27 - main - INFO - Training Configuration:
2025-10-14 23:00:27 - main - INFO -   Dataset: ./imagenet
2025-10-14 23:00:27 - main - INFO -   Epochs: 10
2025-10-14 23:00:27 - main - INFO -   Batch size: 128
```

## 💡 Examples

### Development Workflow
```bash
# 1. Check project info
python main.py --info

# 2. Test that everything works
python main.py --test

# 3. Quick training run
python main.py --quick

# 4. Full training
python main.py
```

### Fine-tuning Workflow
```bash
# Start with pretrained weights
python main.py --pretrained --epochs 20 --lr 0.01 --batch-size 128
```

### CPU Testing (No GPU)
```bash
# Small batch size for CPU
python main.py --test --batch-size 16
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
python main.py
# Error: Training dependencies not available!
# Solution: cd setup && python setup_uv.py
```

#### 2. No Dataset Found
```bash
python main.py
# Error: No dataset found!
# Solution: Use dataset_tools/imagenet_subset_downloader.ipynb
```

#### 3. Invalid Dataset Structure
```bash
python main.py --data-dir /wrong/path
# Error: Invalid dataset structure
# Solution: Ensure dataset/train/ and dataset/val/ exist
```

## 🏆 Best Practices

1. **Start with info**: `python main.py --info`
2. **Test first**: `python main.py --test`
3. **Quick development**: `python main.py --quick`
4. **Monitor logs**: Check `logs/` directory
5. **Use checkpoints**: Results saved in `./checkpoints/`

---

**TSAI ERAv4 Mini Capstone S9** - Professional ImageNet training made simple! 🚀