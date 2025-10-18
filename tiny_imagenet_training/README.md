# Tiny-ImageNet Training - Simplified

A simplified, single-file implementation for training ResNet50 on Tiny-ImageNet with configurable hyperparameters and live progress bars.

## Structure

```
tiny_imagenet_training/
├── train.py           # Complete training implementation (single file)
├── pyproject.toml     # Package configuration
├── README.md          # This file
├── datasets/          # Dataset folder (optional symlink)
├── logs/              # Training logs
└── runs_tiny/         # Training outputs
```

## Features

- **Single file**: All training logic, configuration, and CLI in `train.py`
- **Hardware auto-detection**: Automatically suggests batch size and accumulation steps
- **LR recommendations**: Loads previous LR optimization results if available
- **Live progress bars**: tqdm-based progress bars for training and validation per epoch
- **Repository integration**: Uses models from `imagenet_models.py` and datasets from `imagenet_dataset.py`
- **Comprehensive logging**: Per-run logs with system info, model details, and metrics
- **Percent formatting**: Displays accuracies, loss, and LR with intuitive percent views

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run training:**
   ```bash
   # Using console script
   uv run tiny-imagenet-train --data ./datasets/tiny-imagenet-200 --epochs 10

   # Or directly
   uv run python train.py --data ./datasets/tiny-imagenet-200 --epochs 10
   ```

3. **Available options:**
   ```bash
   uv run tiny-imagenet-train --help
   ```

## Dataset Setup

If you don't have the dataset symlinked locally, you can:

1. Download Tiny-ImageNet using the dataset downloader notebook
2. Create a symlink: `ln -s /path/to/tiny-imagenet-200 ./datasets/tiny-imagenet-200`

## Configuration

The training automatically:
- Detects GPU memory and suggests optimal batch size
- Loads LR/weight decay recommendations from `../lr_optimization/results/onecycle_summary.json`
- Uses OneCycleLR scheduling with cosine annealing
- Applies label smoothing, gradient clipping, and mixed precision

## Output

Each run creates a timestamped directory in `runs_tiny/` containing:
- `config.json` - Full configuration used
- `train.log` - Detailed per-batch and per-epoch logs
- `history.json` - Complete training history with metrics
- `final_metrics.json` - Final results with percent views
- `best.pth` - Best model checkpoint (by validation accuracy)
- `last.pth` - Final model state

## Example Output

```
Epoch 1/50:  64%|██████▍   | 1000/1563 [02:15<01:15,  7.34it/s, Loss=4.123, Acc=12.34%]
Validating: 100%|██████████| 157/157 [00:08<00:00, 19.23it/s]
Train Loss: 4.1234, Train Acc: 12.34%
Val Loss: 4.5678, Val Acc: 8.76%
New best model saved! Accuracy: 8.76%
```

## Simplified from Previous Structure

This version consolidates the previous multi-file package into a single `train.py` file while maintaining all functionality:
- Removed `src/` layout complexity
- Merged config, models, datasets, and trainer modules
- Simplified packaging with minimal `pyproject.toml`
- Retained all training features and progress bars
