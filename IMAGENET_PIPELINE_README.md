# ImageNet Training Pipeline Configuration

## Pipeline Overview
This pipeline implements the 7-step systematic approach for ImageNet training:

1. **LR Range Test** → Find optimal learning rate bounds
2. **Pick LR bounds** → Extract min/max LR from range test
3. **Set OneCycle LR + cyclical momentum** → Configure scheduler
4. **Choose batch size** → Auto-detect optimal batch size
5. **Tune weight-decay & regularizers** → Grid search for best WD
6. **Full OneCycle training** → Complete training run
7. **Monitor & iterate** → Analysis and plotting

## Quick Start

### 1. Full Pipeline (Recommended)
```bash
uv run python imagenet_training_pipeline.py \
  --data /path/to/imagenet \
  --output ./results \
  --epochs 90
```

### 2. Quick Mode (For Testing)
```bash
uv run python imagenet_training_pipeline.py \
  --data /path/to/imagenet \
  --output ./results_quick \
  --quick-mode \
  --epochs 20
```

### 3. Custom Configuration
```bash
uv run python imagenet_training_pipeline.py \
  --data /path/to/imagenet \
  --output ./results_custom \
  --batch-size 256 \
  --epochs 50 \
  --skip-lr-test \
  --skip-wd-search
```

## Pipeline Steps Explained

### Step 1: LR Range Test
- Tests learning rates from 1e-7 to 1.0
- Finds steepest loss decline
- Suggests min/max LR for OneCycle
- Saves plot and configuration

### Step 2-3: OneCycle Setup
- Uses suggested LR bounds
- Configures cyclical momentum (0.85 ↔ 0.95)
- Sets up cosine annealing schedule
- 30% warmup, 70% annealing

### Step 4: Batch Size Optimization
- Auto-detects maximum GPU memory batch size
- Uses 75% of max for stability
- Rounds to nearest power of 2
- Reloads data with optimal batch size

### Step 5: Weight Decay Search
- Tests multiple WD values: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- Quick training (3-5 epochs) for each
- Selects best based on validation accuracy
- Saves detailed results

### Step 6: Full Training
- Uses all optimized hyperparameters
- OneCycle LR + cyclical momentum
- Label smoothing (0.1)
- Gradient clipping (max_norm=1.0)
- Early stopping with patience
- Checkpoint saving

### Step 7: Analysis & Monitoring
- Plots training/validation curves
- Learning rate and momentum schedules
- Saves comprehensive results
- Provides final metrics summary

## Output Files

After completion, the output directory will contain:

```
results/
├── lr_range_test.png          # LR range test plot
├── lr_config.json             # Suggested LR configuration
├── weight_decay_search.json   # WD search results
├── training_results.png       # Training curves and schedules
├── final_results.json         # Complete results summary
├── best_model.pth            # Best model checkpoint
└── pipeline.log              # Detailed logging
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to ImageNet dataset |
| `--output` | `./imagenet_pipeline_results` | Output directory |
| `--batch-size` | Auto-detect | Fixed batch size (skips detection) |
| `--epochs` | 90 | Number of epochs for full training |
| `--skip-lr-test` | False | Skip LR range test (use defaults) |
| `--skip-wd-search` | False | Skip weight decay search |
| `--quick-mode` | False | Fast mode with fewer iterations |

## Expected Results

For ImageNet-1K with ResNet50:
- **Top-1 Validation Accuracy**: 75-78%
- **Training Time**: ~24-48 hours (depending on hardware)
- **Memory Usage**: 8-16GB GPU memory
- **Batch Size**: 256-512 (depending on GPU)

## Hardware Requirements

- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 16GB+ GPU memory, 32GB+ RAM
- **Optimal**: Multiple GPUs with DataParallel support

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use `--batch-size 128`
2. **Slow Training**: Use `--quick-mode` for testing
3. **Dataset Not Found**: Ensure ImageNet path is correct
4. **Low Accuracy**: Try different weight decay values manually

### Performance Tips:

1. Use `num_workers=8` for data loading
2. Enable mixed precision training (TODO: add AMP support)
3. Use multiple GPUs with DataParallel
4. Monitor GPU utilization with `nvidia-smi`

## Advanced Usage

### Custom LR Configuration
If you have specific LR requirements, skip the range test:
```bash
uv run python imagenet_training_pipeline.py \
  --data /path/to/imagenet \
  --skip-lr-test \
  --output ./results
```

Then manually edit `lr_config.json`:
```json
{
  "min_lr": 0.001,
  "max_lr": 0.1,
  "steepest_decline_lr": 0.01,
  "min_loss_lr": 0.005
}
```

### Integration with Existing Code
The pipeline can be imported and used programmatically:

```python
from imagenet_training_pipeline import LRFinder, FullTrainer

# Create model and data loaders
model = create_model()
train_loader, val_loader = get_data()

# Run LR range test
lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.range_test(train_loader)
lr_config = lr_finder.suggest_lr()

# Full training
trainer = FullTrainer(model, train_loader, val_loader, device, save_dir)
history = trainer.train(lr_config, epochs=90)
```