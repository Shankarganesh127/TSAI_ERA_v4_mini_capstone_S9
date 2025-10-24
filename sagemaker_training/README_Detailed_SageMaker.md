# SageMaker Training Directory - Detailed Step-by-Step Guide

## 1. Directory Structure & File Roles

```
sagemaker_training/
├── configs/                  # Pipeline and config examples
│   ├── pipeline_config.json  # Main pipeline config
│   └── config_examples.json  # Example configs
├── documentation/           # Extended documentation
│   └── README.md            # Full pipeline and file explanations
├── ilsvrc_dataset.py        # ImageNet data loader for SageMaker
├── imagenet_dataset.py      # Dataset utilities for ImageNet
├── imagenet_models.py       # Model architectures (ResNet, etc.)
├── imagenet_training_pipeline.py # Main training pipeline (7 stages)
├── launch_sagemaker.py      # CLI entry for launching SageMaker jobs
├── logger_setup.py          # Logging configuration
├── MAIN_ENTRY_POINTS.md     # Guide to main entry scripts
├── model_saver.py           # Automated model saving/replacement
├── monitor_training.py      # Real-time training monitor
├── outputs/                 # Output folder for logs, models
├── README.md                # Quick start and features
├── requirements.txt         # Python dependencies
├── s3_dataset_converter.py  # Convert ILSVRC to SageMaker format
├── sagemaker_logging.py     # Logging integration for SageMaker
├── sagemaker_orchestrator.py# Full pipeline orchestrator
├── sagemaker_wrapper.py     # Main SageMaker wrapper for jobs
├── scripts/                 # Setup scripts
│   └── setup_environment.py # Environment setup
├── training_integration.py  # Model replacement and monitoring
├── training_optimizer_example.py # Example optimizer usage
├── TRAINING_OPTIMIZER_README.md # Optimizer documentation
├── training_performance_optimizer.py # Performance optimizer
├── train_imagenet.py        # CLI for local training
├── utils.py                 # Utility functions
└── __pycache__/             # Compiled Python files
```

## 2. Step-by-Step Pipeline Flow

### Stage 1: Data Preparation
- **ilsvrc_dataset.py, imagenet_dataset.py, s3_dataset_converter.py**
- Loads, preprocesses, and converts ImageNet data for SageMaker.
- Ensures efficient data loading and format compatibility.

### Stage 2: Model Definition
- **imagenet_models.py**
- Defines deep learning architectures (ResNet, etc.)
- Supports model replacement and custom architectures.

### Stage 3: Environment & Logging Setup
- **setup_environment.py, logger_setup.py, sagemaker_logging.py**
- Validates environment, configures logging for real-time monitoring.

### Stage 4: Training Pipeline
- **imagenet_training_pipeline.py, sagemaker_wrapper.py, training_integration.py**
- Implements 7-stage training: memory management, AMP, gradient accumulation, checkpointing, validation, and model saving.
- Handles distributed/multi-GPU training, spot instance support.

### Stage 5: Optimization & Monitoring
- **training_performance_optimizer.py, training_optimizer_example.py, monitor_training.py**
- Optimizes batch size, learning rate, data loading, and GPU utilization.
- Monitors training progress and performance in real time.

### Stage 6: Orchestration & Automation
- **sagemaker_orchestrator.py, launch_sagemaker.py**
- Automates the entire pipeline, including job launch, model replacement, and output management.
- Handles configuration via `configs/pipeline_config.json`.

### Stage 7: Output & Analysis
- **model_saver.py, outputs/**
- Saves models (current/best), logs, and training summaries.
- Supports automatic model replacement every epoch.

## 3. How Each Stage Improves Accuracy
- **Data Preparation**: Ensures clean, well-augmented data for robust training.
- **Model Definition**: Allows selection/tuning of architectures for best accuracy.
- **Environment Setup**: Guarantees reproducibility and proper resource allocation.
- **Training Pipeline**: Implements best practices (AMP, gradient accumulation, checkpointing) for stable, accurate training.
- **Optimization**: Maximizes GPU utilization, prevents bottlenecks, tunes hyperparameters.
- **Orchestration**: Automates error handling, model management, and experiment tracking.
- **Output/Analysis**: Enables selection of best models and detailed result analysis.

## 4. SageMaker Usage & Advantages
- **Scalability**: Train on multi-GPU, multi-node clusters with minimal setup.
- **Managed Infrastructure**: Handles hardware, storage, and logging automatically.
- **Spot Instances**: Reduces cost for large-scale experiments.
- **Integration**: Directly runs pipeline scripts with full automation.
- **Monitoring**: Real-time logs and progress tracking.

## 5. Problems Faced & Solutions
- **OOM Errors**: Solved by adaptive batch sizing, gradient accumulation, mixed precision.
- **Multi-GPU Sync Issues**: Fixed with correct batch size calculation and DDP support.
- **Subprocess/Scoping Errors**: Addressed by restructuring code and robust logging.
- **Memory Fragmentation**: Aggressive cleanup and checkpointing added.
- **Data Loading Bottlenecks**: Optimized with worker scaling and prefetching.

## 6. Usage Example & Commands

### Launch Full Pipeline (Recommended)
```sh
python sagemaker_orchestrator.py --config configs/pipeline_config.json
```

### Launch Single Job
```sh
python launch_sagemaker.py --role-arn <role> --bucket <s3-bucket> --epochs 30
```

### Convert Dataset
```sh
python s3_dataset_converter.py --bucket <s3-bucket>
```

### Monitor Training
```sh
python monitor_training.py --job-name <job>
```

## 7. Additional Notes
- **Extensive Documentation**: See `documentation/README.md` for full details.
- **Modular Design**: Each stage is a separate module for easy extension.
- **Best Practices**: Implements modern deep learning and SageMaker best practices.
- **Configurable**: All major parameters can be set via config files or CLI.
- **Error Handling**: Robust logging and error management throughout.

---
For further details, see the documentation folder and comments in each script.
