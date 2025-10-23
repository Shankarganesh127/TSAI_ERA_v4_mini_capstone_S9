#!/usr/bin/env python3
"""
Usage Example: Training Performance Optimizer for SageMaker

This example shows how to integrate the TrainingPerformanceOptimizer
into your existing ImageNet training pipeline to achieve the target
epoch times of 20-40 minutes instead of 6+ hours.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from training_performance_optimizer import TrainingPerformanceOptimizer, create_optimized_trainer
from imagenet_dataset import get_imagenet_dataloaders
from imagenet_models import resnet50_imagenet


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_with_optimizer(rank: int, world_size: int, args):
    """
    Training function with performance optimization.

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Training arguments
    """
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)

    try:
        # Set device
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(rank)

        # Create model
        model = resnet50_imagenet(num_classes=1000)
        model = model.to(device)

        # Create data loaders
        train_loader, val_loader = get_imagenet_dataloaders(
            train=args.train_data,
            val=args.val_data,
            batch_size=args.batch_size,
            num_workers=4  # Will be optimized later
        )

        # Create optimizer and loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Create performance optimizer
        optimizer_trainer = create_optimized_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            world_size=world_size,
            rank=rank,
            enable_amp=True,  # Enable mixed precision
            enable_profiling=True
        )

        # Optimize data loading for high GPU utilization
        print(f"🔧 Optimizing data loading for {world_size} GPUs...")
        train_loader = optimizer_trainer.optimize_data_loading(target_workers=32)

        # Find optimal batch size
        print("🔍 Finding optimal batch size...")
        optimal_batch_size = optimizer_trainer.get_optimal_batch_size(max_memory_gb=14.0)

        # Scale learning rate for larger batch size
        if optimal_batch_size != args.batch_size:
            new_lr = optimizer_trainer.scale_learning_rate_for_batch_size(
                base_lr=args.lr,
                base_batch_size=args.batch_size,
                current_batch_size=optimal_batch_size
            )

            # Update optimizer with new LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Create warmup scheduler
        warmup_scheduler = optimizer_trainer.create_warmup_scheduler(
            optimizer=optimizer,
            warmup_epochs=5,
            total_epochs=args.epochs,
            base_lr=1e-7,
            max_lr=new_lr
        )

        print("🚀 Starting optimized training...")

        # Training loop
        for epoch in range(args.epochs):
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{args.epochs}")

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                metrics = optimizer_trainer.optimize_training_step(batch, 'train')
                train_loss += metrics['loss']

                if batch_idx % 100 == 0 and rank == 0:
                    print(f"  Batch {batch_idx}: Loss = {metrics['loss']:.4f}")

            # Update learning rate
            warmup_scheduler.step()

            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        metrics = optimizer_trainer.optimize_training_step(batch, 'val')
                        val_loss += metrics['loss']

                val_loss /= len(val_loader)

            # Print epoch summary
            if rank == 0:
                train_loss /= len(train_loader)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
                if val_loader is not None:
                    print(f"              Val Loss = {val_loss:.4f}")

                # Print performance report every 5 epochs
                if (epoch + 1) % 5 == 0:
                    report = optimizer_trainer.get_performance_report()
                    print("\n" + "="*50)
                    print("PERFORMANCE REPORT")
                    print("="*50)
                    for rec in report['recommendations']:
                        print(f"• {rec}")
                    print("="*50 + "\n")

        # Final performance summary
        if rank == 0:
            summary = optimizer_trainer.get_optimization_summary()
            print("\n" + summary)

    finally:
        if world_size > 1:
            cleanup_distributed()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimized ImageNet Training')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str,
                       help='Path to validation data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Initial batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of training epochs')
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of GPUs/processes')

    args = parser.parse_args()

    # Multi-GPU training
    if args.world_size > 1:
        mp.spawn(train_with_optimizer,
                args=(args.world_size, args),
                nprocs=args.world_size,
                join=True)
    else:
        train_with_optimizer(0, 1, args)


if __name__ == '__main__':
    main()