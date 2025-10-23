#!/usr/bin/env python3
"""
Training Performance Optimizer for SageMaker ImageNet Training

Addresses the critical bottlenecks causing slow training (6+ hours per epoch):
1. GPU underutilization due to I/O bottlenecks
2. Inefficient data loading pipelines
3. Lack of mixed precision training
4. Suboptimal distributed training setup
5. Improper batch size and learning rate scaling

This class provides automated optimization and monitoring to achieve
target epoch times of 20-40 minutes instead of 6+ hours.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import psutil
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_UTIL_AVAILABLE = False
from contextlib import contextmanager
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingPerformanceOptimizer:
    """
    Comprehensive training optimizer for ImageNet-scale distributed training.

    Addresses the key bottlenecks:
    - GPU utilization monitoring and optimization
    - Data loading pipeline optimization
    - Mixed precision training setup
    - Distributed training efficiency
    - Batch size and learning rate scaling
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 world_size: int = 1,
                 rank: int = 0,
                 enable_amp: bool = True,
                 enable_profiling: bool = True):
        """
        Initialize the training optimizer.

        Args:
            model: PyTorch model
            optimizer: Optimizer (will be wrapped for distributed training)
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use ('cuda' or 'cpu')
            world_size: Number of processes/GPUs
            rank: Process rank
            enable_amp: Enable automatic mixed precision
            enable_profiling: Enable performance profiling
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.enable_profiling = enable_profiling

        # Performance tracking
        self.performance_stats = {
            'gpu_utilization': [],
            'data_loading_times': [],
            'forward_pass_times': [],
            'backward_pass_times': [],
            'step_times': [],
            'memory_usage': []
        }

        # Setup mixed precision
        self.scaler = GradScaler() if self.enable_amp else None

        # Setup distributed training
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self._setup_distributed_training()

        logger.info(f"🚀 Training Optimizer initialized:")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - World Size: {world_size}")
        logger.info(f"   - Rank: {rank}")
        logger.info(f"   - AMP: {self.enable_amp}")
        logger.info(f"   - Distributed: {self.is_distributed}")
        logger.info(f"   - Profiling: {enable_profiling}")

    def _setup_distributed_training(self):
        """Setup DistributedDataParallel training."""
        # Find unused parameters for efficiency
        self.model = DDP(self.model, device_ids=[self.rank],
                        output_device=self.rank,
                        find_unused_parameters=False)

        logger.info(f"✅ Distributed training setup complete for rank {self.rank}")

    def optimize_data_loading(self, target_workers: int = 32) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Optimize data loading for maximum GPU utilization.

        Args:
            target_workers: Target number of DataLoader workers

        Returns:
            Tuple of (optimized_train_loader, optimized_val_loader)
        """
        logger.info(f"🔧 Optimizing data loading pipelines with {target_workers} workers")

        # Optimize training loader
        train_current_workers = self.train_loader.num_workers
        logger.info(f"   Training loader: {train_current_workers} → {target_workers} workers")

        # Check if shuffle is enabled by examining the sampler type
        from torch.utils.data import RandomSampler
        train_shuffle = isinstance(self.train_loader.sampler, RandomSampler)

        optimized_train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=train_shuffle,
            sampler=self.train_loader.sampler,
            num_workers=target_workers,
            pin_memory=self.train_loader.pin_memory,
            drop_last=self.train_loader.drop_last,
            prefetch_factor=2,  # Prefetch 2 batches per worker
            persistent_workers=True  # Keep workers alive between epochs
        )

        # Optimize validation loader if available
        optimized_val_loader = None
        if self.val_loader is not None:
            val_current_workers = self.val_loader.num_workers
            logger.info(f"   Validation loader: {val_current_workers} → {min(target_workers, 8)} workers")

            # Check if shuffle is enabled by examining the sampler type
            val_shuffle = isinstance(self.val_loader.sampler, RandomSampler)

            optimized_val_loader = DataLoader(
                self.val_loader.dataset,
                batch_size=self.val_loader.batch_size,
                shuffle=val_shuffle,
                sampler=self.val_loader.sampler,
                num_workers=min(target_workers, 8),  # Use fewer workers for validation (typically smaller)
                pin_memory=self.val_loader.pin_memory,
                drop_last=self.val_loader.drop_last,
                prefetch_factor=2,
                persistent_workers=True
            )

        logger.info("✅ Data loading optimization complete")
        return optimized_train_loader, optimized_val_loader

    def get_optimal_batch_size(self, max_memory_gb: float = 14.0) -> int:
        """
        Find optimal batch size that maximizes GPU memory utilization.

        Args:
            max_memory_gb: Maximum GPU memory to use (GB)

        Returns:
            Optimal batch size per GPU
        """
        base_batch_size = 32
        max_batch_size = 1024

        logger.info(f"🔍 Finding optimal batch size (max {max_memory_gb}GB GPU memory)")

        # Test different batch sizes
        optimal_batch = base_batch_size
        for batch_size in [32, 64, 128, 256, 512, 1024]:
            try:
                # Test memory usage with dummy batch
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                dummy_target = torch.randint(0, 1000, (batch_size,)).to(self.device)

                with autocast(enabled=self.enable_amp):
                    output = self.model(dummy_input)
                    loss = self.criterion(output, dummy_target)
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Check memory usage
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                if memory_gb > max_memory_gb:
                    break

                optimal_batch = batch_size
                logger.info(f"   ✅ Batch size {batch_size}: {memory_gb:.1f}GB")

                # Clear gradients and cache
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            except RuntimeError as e:
                logger.warning(f"   ❌ Batch size {batch_size} failed: {e}")
                break

        logger.info(f"🎯 Optimal batch size: {optimal_batch}")
        return optimal_batch

    def scale_learning_rate_for_batch_size(self,
                                         base_lr: float,
                                         base_batch_size: int,
                                         current_batch_size: int,
                                         scaling_factor: float = 1.0) -> float:
        """
        Scale learning rate based on batch size changes.

        Args:
            base_lr: Original learning rate
            base_batch_size: Original batch size
            current_batch_size: New batch size
            scaling_factor: Additional scaling factor

        Returns:
            Scaled learning rate
        """
        # Linear scaling rule: LR_new = LR_old * (batch_new / batch_old)
        batch_ratio = current_batch_size / base_batch_size
        scaled_lr = base_lr * batch_ratio * scaling_factor

        logger.info(f"📈 LR scaling: {base_lr:.2e} → {scaled_lr:.2e} "
                   f"(batch: {base_batch_size} → {current_batch_size})")

        return scaled_lr

    @contextmanager
    def profile_step(self, step_name: str):
        """Context manager for profiling training steps."""
        if not self.enable_profiling:
            yield
            return

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            self.performance_stats[f'{step_name}_times'].append(duration)
            if 'memory' in step_name.lower():
                self.performance_stats['memory_usage'].append(memory_delta)

    def monitor_gpu_utilization(self) -> Dict[str, float]:
        """
        Monitor current GPU utilization and system stats.

        Returns:
            Dictionary with utilization metrics
        """
        gpu_stats = {}

        # CPU stats (always available)
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            gpu_stats['cpu_util'] = cpu_percent
        except Exception as e:
            logger.warning(f"Failed to get CPU stats: {e}")

        # GPU stats (optional)
        if GPU_UTIL_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_stats[f'gpu_{i}_util'] = gpu.load * 100
                    gpu_stats[f'gpu_{i}_memory'] = gpu.memoryUsed / gpu.memoryTotal * 100
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
        else:
            logger.debug("GPUtil not available - GPU monitoring disabled")

        return gpu_stats

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with performance metrics and recommendations
        """
        gpu_stats = self.monitor_gpu_utilization()

        report = {
            'gpu_utilization': gpu_stats,
            'performance_stats': self.performance_stats,
            'recommendations': []
        }

        # Analyze GPU utilization
        gpu_utils = [v for k, v in gpu_stats.items() if 'util' in k and 'gpu' in k]
        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            if avg_gpu_util < 80:
                report['recommendations'].append(
                    f"⚠️ Low GPU utilization ({avg_gpu_util:.1f}%). "
                    "Check data loading bottlenecks."
                )
            else:
                report['recommendations'].append(
                    f"✅ Good GPU utilization ({avg_gpu_util:.1f}%)"
                )

        # Analyze data loading times
        if self.performance_stats['data_loading_times']:
            avg_data_time = sum(self.performance_stats['data_loading_times']) / len(self.performance_stats['data_loading_times'])
            report['avg_data_loading_time'] = avg_data_time

            if avg_data_time > 0.1:  # More than 100ms per batch
                report['recommendations'].append(
                    f"🐌 Slow data loading ({avg_data_time:.3f}s/batch). "
                    "Consider faster storage or format conversion."
                )

        return report

    def create_warmup_scheduler(self,
                               optimizer: optim.Optimizer,
                               warmup_epochs: int,
                               total_epochs: int,
                               base_lr: float,
                               max_lr: float) -> optim.lr_scheduler.LambdaLR:
        """
        Create learning rate warmup scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            base_lr: Starting learning rate
            max_lr: Maximum learning rate

        Returns:
            Learning rate scheduler
        """

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return base_lr + (max_lr - base_lr) * (epoch / warmup_epochs)
            else:
                # Cosine annealing or other schedule
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return max_lr * (1 + torch.cos(torch.pi * progress)) / 2

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"🔥 Warmup scheduler created: {warmup_epochs} warmup epochs, "
                   f"LR: {base_lr:.2e} → {max_lr:.2e}")

        return scheduler

    def optimize_training_step(self, batch, step_type: str = 'train') -> Dict[str, float]:
        """
        Optimized training step with AMP and performance monitoring.

        Args:
            batch: Input batch (inputs, targets)
            step_type: 'train' or 'val'

        Returns:
            Dictionary with step metrics
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        metrics = {}

        with self.profile_step(f'{step_type}_step'):
            # Forward pass
            with self.profile_step('forward_pass'):
                with autocast(enabled=self.enable_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            metrics['loss'] = loss.item()

            # Backward pass (only for training)
            if step_type == 'train':
                with self.profile_step('backward_pass'):
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                self.optimizer.zero_grad()

        # Monitor GPU utilization periodically
        if len(self.performance_stats['gpu_utilization']) % 10 == 0:
            gpu_stats = self.monitor_gpu_utilization()
            self.performance_stats['gpu_utilization'].append(gpu_stats)

        return metrics

    def get_optimization_summary(self) -> str:
        """
        Generate optimization summary and recommendations.

        Returns:
            Formatted summary string
        """
        report = self.get_performance_report()

        summary = "🚀 Training Performance Optimization Summary\n"
        summary += "=" * 50 + "\n\n"

        # GPU utilization
        gpu_stats = report['gpu_utilization']
        if gpu_stats:
            summary += "GPU Utilization:\n"
            for key, value in gpu_stats.items():
                if 'util' in key:
                    summary += f"  {key}: {value:.1f}%\n"
            summary += "\n"

        # Performance stats
        stats = report['performance_stats']
        if stats.get('data_loading_times'):
            avg_data_time = sum(stats['data_loading_times']) / len(stats['data_loading_times'])
            summary += f"Average data loading time: {avg_data_time:.3f}s\n"

        if stats.get('step_times'):
            avg_step_time = sum(stats['step_times']) / len(stats['step_times'])
            summary += f"Average step time: {avg_step_time:.3f}s\n"

        # Recommendations
        if report['recommendations']:
            summary += "\nRecommendations:\n"
            for rec in report['recommendations']:
                summary += f"  • {rec}\n"

        # Target improvements
        summary += "\n🎯 Target Improvements:\n"
        summary += "  • Epoch time: 6 hours → 20-40 minutes\n"
        summary += "  • GPU utilization: >90%\n"
        summary += "  • Data loading: <50ms per batch\n"
        summary += "  • Mixed precision: 2-3x speedup\n"

        return summary


# Convenience function for easy integration
def create_optimized_trainer(model: nn.Module,
                           optimizer: optim.Optimizer,
                           criterion: nn.Module,
                           train_loader: DataLoader,
                           val_loader: Optional[DataLoader] = None,
                           **kwargs) -> TrainingPerformanceOptimizer:
    """
    Create an optimized trainer instance with sensible defaults.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        **kwargs: Additional arguments for TrainingPerformanceOptimizer

    Returns:
        Configured TrainingPerformanceOptimizer instance
    """
    return TrainingPerformanceOptimizer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        **kwargs
    )