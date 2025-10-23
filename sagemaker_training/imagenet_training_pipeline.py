#!/usr/bin/env python3
"""
Complete ImageNet Training Pipeline
7-Step Systematic Approach:
1) LR Range Test → 2) Pick LR bounds → 3) Set OneCycle LR + cyclical momentum 
→ 4) Choose batch size → 5) Tune weight-decay & regularizers → 6) Full OneCycle training → 7) Monitor & iterate
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import gc
from typing import Dict, List, Tuple, Optional
import logging

from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_dataloaders
from ilsvrc_dataset import get_ilsvrc_dataloaders
from training_performance_optimizer import TrainingPerformanceOptimizer, create_optimized_trainer
from logger_setup import get_unified_logger

# Set memory fragmentation fix BEFORE any PyTorch operations
# Increased to 512MB to aggressively prevent fragmentation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    print("🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 to aggressively prevent memory fragmentation")

def optimize_num_workers(batch_size, available_memory_gb=None, cpu_count=None):
    """
    Optimize num_workers for DataLoader based on system resources and batch size.

    Strategy:
    1. Start with CPU cores - 1 as baseline
    2. Adjust based on available memory to prevent excessive RAM usage
    3. Consider batch size impact on memory footprint
    4. Cap at reasonable maximum (8) to avoid diminishing returns

    Args:
        batch_size: Current batch size being used
        available_memory_gb: Available system RAM in GB (auto-detected if None)
        cpu_count: Number of CPU cores (auto-detected if None)

    Returns:
        int: Optimal num_workers value
    """
    import psutil
    import multiprocessing as mp

    # Auto-detect system resources
    if cpu_count is None:
        cpu_count = mp.cpu_count()
    if available_memory_gb is None:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Base calculation: CPU cores - 1 (leave one core for main process)
    base_workers = max(1, cpu_count - 1)

    # Memory-based adjustment (rough heuristic)
    # Assume ~500MB per worker for typical ImageNet preprocessing
    memory_per_worker_mb = 500
    max_workers_by_memory = int(available_memory_gb * 1024 / memory_per_worker_mb)

    # Batch size adjustment (larger batches need fewer workers to avoid memory pressure)
    if batch_size >= 128:
        batch_factor = 0.7  # Reduce workers for large batches
    elif batch_size >= 64:
        batch_factor = 0.8
    elif batch_size >= 32:
        batch_factor = 0.9
    else:
        batch_factor = 1.0  # Small batches can use more workers

    # Calculate optimal workers
    optimal_workers = min(base_workers, max_workers_by_memory)
    optimal_workers = int(optimal_workers * batch_factor)
    optimal_workers = max(1, min(optimal_workers, 8))  # Cap between 1-8

    return optimal_workers

def monitor_gpu_utilization(duration_seconds=5):
    """
    Monitor GPU utilization over a period to determine if data loading is bottleneck.

    Returns:
        dict: GPU utilization statistics
    """
    if not torch.cuda.is_available():
        return {'avg_utilization': 0, 'is_bottleneck': False}
    
    try:
        import subprocess
        import time
        
        # Run nvidia-smi monitoring for the specified duration
        cmd = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '--loop-ms=1000']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        utilizations = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            line = process.stdout.readline().strip()
            if line and line.isdigit():
                utilizations.append(int(line))
            time.sleep(0.1)
        
        process.terminate()
        
        if utilizations:
            avg_utilization = sum(utilizations) / len(utilizations)
            # Consider GPU bottleneck if utilization is below 80%
            is_bottleneck = avg_utilization < 80
            return {
                'avg_utilization': avg_utilization,
                'is_bottleneck': is_bottleneck,
                'samples': len(utilizations)
            }
        else:
            return {'avg_utilization': 0, 'is_bottleneck': True, 'samples': 0}
            
    except Exception as e:
        # Fallback if nvidia-smi monitoring fails
        return {'avg_utilization': 50, 'is_bottleneck': False, 'error': str(e)}

def aggressive_memory_cleanup():
    """
    Aggressive GPU memory cleanup to prevent fragmentation and OOM errors.
    """
    if torch.cuda.is_available():
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache multiple times
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Additional cleanup - reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger = get_unified_logger("memory_cleanup")
        logger.info(f"[MEMORY] Cleanup completed - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def check_memory_fragmentation():
    """
    Check for memory fragmentation and warn if severe.
    
    Returns:
        bool: True if fragmentation is severe and may cause OOM
    """
    if not torch.cuda.is_available():
        return False
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    if reserved == 0:
        return False
    
    fragmentation_ratio = (reserved - allocated) / reserved
    
    # Severe fragmentation if reserved is much higher than allocated
    is_severe = fragmentation_ratio > 0.8  # More than 80% fragmentation
    
    if is_severe:
        allocated_gb = allocated / 1024**3
        reserved_gb = reserved / 1024**3
        logger = get_unified_logger("memory_monitor")
        logger.warning(f"[MEMORY] Severe fragmentation detected - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB, Ratio: {fragmentation_ratio:.2%}")
        logger.warning("[MEMORY] Consider reducing batch size or increasing max_split_size_mb")
    
    return is_severe

def log_detailed_memory_usage(stage="unknown"):
    """
    Log detailed GPU memory usage for debugging.
    
    Args:
        stage: String describing the current stage for logging
    """
    if not torch.cuda.is_available():
        return
    
    logger = get_unified_logger("memory_detailed")
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    
    # Calculate fragmentation
    fragmentation_ratio = (reserved - allocated) / reserved if reserved > 0 else 0
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name()
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"[MEMORY:{stage}] GPU: {gpu_name} ({gpu_memory_total:.1f}GB total)")
    logger.info(f"[MEMORY:{stage}] Current - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Fragmentation: {fragmentation_ratio:.1%}")
    logger.info(f"[MEMORY:{stage}] Peak - Allocated: {max_allocated:.2f}GB, Reserved: {max_reserved:.2f}GB")
    
    # Warn if approaching limits
    if allocated > gpu_memory_total * 0.9:
        logger.error(f"[MEMORY:{stage}] CRITICAL: Allocated memory ({allocated:.2f}GB) > 90% of GPU capacity ({gpu_memory_total:.1f}GB)")
    elif allocated > gpu_memory_total * 0.8:
        logger.warning(f"[MEMORY:{stage}] WARNING: Allocated memory ({allocated:.2f}GB) > 80% of GPU capacity ({gpu_memory_total:.1f}GB)")
    
    if fragmentation_ratio > 0.5:
        logger.warning(f"[MEMORY:{stage}] High fragmentation detected: {fragmentation_ratio:.1%}")

def get_ultra_conservative_batch_size(model, input_shape=(3, 224, 224), device='cuda', max_memory_gb=22.0):
    """
    Calculate ultra-conservative batch size that leaves significant memory buffer.
    Uses extremely conservative estimates to prevent OOM errors.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Device to test on
        max_memory_gb: Maximum GPU memory in GB
    
    Returns:
        int: Ultra-conservative batch size
    """
    if not torch.cuda.is_available():
        return 8  # Very conservative for CPU
    
    logger = get_unified_logger("batch_size_calc")
    
    # Reserve massive memory for overhead (model params, gradients, optimizer states, etc.)
    # Use only 40% of GPU memory for batch processing to leave huge buffer
    available_for_batch = max_memory_gb * 0.4
    logger.info(f"[BATCH_CALC] Using only {available_for_batch:.1f}GB ({available_for_batch/max_memory_gb:.0%}) of GPU memory for batch processing")
    
    # Estimate memory per sample (very conservative heuristic)
    # ImageNet 224x224 with ResNet50: ~3-5MB per sample including all overhead
    estimated_mb_per_sample = 5.0  # Ultra-conservative estimate
    
    # Calculate maximum batch size
    max_batch = int((available_for_batch * 1024) / estimated_mb_per_sample)
    
    # Apply extremely conservative safety factors - target very small batches
    conservative_batch = max(1, min(max_batch // 8, 4))  # Max 4, min 1 - very aggressive
    
    logger.info(f"[BATCH_CALC] Estimated max batch: {max_batch}, Ultra-conservative batch: {conservative_batch}")
    logger.info(f"[BATCH_CALC] Memory estimate: {estimated_mb_per_sample:.1f}MB per sample, {available_for_batch:.1f}GB available")
    
    return conservative_batch

# Global progress bar manager to avoid subprocess complexity
class LiveProgressManager:
    """Manages live updating progress bars and status messages to avoid subprocess issues"""

    def __init__(self):
        self.current_bar = None
        self.logger = get_unified_logger("LiveProgressManager")
        self._status_lines = {}  # Track status messages for in-place updates

    def create_status_updater(self, status_key, initial_message=""):
        """Create a status updater that can update messages in place"""
        self._status_lines[status_key] = {
            'message': initial_message,
            'first_print': True
        }
        if initial_message:
            self.logger.info(initial_message)

    def update_status(self, status_key, message):
        """Update a status message in place"""
        if status_key not in self._status_lines:
            self.create_status_updater(status_key, message)
            return

        status_info = self._status_lines[status_key]

        # Handle cursor control for status updates
        if not status_info['first_print']:
            # Move cursor up one line and clear it for subsequent prints
            print('\033[1A\033[K', end='', flush=True)
        else:
            status_info['first_print'] = False

        status_info['message'] = message
        self.logger.info(message)

    def finalize_status(self, status_key, final_message=None):
        """Finalize a status message (print final message on new line)"""
        if status_key in self._status_lines:
            if final_message:
                # Print final message on a new line
                print()
                self.logger.info(final_message)
            del self._status_lines[status_key]

    def create_progress_bar(self, desc, total, disable_tqdm=False):
        """Create a new progress bar for the current step"""
        if self.current_bar:
            self.current_bar.close()

        # Check if we should disable tqdm (e.g., in SageMaker wrapper mode)
        if os.environ.get('TQDM_DISABLE', '0') == '1' or disable_tqdm:
            self.current_bar = None
            self.logger.info(f"[PROGRESS] Starting {desc} (progress via logs)")
            # Initialize progress tracking for log-based progress
            self._log_progress_desc = desc
            self._log_progress_total = total
            self._log_progress_current = 0
            self._log_progress_start_time = time.time()
            self._last_log_time = 0
            self._last_log_step = 0
            self._draw_ascii_progress_bar(0, total, desc)
            return None
        else:
            self.current_bar = tqdm(
                total=total,
                desc=f"[PROGRESS] {desc}",
                unit="it",
                ncols=120,
                leave=True,  # Keep progress bars visible after completion
                mininterval=1.0,  # Update every 1 second for live progress
                maxinterval=5.0,  # Force update every 5 seconds
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            return self.current_bar
    
    def _draw_ascii_progress_bar(self, current, total, desc, metrics=None):
        """Draw an ASCII progress bar for log environments (SageMaker-compatible)"""
        if total == 0:
            return
            
        percentage = (current / total) * 100
        bar_width = 30
        filled_width = int(bar_width * current / total)
        bar = '#' * filled_width + '-' * (bar_width - filled_width)
        
        elapsed = time.time() - self._log_progress_start_time
        eta = (elapsed / current * (total - current)) if current > 0 else 0
        
        progress_str = f"[PROGRESS] {desc}: {percentage:5.1f}%|{bar}| {current}/{total} [{elapsed:.1f}s<{eta:.1f}s]"
        
        if metrics:
            # Standard training metrics
            if 'loss' in metrics:
                progress_str += f" | Loss: {metrics['loss']:.4f}"
            if 'accuracy' in metrics:
                progress_str += f" | Acc: {metrics['accuracy']:.2f}%"
            if 'lr' in metrics:
                progress_str += f" | LR: {metrics['lr']:.6f}"
            
            # Batch size detection metrics
            if 'batch_size' in metrics:
                progress_str += f" | Batch: {metrics['batch_size']}"
            if 'status' in metrics:
                progress_str += f" | Status: {metrics['status']}"
            
            # Weight decay search metrics
            if 'weight_decay' in metrics:
                progress_str += f" | WD: {metrics['weight_decay']:.2e}"
            if 'val_acc' in metrics:
                progress_str += f" | Val Acc: {metrics['val_acc']:.2f}%"
            if 'best_so_far' in metrics:
                progress_str += f" | Best: {metrics['best_so_far']:.2f}%"
            
            # Full training epoch metrics
            if 'epoch' in metrics:
                progress_str += f" | Epoch: {metrics['epoch']}"
            if 'train_loss' in metrics:
                progress_str += f" | Train Loss: {metrics['train_loss']:.4f}"
            if 'train_acc' in metrics:
                progress_str += f" | Train Acc: {metrics['train_acc']:.2f}%"
            if 'val_loss' in metrics:
                progress_str += f" | Val Loss: {metrics['val_loss']:.4f}"
            if 'val_acc' in metrics:
                progress_str += f" | Val Acc: {metrics['val_acc']:.2f}%"
            if 'best_val_acc' in metrics:
                progress_str += f" | Best Val: {metrics['best_val_acc']:.2f}%"
            
            # Results analysis metrics
            if 'step' in metrics:
                progress_str += f" | {metrics['step']}"
        
        # For SageMaker compatibility: only log at significant progress points
        # Log every 10% progress or every 2 minutes, whichever comes first
        current_time = time.time()
        time_since_last_log = current_time - getattr(self, '_last_log_time', 0)
        progress_since_last_log = current - getattr(self, '_last_log_step', 0)
        
        should_log = (
            current == 0 or  # Always log start
            current == total or  # Always log completion
            percentage % 10 < (getattr(self, '_last_logged_percentage', 0) % 10) or  # Every 10%
            time_since_last_log >= 120  # Every 2 minutes
        )
        
        if should_log:
            self.logger.info(progress_str)
            self._last_log_time = current_time
            self._last_log_step = current
            self._last_logged_percentage = percentage
    
    def update_progress(self, step, metrics=None):
        """Update progress bar with metrics"""
        if self.current_bar:
            self.current_bar.n = step
            if metrics:
                desc = self.current_bar.desc.split('|')[0].strip()  # Keep base description
                if 'loss' in metrics:
                    desc += f" | Loss: {metrics['loss']:.4f}"
                if 'accuracy' in metrics:
                    desc += f" | Acc: {metrics['accuracy']:.2f}%"
                if 'lr' in metrics:
                    desc += f" | LR: {metrics['lr']:.6f}"
                self.current_bar.set_description(desc)
            self.current_bar.refresh()
        
        # Log progress more frequently for better visibility
        if step > 0 and self.current_bar:
            total = self.current_bar.total
            percentage = (step / total) * 100
            # Log every 20% or every 50 steps, whichever is more frequent
            log_interval = min(max(1, total // 5), 50)
            if step % log_interval == 0:
                self.logger.info(f"[ANALYSIS] Progress: {percentage:.1f}% ({step}/{total})")
    def update_progress(self, step, metrics=None):
        """Update progress bar with metrics"""
        if self.current_bar:
            self.current_bar.n = step
            if metrics:
                desc = self.current_bar.desc.split('|')[0].strip()  # Keep base description
                if 'loss' in metrics:
                    desc += f" | Loss: {metrics['loss']:.4f}"
                if 'accuracy' in metrics:
                    desc += f" | Acc: {metrics['accuracy']:.2f}%"
                if 'lr' in metrics:
                    desc += f" | LR: {metrics['lr']:.6f}"
                self.current_bar.set_description(desc)
            self.current_bar.refresh()
        
        elif not self.current_bar:
            # ASCII progress bar for log environments (SageMaker-compatible)
            # The _draw_ascii_progress_bar method now handles its own logging intervals
            self._log_progress_current = step
            self._draw_ascii_progress_bar(step, self._log_progress_total, self._log_progress_desc, metrics)
    
    def close_progress_bar(self):
        """Close current progress bar"""
        if self.current_bar:
            self.current_bar.close()
            self.current_bar = None
        elif hasattr(self, '_log_progress_total'):
            # Show final completion for ASCII progress bar
            self._draw_ascii_progress_bar(self._log_progress_total, self._log_progress_total, self._log_progress_desc)
            # Print completion message on a new line (don't overwrite the final progress bar)
            print()  # New line
            self.logger.info(f"[OK] {self._log_progress_desc} completed!")

# Global progress manager instance
progress_manager = LiveProgressManager()


class LRFinder:
    """Learning Rate Range Test Implementation"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def range_test(self, dataloader, start_lr=1e-7, end_lr=1, num_iter=100, smooth_factor=0.05):
        """Perform LR range test"""
        logger = get_unified_logger("lr_range_test")
        logger.info(f"[DEBUG] Starting LR Range Test: {start_lr:.2e} → {end_lr:.2e}")
        
        # Calculate multiplicative factor
        lr_lambda = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
            
        self.model.train()
        losses = []
        lrs = []
        best_loss = float('inf')
        
        # Use progress manager for clean progress tracking
        progress_manager.create_progress_bar("LR Range Test", num_iter)
        data_iter = iter(dataloader)
        
        for i in range(num_iter):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Smoothed loss
            if i == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = smooth_factor * loss.item() + (1 - smooth_factor) * losses[-1]
            
            losses.append(smoothed_loss)
            
            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss or torch.isnan(loss):
                logger.warning(f"[ERROR] Stopping early at iteration {i}, loss exploded")
                break
                
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Memory cleanup to prevent OOM
            del outputs, loss
            if i % 10 == 0:  # Clean cache every 10 iterations
                torch.cuda.empty_cache()
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_lambda
                
            # Update progress with metrics
            progress_manager.update_progress(i + 1, {
                'lr': current_lr,
                'loss': smoothed_loss
            })
            
        progress_manager.close_progress_bar()
        
        self.history['lr'] = lrs
        self.history['loss'] = losses
        
        return lrs, losses
    
    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plot LR range test results"""
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if log_lr:
            ax.semilogx(lrs, losses)
            ax.set_xlabel('Learning Rate (log scale)')
        else:
            ax.plot(lrs, losses)
            ax.set_xlabel('Learning Rate')
        
        ax.set_ylabel('Loss')
        ax.set_title('LR Range Test')
        ax.grid(True, alpha=0.3)
        
        # Find minimum
        min_idx = np.argmin(losses)
        min_lr = lrs[min_idx]
        min_loss = losses[min_idx]
        ax.annotate(f'Min: LR={min_lr:.2e}', 
                   xy=(min_lr, min_loss), 
                   xytext=(min_lr*10, min_loss*1.1),
                   arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        return fig, min_lr
    
    def suggest_lr(self, skip_start=10, skip_end=5):
        """Suggest optimal learning rate"""
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        
        # Find steepest decline
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)
        
        # Find minimum loss
        min_loss_idx = np.argmin(losses)
        
        steepest_lr = lrs[min_gradient_idx]
        min_loss_lr = lrs[min_loss_idx]
        
        # Suggest LR (typically 10x smaller than steepest decline)
        suggested_max_lr = steepest_lr / 10
        suggested_min_lr = suggested_max_lr / 25  # OneCycle typically uses 1/25 ratio
        
        return {
            'min_lr': suggested_min_lr,
            'max_lr': suggested_max_lr,
            'steepest_decline_lr': steepest_lr,
            'min_loss_lr': min_loss_lr
        }


class BatchSizeFinder:
    """Find optimal batch size"""
    
    @staticmethod
    def find_max_batch_size(model, input_shape, device, max_batch_size=2048):
        """Find maximum batch size that fits in memory during training (more realistic test)"""
        logger = get_unified_logger("batch_size_finder")
        model.train()  # Use training mode for realistic memory usage
        batch_size = 1
        criterion = nn.CrossEntropyLoss()
        
        logger.info("[DEBUG] Finding maximum batch size (training mode)...")
        
        # Calculate number of batch size tests needed
        test_count = 0
        temp_batch = 1
        while temp_batch <= max_batch_size:
            test_count += 1
            temp_batch *= 2
        
        # Create progress bar for batch size testing
        progress_manager.create_progress_bar("Batch Size Detection", test_count)
        
        test_idx = 0
        batch_size = 1
        
        while batch_size <= max_batch_size:
            try:
                # Create dummy input and target
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                dummy_target = torch.randint(0, 1000, (batch_size,)).to(device)
                
                # Test forward and backward pass (more realistic)
                outputs = model(dummy_input)
                loss = criterion(outputs, dummy_target)
                loss.backward()
                
                # Clean up
                model.zero_grad()
                del dummy_input, dummy_target, outputs, loss
                torch.cuda.empty_cache()
                
                logger.info(f"[OK] Batch size {batch_size} works (train mode)")
                
                # Update progress
                test_idx += 1
                progress_manager.update_progress(test_idx, {
                    'batch_size': batch_size,
                    'status': 'success'
                })
                
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"[ERROR] Batch size {batch_size} failed (OOM)")
                    max_working_batch_size = batch_size // 2
                    logger.info(f"[COMPLETE] Maximum batch size: {max_working_batch_size}")
                    
                    # Update final progress
                    progress_manager.update_progress(test_count, {
                        'batch_size': max_working_batch_size,
                        'status': 'complete'
                    })
                    progress_manager.close_progress_bar()
                    
                    return max_working_batch_size
                else:
                    progress_manager.close_progress_bar()
                    raise e
        
        max_working_batch_size = max_batch_size // 2
        
        # Update final progress
        progress_manager.update_progress(test_count, {
            'batch_size': max_working_batch_size,
            'status': 'complete'
        })
        progress_manager.close_progress_bar()
        
        return max_working_batch_size


class HyperparameterOptimizer:
    """Grid/Random search for hyperparameters"""
    
    def __init__(self, model_fn, train_loader, val_loader, device):
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def weight_decay_search(self, lr_config, batch_size, wd_values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], epochs=5):
        """Search for optimal weight decay"""
        logger = get_unified_logger()
        logger.info(f"weight_decay_search called with:")
        logger.info(f"  lr_config: min_lr={lr_config.get('min_lr', 'N/A')}, max_lr={lr_config.get('max_lr', 'N/A')}")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  wd_values: {wd_values}")
        logger.info(f"  epochs: {epochs}")
        logger.info(f"[DEBUG] Weight Decay Search: {wd_values}")
        
        results = []
        
        # Create progress bar for weight decay search
        progress_manager.create_progress_bar("Weight Decay Search", len(wd_values))
        
        for idx, wd in enumerate(wd_values):
            logger.info(f"[ANALYSIS] Testing Weight Decay: {wd:.2e} ({idx+1}/{len(wd_values)})")
            
            # Create fresh model
            model = self.model_fn().to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=lr_config['min_lr'], 
                                momentum=0.9, weight_decay=wd, nesterov=True)
            criterion = nn.CrossEntropyLoss()
            
            # OneCycle scheduler
            scheduler = OneCycleLR(optimizer, max_lr=lr_config['max_lr'], 
                                 epochs=epochs, steps_per_epoch=len(self.train_loader))
            
            # Train for a few epochs
            train_losses, val_losses, val_accs = self._quick_train(
                model, optimizer, criterion, scheduler, epochs)
            
            # Store results
            result = {
                'weight_decay': wd,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_acc': max(val_accs),
                'final_val_acc': val_accs[-1],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            results.append(result)
            
            logger.info(f"[PLOT] Results - Val Acc: {result['final_val_acc']:.2f}%, "
                  f"Val Loss: {result['final_val_loss']:.3f}")
            
            # Update progress
            progress_manager.update_progress(idx + 1, {
                'weight_decay': wd,
                'val_acc': result['final_val_acc'],
                'best_so_far': max(r['best_val_acc'] for r in results)
            })
            
            # Clean up GPU memory to prevent OOM
            del model, optimizer, criterion, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        progress_manager.close_progress_bar()
        
        # Find best weight decay
        best_result = max(results, key=lambda x: x['best_val_acc'])
        logger.info(f"[COMPLETE] Best Weight Decay: {best_result['weight_decay']:.2e} "
              f"(Val Acc: {best_result['best_val_acc']:.2f}%)")
        
        return results, best_result['weight_decay']
    
    def _quick_train(self, model, optimizer, criterion, scheduler, epochs):
        """Quick training for hyperparameter search"""
        train_losses = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            # Use progress manager for clean progress tracking
            total_batches = min(100, len(self.train_loader))  # Limit for speed
            progress_manager.create_progress_bar(f"Epoch {epoch+1}/{epochs}", total_batches)
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if batch_idx >= total_batches:  # Limit training batches for speed
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Update progress with metrics
                progress_manager.update_progress(batch_idx + 1, {
                    'loss': train_loss/train_batches,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            progress_manager.close_progress_bar()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0
            
            # Use progress manager for validation progress
            val_limit = min(50, len(self.val_loader))  # Limit validation batches for speed
            progress_manager.create_progress_bar(f"Validation {epoch+1}/{epochs}", val_limit)
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_batches += 1
                    
                    # Update progress with validation metrics
                    progress_manager.update_progress(val_batches, {
                        'loss': val_loss/val_batches,
                        'accuracy': 100.*val_correct/val_total
                    })
                    
                    # Limit validation batches for speed
                    if val_batches >= 50:
                        break
            
            progress_manager.close_progress_bar()
            
            train_losses.append(train_loss / train_batches)
            val_losses.append(val_loss / val_batches)
            val_accs.append(100. * val_correct / val_total)
            
            # Clean up GPU memory to prevent OOM accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        return train_losses, val_losses, val_accs


class FullTrainer:
    """Full training with monitoring"""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'momentum': []
        }
        
    def train(self, lr_config, epochs, batch_size, weight_decay=1e-4, 
              save_checkpoints=True, early_stopping_patience=10, args=None, performance_optimizer=None,
              gradient_accumulation_steps=1):
        """Full training with OneCycle LR and cyclical momentum"""
        
        logger = get_unified_logger()
        logger.info("[START] Starting Full Training:")
        logger.info(f"   📚 Epochs: {epochs}")
        logger.info(f"   [SIZE] LR Range: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
        logger.info(f"   [WEIGHT]  Weight Decay: {weight_decay:.2e}")
        logger.info(f"   [BATCH] Batch Size: {batch_size}")
        logger.info(f"   [GRAD] Gradient Accumulation Steps: {gradient_accumulation_steps}")
        if gradient_accumulation_steps > 1:
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.info(f"   [EFFECTIVE] Effective Batch Size: {effective_batch_size}")
        
        # Scale learning rate based on batch size using TrainingPerformanceOptimizer
        base_min_lr = lr_config['min_lr']
        base_max_lr = lr_config['max_lr']
        
        if performance_optimizer:
            logger.info("[OPTIMIZER] Scaling learning rates for batch size...")
            scaled_min_lr = performance_optimizer.scale_learning_rate_for_batch_size(
                base_lr=base_min_lr,
                base_batch_size=32,  # Standard batch size used for LR range test
                current_batch_size=batch_size,
                scaling_factor=1.0
            )
            scaled_max_lr = performance_optimizer.scale_learning_rate_for_batch_size(
                base_lr=base_max_lr,
                base_batch_size=32,  # Standard batch size used for LR range test
                current_batch_size=batch_size,
                scaling_factor=1.0
            )
            logger.info(f"[OPTIMIZER] LR scaled: {base_min_lr:.2e}→{scaled_min_lr:.2e}, {base_max_lr:.2e}→{scaled_max_lr:.2e}")
        else:
            scaled_min_lr = base_min_lr
            scaled_max_lr = base_max_lr
            logger.info("[OPTIMIZER] No performance optimizer provided, using original LRs")
        
        # Setup optimizer and scheduler with scaled LRs
        optimizer = optim.SGD(self.model.parameters(), lr=scaled_min_lr,
                            momentum=0.85, weight_decay=weight_decay, nesterov=True)
        
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=scaled_max_lr,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader) // gradient_accumulation_steps,  # Adjust for gradient accumulation
            pct_start=0.3,
            div_factor=lr_config['max_lr'] / lr_config['min_lr'],
            final_div_factor=1000,
            base_momentum=0.85,
            max_momentum=0.95
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Enable torch.compile for additional speedup (PyTorch 2.0+)
        if not args.no_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                logger.info("[SPEED] torch.compile enabled - additional performance boost")
            except Exception as e:
                logger.warning(f"[SPEED] torch.compile failed: {e}, continuing without it")
        
        # Enable mixed precision training for massive speedup
        scaler = None
        if torch.cuda.is_available() and not args.no_amp:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("[SPEED] Mixed precision training enabled (AMP) - expect 2-3x speedup")
        
        # Enable cuDNN benchmark for faster convolutions
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("[SPEED] cuDNN benchmark enabled for faster convolutions")
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        # Create epoch-level progress bar
        progress_manager.create_progress_bar("Full Training", epochs)
        
        for epoch in range(epochs):
            # Create status updater for epoch progress
            epoch_status_key = f"epoch_{epoch+1}"
            progress_manager.create_status_updater(epoch_status_key, f"[PROGRESS] Epoch {epoch+1}/{epochs} - Training...")

            # Training
            train_loss, train_acc = self._train_epoch(optimizer, criterion, scheduler, scaler, performance_optimizer, gradient_accumulation_steps)

            # Update status to validation phase
            progress_manager.update_status(epoch_status_key, f"[PROGRESS] Epoch {epoch+1}/{epochs} - Validating...")

            # Validation
            val_loss, val_acc = self._validate_epoch(criterion)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['momentum'].append(optimizer.param_groups[0]['momentum'])

            # Update status with final epoch results
            progress_manager.update_status(epoch_status_key,
                f"[OK] Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% | Val: {val_loss:.4f}/{val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Finalize the epoch status
            progress_manager.finalize_status(epoch_status_key)
            
            # Clean up GPU memory at end of each epoch to prevent accumulation across epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_checkpoints:
                    self._save_checkpoint(epoch, val_acc, optimizer, scheduler)
                # Use status updater for best model message
                best_model_key = "best_model"
                progress_manager.create_status_updater(best_model_key, f"[SAVE] New best model saved! Val Acc: {val_acc:.2f}%")
                progress_manager.finalize_status(best_model_key)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                early_stop_key = "early_stopping"
                progress_manager.create_status_updater(early_stop_key, f"[TIME] Early stopping after {patience_counter} epochs without improvement")
                progress_manager.finalize_status(early_stop_key)
                break

        progress_manager.close_progress_bar()
        completion_key = "training_complete"
        progress_manager.create_status_updater(completion_key, f"[COMPLETE] Training completed! Best Val Acc: {best_val_acc:.2f}%")
        progress_manager.finalize_status(completion_key)
        return self.history
    
    def _train_epoch(self, optimizer, criterion, scheduler, scaler=None, performance_optimizer=None, gradient_accumulation_steps=1):
        """Train one epoch with optional mixed precision, performance optimization, and gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use progress manager for clean progress tracking
        progress_manager.create_progress_bar("Training", len(self.train_loader))
        
        accumulation_counter = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs, targets = batch_data
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Use TrainingPerformanceOptimizer for optimized training step if available
            if performance_optimizer:
                step_metrics = performance_optimizer.optimize_training_step((inputs, targets), step_type='train')
                loss = step_metrics['loss']
                
                # Get predictions for accuracy calculation
                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                
                # For gradient accumulation with performance optimizer, we need to handle accumulation manually
                accumulation_counter += 1
                if accumulation_counter % gradient_accumulation_steps == 0:
                    # Update optimizer and scheduler manually since optimizer is managed by performance_optimizer
                    scheduler.step()
                    accumulation_counter = 0
            else:
                # Original training step logic with gradient accumulation
                # Only zero gradients at the start of accumulation cycle
                if accumulation_counter == 0:
                    optimizer.zero_grad()
                
                # Mixed precision training
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Scale loss and backpropagate (accumulate gradients)
                    scaler.scale(loss / gradient_accumulation_steps).backward()
                else:
                    # Standard precision training
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    (loss / gradient_accumulation_steps).backward()
                
                _, predicted = outputs.max(1)
                accumulation_counter += 1
                
                # Perform optimizer step only after accumulating gradients
                if accumulation_counter % gradient_accumulation_steps == 0:
                    if scaler:
                        # Gradient clipping with scaler
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Optimizer step with scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        optimizer.step()
                    
                    # Step scheduler
                    scheduler.step()
                    accumulation_counter = 0
            
            running_loss += loss
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress with metrics
            progress_manager.update_progress(batch_idx + 1, {
                'loss': running_loss/(batch_idx + 1),
                'accuracy': 100.*correct/total,
                'lr': optimizer.param_groups[0]["lr"]
            })
            
            # Clean up GPU memory after each batch to prevent accumulation
            del inputs, targets, outputs, loss, predicted
            if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:  # Clean every 50 batches (more frequent)
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        progress_manager.close_progress_bar()
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def _validate_epoch(self, criterion):
        """Validate one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Use progress manager for clean progress tracking
            progress_manager.create_progress_bar("Validation", len(self.val_loader))
            
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress with metrics
                progress_manager.update_progress(batch_idx + 1, {
                    'loss': running_loss/(batch_idx + 1),
                    'accuracy': 100.*correct/total
                })
                
                # Clean up GPU memory after each validation batch
                del inputs, targets, outputs, loss, predicted
                if torch.cuda.is_available() and (batch_idx + 1) % 25 == 0:  # Clean every 25 batches (more frequent for validation)
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            progress_manager.close_progress_bar()
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def _save_checkpoint(self, epoch, val_acc, optimizer, scheduler):
        """Save model checkpoint"""
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))


def detect_dataset_format(data_path):
    """
    Detect whether the dataset is in standard ImageNet format or ILSVRC format
    
    Args:
        data_path: Path to dataset directory
        
    Returns:
        'imagenet' or 'ilsvrc'
    """
    logger = get_unified_logger()
    logger.info(f"[DEBUG] Checking dataset format for: {data_path}")
    
    # Case 1: Check if data_path points directly to ILSVRC root
    ilsvrc_root_indicators = [
        os.path.join(data_path, "Data", "CLS-LOC"),
        os.path.join(data_path, "ImageSets", "CLS-LOC"),
        os.path.join(data_path, "ImageSets", "CLS-LOC", "val.txt")
    ]
    
    if all(os.path.exists(path) for path in ilsvrc_root_indicators):
        logger.info("[OK] Detected ILSVRC format (root directory)")
        return 'ilsvrc'
    
    # Case 2: Check if data_path points to CLS-LOC subdirectory
    # Look for parent ILSVRC structure
    if data_path.endswith("Data/CLS-LOC") or data_path.endswith("Data\\CLS-LOC"):
        # Go up two levels to find ILSVRC root
        potential_root = os.path.dirname(os.path.dirname(data_path))
        imagesets_path = os.path.join(potential_root, "ImageSets", "CLS-LOC", "val.txt")
        if os.path.exists(imagesets_path):
            logger.info("[OK] Detected ILSVRC format (CLS-LOC subdirectory)")
            return 'ilsvrc'
    
    # Case 3: Check if we have flat validation directory (ILSVRC-style)
    val_dir = os.path.join(data_path, "val")
    if os.path.exists(val_dir):
        # Check if validation directory has subdirectories (standard) or flat files (ILSVRC)
        val_contents = os.listdir(val_dir)
        if val_contents:
            first_item = os.path.join(val_dir, val_contents[0])
            if os.path.isfile(first_item) and first_item.lower().endswith(('.jpg', '.jpeg')):
                logger.info("[OK] Detected ILSVRC format (flat validation directory)")
                return 'ilsvrc'
    
    # Case 4: Check for standard ImageNet format
    standard_paths = [
        os.path.join(data_path, "train"),
        os.path.join(data_path, "val")
    ]
    
    if all(os.path.exists(path) for path in standard_paths):
        logger.info("[OK] Detected standard ImageNet format")
        return 'imagenet'
    
    # Default to ILSVRC if we can't determine
    logger.warning("⚠️  Could not determine format, defaulting to ILSVRC")
    return 'ilsvrc'


def main():
    """Main training pipeline"""
    import sys
    import os
    
    # Set up unified logging first thing
    logger = get_unified_logger("imagenet_training_pipeline")
    
    # =============================================================================
    # SAGEMAKER TRAINING STARTED - SIMPLE STATUS LOG
    # =============================================================================
    logger.info("=" * 80)
    logger.info("[START] SAGEMAKER IMAGENET TRAINING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"[TIME] Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[PYTHON] Python: {sys.version}")
    logger.info(f"[PYTORCH] PyTorch: {torch.__version__}")
    logger.info(f"[SYSTEM] Working Directory: {os.getcwd()}")
    logger.info("=" * 80)
    
    # Log to unified log file
    logger.info("="*80)
    logger.debug("[DEBUG] DEBUG: Entered main() function")
    logger.info("="*80)
    logger.debug("[DEBUG] DEBUG: Creating argument parser")
    logger.info(f"[PYTHON] Python version: {sys.version}")
    logger.info(f"[PYTORCH] PyTorch version: {torch.__version__}")
    logger.info(f"[SYSTEM] Working directory: {os.getcwd()}")
    logger.info(f"[FILE] Script path: {sys.argv[0]}")
    logger.info(f"[ARGS] Command line args: {sys.argv[1:] if len(sys.argv) > 1 else 'None'}")
    
    print("[DEBUG] DEBUG: Entered main() function")
    sys.stdout.flush()
    logger.debug("[DEBUG] DEBUG: Entered main() function")
    
    print("[DEBUG] DEBUG: Creating argument parser")
    logger.debug("[DEBUG] DEBUG: Creating argument parser")
    logger.debug("[DEBUG] DEBUG: Parsing arguments")
    parser = argparse.ArgumentParser(description='ImageNet Training Pipeline')
    parser.add_argument('--train', type=str, required=True, help='ImageNet training dataset path')
    parser.add_argument('--val', type=str, required=True, help='ImageNet validation dataset path')
    parser.add_argument('--output', type=str, default='./imagenet_pipeline_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detect if not specified)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--quick-mode', action='store_true', help='Enable quick mode for faster testing')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile optimization')
    parser.add_argument('--lightweight-augs', action='store_true', help='Use lightweight augmentations for maximum speed')
    logger.debug("[DEBUG] DEBUG: Setting up logger")
    parser.add_argument('--skip-lr-test', action='store_true', help='Skip LR range test')
    parser.add_argument('--skip-wd-search', action='store_true', help='Skip weight decay search')
    logger.debug("[DEBUG] DEBUG: Logger setup completed")
    
    print("[DEBUG] DEBUG: Parsing arguments")
    sys.stdout.flush()
    args = parser.parse_args()
    print(f"[DEBUG] DEBUG: Arguments parsed successfully - train: {args.train}, val: {args.val}, epochs: {args.epochs}")
    sys.stdout.flush()
    
    # Setup logging
    print("[DEBUG] DEBUG: Setting up logger")
    sys.stdout.flush()
    logger = get_unified_logger('imagenet_pipeline')
    print("[DEBUG] DEBUG: Logger setup completed")
    sys.stdout.flush()
    
    # DEBUG: Add extensive early debugging
    logger.info("[DEBUG] DEBUG: Arguments parsed successfully")
    logger.info(f"[DEBUG] DEBUG: Training data path: {args.train}")
    logger.info(f"[DEBUG] DEBUG: Validation data path: {args.val}")
    logger.info(f"[DEBUG] DEBUG: Output path: {args.output}")
    logger.info(f"[DEBUG] DEBUG: Epochs: {args.epochs}")
    
    # Check if paths exist
    import os
    logger.info(f"[DEBUG] DEBUG: Train path exists: {os.path.exists(args.train)}")
    logger.info(f"[DEBUG] DEBUG: Val path exists: {os.path.exists(args.val)}")
    
    # Check if paths have content
    if os.path.exists(args.train):
        try:
            train_contents = os.listdir(args.train)
            logger.info(f"[DEBUG] DEBUG: Train path has {len(train_contents)} items")
            if len(train_contents) > 0:
                logger.info(f"[DEBUG] DEBUG: First few train items: {train_contents[:3]}")
        except Exception as e:
            logger.error(f"[ERROR] DEBUG: Error listing train directory: {e}")
    
    if os.path.exists(args.val):
        try:
            val_contents = os.listdir(args.val)
            logger.info(f"[DEBUG] DEBUG: Val path has {len(val_contents)} items")
            if len(val_contents) > 0:
                logger.info(f"[DEBUG] DEBUG: First few val items: {val_contents[:3]}")
        except Exception as e:
            logger.error(f"[ERROR] DEBUG: Error listing val directory: {e}")
    
    # Setup
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[DEVICE]  Using device: {device}")
        logger.info(f"[DEBUG] DEBUG: CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"[DEBUG] DEBUG: CUDA device count: {torch.cuda.device_count()}")
    except Exception as e:
        logger.error(f"[ERROR] DEBUG: Error setting up device: {e}")
        raise
    
    try:
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"[DEBUG] DEBUG: Output directory created: {args.output}")
    except Exception as e:
        logger.error(f"[ERROR] DEBUG: Error creating output directory: {e}")
        raise
    
    # Detect dataset format
    #dataset_format = detect_dataset_format(args.data)
    #logger.info(f"[DATASET] Detected dataset format: {dataset_format.upper()}")
    
    # Model factory
    def create_model():
        logger.info("[DEBUG] DEBUG: Creating model...")
        try:
            model = resnet50_imagenet(num_classes=1000, pretrained=False)
            logger.info("[DEBUG] DEBUG: Model created successfully")
            return model
        except Exception as e:
            logger.error(f"[ERROR] DEBUG: Error creating model: {e}")
            raise
    
        # STEP 0: Batch Size Detection (if not specified)
        logger.debug("[DEBUG] DEBUG: Checking batch size")
    if args.batch_size is None:
        logger.debug("[DEBUG] DEBUG: No batch size specified, starting batch size detection")
        logger.info("="*60)
        logger.info("[CONFIG] STEP 0: Batch Size Detection")
        logger.info("="*60)
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[DEVICE]  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total")
            
            # Create temporary model and optimizer for batch size detection using TrainingPerformanceOptimizer
            logger.info("[DEBUG] DEBUG: About to create temporary model for batch size detection")
            temp_model = create_model().to(device)
            temp_optimizer = optim.SGD(temp_model.parameters(), lr=1e-3, momentum=0.9)
            temp_criterion = nn.CrossEntropyLoss()
            
            # Create temporary optimizer instance for batch size detection
            temp_performance_optimizer = TrainingPerformanceOptimizer(
                model=temp_model,
                optimizer=temp_optimizer,
                criterion=temp_criterion,
                train_loader=None,  # Not needed for batch size detection
                val_loader=None,
                device=device,
                enable_amp=not args.no_amp,
                enable_profiling=False  # Disable profiling for batch size detection
            )
            
            # Use optimizer's batch size detection method
            max_memory_gb = 14.0 if torch.cuda.is_available() else 4.0  # Adjust based on GPU availability
            optimal_batch_size = temp_performance_optimizer.get_optimal_batch_size(max_memory_gb=max_memory_gb)
            
            # Apply safety factors based on mode - now much more conservative
            if args.quick_mode:
                safety_factor = 0.2  # Ultra-conservative for quick mode to prevent OOM
                logger.info("[QUICK] Quick mode: Using ultra-conservative batch size for training stability")
            else:
                safety_factor = 0.3  # Very conservative for full training to prevent OOM
                logger.info("[FULL] Full training: Using very conservative batch size for stability")
            
            initial_batch_size = int(optimal_batch_size * safety_factor)
            # Ensure it's a power of 2 and at least 1, but cap at 8 for safety
            initial_batch_size = max(1, min(2 ** int(np.log2(initial_batch_size)), 8)) if initial_batch_size > 0 else 4
            
            logger.info(f"[COMPLETE] Optimal batch size: {initial_batch_size} (optimizer: {optimal_batch_size}, safety: {safety_factor})")
            
            # Clean up temporary resources
            del temp_model, temp_optimizer, temp_criterion, temp_performance_optimizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"[ERROR] DEBUG: Error in batch size detection: {e}")
            # Fallback to default batch size
            initial_batch_size = 32
            logger.warning(f"[FALLBACK] Using default batch size: {initial_batch_size}")
    else:
        initial_batch_size = args.batch_size
        logger.info(f"[SIZE] Using specified batch size: {initial_batch_size}")
        logger.debug(f"[DEBUG] DEBUG: Using specified batch size: {initial_batch_size}")
    
    # Optimize num_workers for balanced CPU/GPU utilization and memory usage
    logger.info("[OPTIMIZE] Optimizing num_workers for DataLoader...")
    original_num_workers = args.num_workers
    optimized_num_workers = optimize_num_workers(initial_batch_size)
    logger.info(f"[OPTIMIZE] Using optimized num_workers: {optimized_num_workers} (batch_size: {initial_batch_size}, original: {original_num_workers})")
    
    # Monitor GPU utilization to validate optimization
    if torch.cuda.is_available():
        gpu_stats = monitor_gpu_utilization(duration_seconds=2)
        logger.info(f"[GPU] Initial GPU utilization: {gpu_stats['avg_utilization']:.1f}% (bottleneck: {gpu_stats['is_bottleneck']})")
        if gpu_stats['is_bottleneck']:
            logger.warning("[GPU] GPU utilization low - data loading may be bottleneck, num_workers optimization should help")
    
    # Override args.num_workers with optimized value
    args.num_workers = optimized_num_workers
    
    # Load data
    logger.info("[DATASET] Loading ImageNet dataset...")
    logger.debug("[DEBUG] DEBUG: About to load dataset")
    
    # Create progress bar for dataset loading (2 steps: train + val)
    progress_manager.create_progress_bar("Dataset Loading", 2)
    
    #if dataset_format == 'ilsvrc':
    #    logger.info("Using ILSVRC dataset loader (handles flat validation directory)")
    #    train_loader, val_loader = get_ilsvrc_dataloaders(
    #        args.data, batch_size=initial_batch_size, num_workers=4)
    #else:
    logger.info("Using standard ImageNet dataset loader")
    try:
        progress_manager.update_progress(1, {'step': 'Loading training dataset'})
        train_loader, val_loader = get_imagenet_dataloaders(
            train=args.train, val=args.val, batch_size=initial_batch_size, num_workers=args.num_workers, 
            lightweight_augs=args.lightweight_augs)
        progress_manager.update_progress(2, {'step': 'Loading validation dataset'})
    except Exception as e:
        progress_manager.close_progress_bar()
        logger.error(f"Dataset loading failed: {e}")
        raise
    
    progress_manager.close_progress_bar()
    logger.info(f"Dataset loaded - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Initialize Training Performance Optimizer for data loading optimization
    logger.info("[OPTIMIZER] Initializing TrainingPerformanceOptimizer for data loading optimization...")
    try:
        # Create temporary model and optimizer for optimizer initialization
        temp_model = create_model().to(device)
        temp_optimizer = optim.SGD(temp_model.parameters(), lr=1e-3, momentum=0.9)
        temp_criterion = nn.CrossEntropyLoss()
        
        # Create optimizer instance
        performance_optimizer = TrainingPerformanceOptimizer(
            model=temp_model,
            optimizer=temp_optimizer,
            criterion=temp_criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            enable_amp=not args.no_amp,
            enable_profiling=True
        )
        
        # Optimize data loading
        logger.info("[OPTIMIZER] Optimizing data loading pipeline...")
        optimized_train_loader, optimized_val_loader = performance_optimizer.optimize_data_loading(target_workers=16)
        
        # Replace original loaders with optimized ones
        train_loader = optimized_train_loader
        if optimized_val_loader is not None:
            val_loader = optimized_val_loader
        logger.info("[OPTIMIZER] ✅ Data loading optimization complete")
        
        # Clean up temporary resources
        del temp_model, temp_optimizer, temp_criterion
        
    except Exception as e:
        logger.warning(f"[OPTIMIZER] ⚠️ Failed to initialize performance optimizer: {e}")
        logger.warning("[OPTIMIZER] Continuing with standard data loading...")
        performance_optimizer = None
    
    # =============================================================================
    # STARTING 7-STEP IMAGENET TRAINING PIPELINE
    # =============================================================================
    logger.debug("\n" + "=" * 80)
    logger.debug("[COMPLETE] STARTING 7-STEP IMAGENET TRAINING PIPELINE")
    logger.debug("=" * 80)
    logger.debug("[ARGS] Pipeline Steps:")
    logger.debug("   1️⃣  LR Range Test")
    logger.debug("   2️⃣  Weight Decay Search") 
    logger.debug("   3️⃣  Full Training with OneCycle LR")
    logger.debug(f"[BATCH] Batch Size: {initial_batch_size}")
    logger.debug(f"[PROGRESS] Total Epochs: {args.epochs}")
    logger.debug("=" * 80)
    print("[START] Starting Step 1: LR Range Test...")
    print("=" * 80)
    sys.stdout.flush()
    
    # STEP 1: LR Range Test
    lr_config = None
    if not args.skip_lr_test:
        lr_step_key = "lr_range_test"
        progress_manager.create_status_updater(lr_step_key, "[DEBUG] STEP 1: LR Range Test - Starting...")

        model = create_model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        lr_finder = LRFinder(model, optimizer, criterion, device)

        # Create a small batch size dataloader specifically for LR range test to prevent OOM
        lr_test_batch_size = min(16, initial_batch_size // 4)  # Use much smaller batch size for LR test
        lr_test_batch_size = max(8, lr_test_batch_size)  # Minimum batch size of 8
        
        # Optimize num_workers for LR test based on smaller batch size
        lr_test_num_workers = optimize_num_workers(lr_test_batch_size)
        lr_test_num_workers = min(lr_test_num_workers, 4)  # Cap at 4 for LR test to avoid overhead
        
        logger.info(f"[MEMORY] Creating LR range test dataloader with batch_size: {lr_test_batch_size}, num_workers: {lr_test_num_workers}")

        lr_test_loader = get_imagenet_dataloaders(
            train=args.train, val=args.val, batch_size=lr_test_batch_size,
            num_workers=lr_test_num_workers, lightweight_augs=True  # Use lightweight augs for speed
        )[0]  # Only need train loader

        num_iter = 100 if args.quick_mode else 200

        progress_manager.update_status(lr_step_key, f"[DEBUG] STEP 1: LR Range Test - Running {num_iter} iterations...")
        lrs, losses = lr_finder.range_test(lr_test_loader, num_iter=num_iter)

        # Plot results
        fig, min_lr = lr_finder.plot()
        fig.savefig(os.path.join(args.output, 'lr_range_test.png'))
        plt.close(fig)

        # Get suggestions
        lr_config = lr_finder.suggest_lr()

        progress_manager.update_status(lr_step_key,
            f"[OK] STEP 1: LR Range Test Complete - Min: {lr_config['min_lr']:.2e}, Max: {lr_config['max_lr']:.2e}")
        progress_manager.finalize_status(lr_step_key)

        # Save results
        with open(os.path.join(args.output, 'lr_config.json'), 'w') as f:
            json.dump({k: float(v) for k, v in lr_config.items()}, f, indent=2)
            
        # Clean up GPU memory to prevent OOM
        del model, optimizer, criterion, lr_finder, lr_test_loader
        aggressive_memory_cleanup()
    else:
        # Default LR config
        lr_config = {'min_lr': 1e-3, 'max_lr': 0.1}
        lr_skip_key = "lr_skip"
        progress_manager.create_status_updater(lr_skip_key, "⏭️ STEP 1: Skipping LR Range Test, using default config")
        progress_manager.finalize_status(lr_skip_key)
    
    # STEP 2 & 3: Already incorporated in lr_config
    logger.info(f"[OK] LR bounds selected: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
    
    # STEP 4: Batch Size Already Optimized
    optimal_batch_size = initial_batch_size
    logger.info(f"[OK] Using optimized batch size: {optimal_batch_size}")
    
    # STEP 5: Weight Decay Search
    best_weight_decay = 1e-4  # Default
    if not args.skip_wd_search:
        wd_step_key = "weight_decay_search"
        progress_manager.create_status_updater(wd_step_key, "[WEIGHT] STEP 5: Weight Decay Search - Starting...")

        optimizer = HyperparameterOptimizer(create_model, train_loader, val_loader, device)

        wd_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] if not args.quick_mode else [1e-4, 5e-4]
        search_epochs = 3 if args.quick_mode else 5

        progress_manager.update_status(wd_step_key, f"[WEIGHT] STEP 5: Weight Decay Search - Testing {len(wd_values)} values for {search_epochs} epochs each...")
        wd_results, best_weight_decay = optimizer.weight_decay_search(
            lr_config, optimal_batch_size, wd_values, epochs=search_epochs)

        # Save results
        with open(os.path.join(args.output, 'weight_decay_search.json'), 'w') as f:
            json.dump(wd_results, f, indent=2)

        progress_manager.update_status(wd_step_key, f"[OK] STEP 5: Weight Decay Search Complete - Best WD: {best_weight_decay:.2e}")
        progress_manager.finalize_status(wd_step_key)
        
        # Check for memory fragmentation after weight decay search
        check_memory_fragmentation()
        
    else:
        wd_skip_key = "wd_skip"
        progress_manager.create_status_updater(wd_skip_key, "⏭️ STEP 5: Skipping weight decay search, using default 1e-4")
        progress_manager.finalize_status(wd_skip_key)
    
    # Aggressive memory cleanup before full training to prevent fragmentation
    logger.info("[MEMORY] Performing aggressive memory cleanup before full training...")
    aggressive_memory_cleanup()
    
    # Check for memory fragmentation before starting full training
    if check_memory_fragmentation():
        logger.warning("[MEMORY] High fragmentation detected - applying ultra-conservative sizing")
        
        # Use ultra-conservative batch size calculation
        try:
            temp_model = create_model()
            ultra_conservative_batch = get_ultra_conservative_batch_size(temp_model, max_memory_gb=22.0)
            del temp_model
            torch.cuda.empty_cache()
            
            # Use the more conservative of current batch size or ultra-conservative calculation
            optimal_batch_size = min(optimal_batch_size, ultra_conservative_batch)
            logger.info(f"[MEMORY] Ultra-conservative batch size: {ultra_conservative_batch}, Using: {optimal_batch_size}")
        except Exception as e:
            logger.warning(f"[MEMORY] Could not calculate ultra-conservative batch size: {e}")
            optimal_batch_size = max(2, optimal_batch_size // 8)  # Fallback: divide by 8
            logger.info(f"[MEMORY] Fallback: Reduced batch size to: {optimal_batch_size}")
    
    # Final memory check and cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_detailed_memory_usage("pre_training_final_check")
    
    # STEP 6: Full Training
    training_epochs = 20 if args.quick_mode else args.epochs
    
    # Log memory usage before starting full training
    log_detailed_memory_usage("before_full_training")
    
    # Verify AMP configuration
    if torch.cuda.is_available() and not args.no_amp:
        logger.info("[AMP] Mixed precision training enabled - this should provide ~50% memory reduction")
        try:
            # Test AMP scaler creation
            test_scaler = torch.cuda.amp.GradScaler()
            del test_scaler
            logger.info("[AMP] AMP scaler test successful")
        except Exception as e:
            logger.warning(f"[AMP] AMP test failed: {e}")
    else:
        logger.warning("[AMP] Mixed precision training disabled - memory usage will be higher")

    full_train_key = "full_training"
    progress_manager.create_status_updater(full_train_key,
        f"[START] STEP 6: Full OneCycle Training - Starting {training_epochs} epochs...")

    logger.info("="*60)
    logger.info("[START] STEP 6: Full OneCycle Training")
    logger.info("="*60)
    logger.info(f"[MEMORY] Final batch size: {optimal_batch_size}")
    logger.info(f"[MEMORY] Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"[MEMORY] Effective batch size: {optimal_batch_size * gradient_accumulation_steps}")
    if optimal_batch_size <= 8:
        logger.info("[MEMORY] Gradient checkpointing: ENABLED")
    else:
        logger.info("[MEMORY] Gradient checkpointing: DISABLED")

    model = create_model().to(device)
    
    # Log memory usage after model creation
    log_detailed_memory_usage("after_model_creation")
    
    # Enable gradient checkpointing for memory efficiency if batch size is very small
    if optimal_batch_size <= 8:
        logger.info("[MEMORY] Enabling gradient checkpointing for ultra-low batch sizes")
        try:
            # Import checkpoint utilities
            from torch.utils.checkpoint import checkpoint_sequential
            
            # Apply gradient checkpointing to ResNet layers to reduce memory usage
            # This trades computation for memory by recomputing activations during backward pass
            def apply_checkpointing(model):
                """Apply gradient checkpointing to model layers"""
                if hasattr(model, 'layer1'):
                    # ResNet-style model with layer1, layer2, etc.
                    model.layer1 = torch.nn.Sequential(*[torch.nn.utils.checkpoint.checkpoint_wrapper(module) for module in model.layer1])
                    model.layer2 = torch.nn.Sequential(*[torch.nn.utils.checkpoint.checkpoint_wrapper(module) for module in model.layer2])
                    model.layer3 = torch.nn.Sequential(*[torch.nn.utils.checkpoint.checkpoint_wrapper(module) for module in model.layer3])
                    model.layer4 = torch.nn.Sequential(*[torch.nn.utils.checkpoint.checkpoint_wrapper(module) for module in model.layer4])
                    logger.info("[MEMORY] Applied gradient checkpointing to ResNet layers")
                else:
                    # Fallback: try to checkpoint the entire model if it's sequential
                    logger.warning("[MEMORY] Could not identify ResNet layers, attempting sequential checkpointing")
                    if isinstance(model, torch.nn.Sequential):
                        model = torch.nn.utils.checkpoint.checkpoint_wrapper(model)
                        logger.info("[MEMORY] Applied sequential gradient checkpointing")
            
            apply_checkpointing(model)
            logger.info("[MEMORY] Gradient checkpointing enabled - expect ~50% memory reduction")
        except Exception as e:
            logger.warning(f"[MEMORY] Could not enable gradient checkpointing: {e}")
            logger.warning("[MEMORY] Continuing without gradient checkpointing")
    
    trainer = FullTrainer(model, train_loader, val_loader, device, args.output)

    progress_manager.update_status(full_train_key,
        f"[START] STEP 6: Full OneCycle Training - LR: {lr_config['min_lr']:.2e}→{lr_config['max_lr']:.2e}, WD: {best_weight_decay:.2e}, Batch: {optimal_batch_size}")

    # Determine gradient accumulation steps based on batch size to maintain effective batch size
    gradient_accumulation_steps = 1
    if optimal_batch_size <= 4:
        gradient_accumulation_steps = 16  # Effective batch size: 64
        logger.info(f"[GRAD] Ultra-low batch size detected, using aggressive gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 8:
        gradient_accumulation_steps = 8  # Effective batch size: 64
        logger.info(f"[GRAD] Low batch size detected, using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 16:
        gradient_accumulation_steps = 4  # Effective batch size: 64
        logger.info(f"[GRAD] Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 32:
        gradient_accumulation_steps = 2  # Effective batch size: 64
        logger.info(f"[GRAD] Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    else:
        logger.info(f"[GRAD] No gradient accumulation needed (batch size: {optimal_batch_size})")

    history = trainer.train(
        lr_config=lr_config,
        epochs=training_epochs,
        batch_size=optimal_batch_size,
        weight_decay=best_weight_decay,
        save_checkpoints=True,
        early_stopping_patience=15 if not args.quick_mode else 5,
        args=args,
        performance_optimizer=performance_optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    progress_manager.update_status(full_train_key, f"[OK] STEP 6: Full Training Complete - Best Val Acc: {max(history['val_acc']):.2f}%")
    progress_manager.finalize_status(full_train_key)
    
    # Performance Optimization Summary
    if performance_optimizer:
        logger.info("="*60)
        logger.info("[OPTIMIZER] Performance Optimization Summary")
        logger.info("="*60)
        try:
            optimization_summary = performance_optimizer.get_optimization_summary()
            logger.info(optimization_summary)
            
            # Save optimization summary to file
            summary_file = os.path.join(args.output, 'optimization_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(optimization_summary)
            logger.info(f"[SAVE] Optimization summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"[OPTIMIZER] Failed to generate optimization summary: {e}")
    
    # STEP 7: Results Analysis and Plotting
    logger.info("="*60)
    logger.info("[ANALYSIS] STEP 7: Results Analysis")
    logger.info("="*60)
    
    # Create progress bar for results analysis
    analysis_steps = 4  # Plot creation, saving plot, saving JSON, final summary
    progress_manager.create_progress_bar("Results Analysis", analysis_steps)
    
    # Plot training curves
    progress_manager.update_progress(1, {'step': 'Creating plots'})
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc')
    ax2.plot(epochs_range, history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate schedule
    ax3.plot(epochs_range, history['lr'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('OneCycle Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Momentum schedule
    ax4.plot(epochs_range, history['momentum'])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Momentum')
    ax4.set_title('Cyclical Momentum Schedule')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    progress_manager.update_progress(2, {'step': 'Saving training curves plot'})
    plt.savefig(os.path.join(args.output, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final results
    progress_manager.update_progress(3, {'step': 'Saving final results JSON'})
    final_results = {
        'lr_config': lr_config,
        'batch_size': optimal_batch_size,
        'weight_decay': best_weight_decay,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'total_epochs': len(history['train_acc'])
    }
    
    with open(os.path.join(args.output, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Final summary
    progress_manager.update_progress(4, {'step': 'Generating final summary'})
    progress_manager.close_progress_bar()
    
    # =============================================================================
    # TRAINING PIPELINE COMPLETED!
    # =============================================================================
    completion_key = "pipeline_complete"
    progress_manager.create_status_updater(completion_key,
        f"[SUCCESS] ImageNet Training Pipeline Completed! Best Val Acc: {final_results['best_val_acc']:.2f}%")
    progress_manager.finalize_status(completion_key)

    logger.info("[SUCCESS] Pipeline Complete!")
    logger.info("[ANALYSIS] Final Results:")
    logger.info(f"   Best Validation Accuracy: {final_results['best_val_acc']:.2f}%")
    logger.info(f"   Final Training Accuracy: {final_results['final_train_acc']:.2f}%")
    logger.info(f"   Final Validation Accuracy: {final_results['final_val_acc']:.2f}%")
    logger.info(f"   Optimal Batch Size: {optimal_batch_size}")
    logger.info(f"   Best Weight Decay: {best_weight_decay:.2e}")
    logger.info(f"   LR Range: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
    logger.info(f"[DIR] Results saved to: {args.output}")


if __name__ == '__main__':
    import sys
    import os
    import inspect
    
    # =============================================================================
    # SCRIPT EXECUTION CONTEXT LOGGING
    # =============================================================================
    print("=" * 80)
    print("SCRIPT: IMAGENET_TRAINING_PIPELINE.PY CALLED")
    print("=" * 80)
    print(f"PATH: Script Path: {__file__}")
    print(f"DIR:  Working Directory: {os.getcwd()}")
    print(f"PY:   Python Executable: {sys.executable}")
    print(f"ARGS: Command Line Args: {sys.argv}")
    print(f"COUNT: Number of Args: {len(sys.argv)}")
    
    # Show calling context
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        print(f"CALLER: Called From: {caller_frame.f_code.co_filename}:{caller_frame.f_lineno}")
        print(f"FUNC:   Caller Function: {caller_frame.f_code.co_name}")
    else:
        print("CALLER: Called From: Direct execution (no caller frame)")
    
    # Environment context
    print(f"[ENV] Environment Variables (SageMaker related):")
    sm_vars = {k: v for k, v in os.environ.items() if 'SM_' in k or 'SAGEMAKER' in k}
    if sm_vars:
        for key, value in list(sm_vars.items())[:10]:  # Show first 10
            print(f"   {key}: {value}")
        if len(sm_vars) > 10:
            print(f"   ... and {len(sm_vars) - 10} more SM_ variables")
    else:
        print("   No SageMaker environment variables found")
    
    print("=" * 80)
    sys.stdout.flush()
    
    print("[DEBUG] DEBUG: Starting imagenet_training_pipeline.py")
    sys.stdout.flush()
    try:
        print("[DEBUG] DEBUG: About to call main()")
        sys.stdout.flush()
        main()
        print("[DEBUG] DEBUG: main() completed successfully")
        sys.stdout.flush()
    except Exception as e:
        # Setup unified logger for error reporting if main logger fails
        print(f"[DEBUG] DEBUG: Exception caught in __main__: {e}")
        sys.stdout.flush()
        try:
            from logger_setup import setup_unified_logger
            logger = setup_unified_logger()
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] CRITICAL ERROR: Pipeline failed with exception: {e}")
        logger.error(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[ERROR] Full traceback:")
        traceback_lines = traceback.format_exc()
        print(f"[DEBUG] DEBUG: Full traceback (print): {traceback_lines}")
        sys.stdout.flush()
        for line in traceback_lines.split('\n'):
            if line.strip():
                logger.error(f"   {line}")
        print(f"[DEBUG] DEBUG: About to re-raise exception")
        sys.stdout.flush()
        raise
