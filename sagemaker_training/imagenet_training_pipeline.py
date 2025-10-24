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
import logging

from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_dataloaders
from ilsvrc_dataset import get_ilsvrc_dataloaders
from training_performance_optimizer import TrainingPerformanceOptimizer, create_optimized_trainer
from logger_setup import get_unified_logger

# Set memory fragmentation fix BEFORE any PyTorch operations
# Make split size adaptive to GPU memory size for optimal performance
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    try:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Use 2-4% of GPU memory for split size (optimal range for most GPUs)
            split_size_mb = max(128, min(512, int(gpu_memory_gb * 25)))  # 128MB min, 512MB max
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{split_size_mb}'
            print(f"🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{split_size_mb} (adaptive to {gpu_memory_gb:.1f}GB GPU)")
        else:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            print("🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (CPU mode)")
    except Exception as e:
        # Fallback to conservative setting
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        print("🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 (fallback)")

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

    # Get instance profile for optimal worker scaling
    instance_profile = get_instance_resource_profile()
    max_workers_per_core = instance_profile['max_workers_per_core']

    # Base calculation: Use instance-optimized workers per core
    base_workers = max(1, int(cpu_count * max_workers_per_core))

    # Memory-based adjustment (rough heuristic)
    # Adjust memory per worker based on instance type
    if instance_profile['instance_type'] in ['high_end', 'mid_high_end']:
        memory_per_worker_mb = 400  # High-end instances can handle more memory per worker
    elif instance_profile['instance_type'] == 'mid_range':
        memory_per_worker_mb = 500  # Standard memory per worker
    else:
        memory_per_worker_mb = 600  # Conservative for lower-end instances

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


def get_ultra_conservative_batch_size(model, max_memory_gb=8.0):
    """
    Ultra-conservative batch size calculation implementing all 4 OOM prevention strategies:

    1.1 Reduce the Batch Size per GPU ⬇️ - Calculate per-GPU batch size for multi-GPU scenarios
    1.2 Enable Mixed Precision Training 💾 - Use torch.cuda.amp for 50% memory reduction
    1.3 Use Gradient Accumulation 📈 - Simulate larger batches with smaller memory footprint
    1.4 Clear Unused Variables 🧹 - Aggressive memory cleanup and variable deletion

    Args:
        model: PyTorch model to test batch sizes with
        max_memory_gb: Maximum GPU memory available (GB)

    Returns:
        Conservative batch size per GPU that should prevent OOM
    """
    logger = get_unified_logger("ultra_conservative_batch")

    if not torch.cuda.is_available():
        return 2  # Very conservative for CPU

    # Strategy 1.1: Reduce batch size per GPU for multi-GPU scenarios
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # For multi-GPU distributed training, batch size is per GPU
        memory_per_gpu = max_memory_gb / num_gpus
        logger.info(f"[ULTRA] Multi-GPU ({num_gpus} GPUs): {max_memory_gb:.1f}GB total → {memory_per_gpu:.1f}GB per GPU")
    else:
        memory_per_gpu = max_memory_gb
        logger.info(f"[ULTRA] Single GPU: {memory_per_gpu:.1f}GB available")

    # Start with very small batch size and test upwards
    test_batch_sizes = [1, 2, 4, 8, 16, 32]
    max_working_batch = 1

    logger.info("[ULTRA] Testing batch sizes with all OOM prevention strategies...")

    for batch_size in test_batch_sizes:
        try:
            # Strategy 1.4: Clear unused variables before testing
            aggressive_memory_cleanup()

            # Create test data
            dummy_input = torch.randn(batch_size, 3, 224, 224).to('cuda')
            dummy_target = torch.randint(0, 1000, (batch_size,)).to('cuda')

            # Strategy 1.2: Enable mixed precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(dummy_input)
                loss = torch.nn.functional.cross_entropy(outputs, dummy_target)

                # Strategy 1.3: Use gradient accumulation (simulate multiple steps)
                # Scale loss for accumulation (simulate 4 accumulation steps)
                scaled_loss = loss / 4

                # Backward pass
                scaled_loss.backward()

            # Check memory usage
            memory_gb = torch.cuda.memory_allocated() / (1024**3)

            # Strategy 1.4: Clear unused variables immediately
            del dummy_input, dummy_target, outputs, loss, scaled_loss
            aggressive_memory_cleanup()

            # Success! This batch size works
            max_working_batch = batch_size
            logger.info(f"[ULTRA] ✅ Batch size {batch_size}: {memory_gb:.2f}GB - SUCCESS")

            # Continue testing larger sizes
            continue

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"[ULTRA] ❌ Batch size {batch_size} failed (OOM)")
                break  # Stop at first failure
            else:
                # Non-OOM error, re-raise
                raise e

    # Apply ultra-conservative safety factor (50% of max working batch)
    ultra_conservative_batch = max(1, max_working_batch // 2)

    logger.info(f"[ULTRA] Max working batch: {max_working_batch}, Ultra-conservative: {ultra_conservative_batch}")
    logger.info("[ULTRA] All 4 OOM prevention strategies applied successfully")

    return ultra_conservative_batch


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

def get_instance_resource_profile():
    """
    Detect instance type and return optimal resource utilization profile.

    Returns:
        dict: Resource profile with optimal settings for the detected instance
    """
    profile = {
        'gpu_memory_gb': 0,
        'cpu_cores': 0,
        'memory_per_core_gb': 0,
        'instance_type': 'unknown',
        'optimal_batch_memory_fraction': 0.4,  # Conservative default
        'max_workers_per_core': 1,
        'enable_gradient_checkpointing': False,
        'gradient_accumulation_base': 1
    }

    try:
        import multiprocessing as mp

        # Detect CPU resources
        profile['cpu_cores'] = mp.cpu_count()

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            profile['gpu_memory_gb'] = gpu_props.total_memory / (1024**3)

            # Classify instance type based on GPU memory and CPU cores
            gpu_mem = profile['gpu_memory_gb']
            cpu_cores = profile['cpu_cores']

            # AWS SageMaker instance classification
            if gpu_mem >= 80:  # A100, H100 instances
                profile['instance_type'] = 'high_end'
                profile['optimal_batch_memory_fraction'] = 0.5  # Can use more memory
                profile['max_workers_per_core'] = 1.5
                profile['enable_gradient_checkpointing'] = False  # Don't need for large memory
                profile['gradient_accumulation_base'] = 1
            elif gpu_mem >= 40:  # V100, A10G instances
                profile['instance_type'] = 'mid_high_end'
                profile['optimal_batch_memory_fraction'] = 0.45
                profile['max_workers_per_core'] = 1.2
                profile['enable_gradient_checkpointing'] = False
                profile['gradient_accumulation_base'] = 1
            elif gpu_mem >= 20:  # T4, M60 instances (like ml.g6.12xlarge)
                profile['instance_type'] = 'mid_range'
                profile['optimal_batch_memory_fraction'] = 0.4
                profile['max_workers_per_core'] = 1.0
                profile['enable_gradient_checkpointing'] = True  # Enable for smaller batches
                profile['gradient_accumulation_base'] = 2
            elif gpu_mem >= 8:  # GTX 1080, smaller instances
                profile['instance_type'] = 'entry_level'
                profile['optimal_batch_memory_fraction'] = 0.35
                profile['max_workers_per_core'] = 0.8
                profile['enable_gradient_checkpointing'] = True
                profile['gradient_accumulation_base'] = 4
            else:  # Very small GPUs
                profile['instance_type'] = 'low_end'
                profile['optimal_batch_memory_fraction'] = 0.3
                profile['max_workers_per_core'] = 0.5
                profile['enable_gradient_checkpointing'] = True
                profile['gradient_accumulation_base'] = 8

            # Adjust based on CPU cores (for multi-GPU instances)
            if cpu_cores >= 64:  # High CPU core count instances
                profile['max_workers_per_core'] *= 1.2
            elif cpu_cores >= 32:
                profile['max_workers_per_core'] *= 1.1
            elif cpu_cores <= 8:  # Low CPU core instances
                profile['max_workers_per_core'] *= 0.8

        else:
            profile['instance_type'] = 'cpu_only'
            profile['optimal_batch_memory_fraction'] = 0.2  # Very conservative for CPU

        # Calculate memory per core
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            profile['memory_per_core_gb'] = total_memory_gb / cpu_cores
        except:
            profile['memory_per_core_gb'] = 4.0  # Default assumption

    except Exception as e:
        print(f"⚠️  Warning: Could not detect instance profile: {e}")
        # Keep default conservative settings

    return profile

def get_ultra_conservative_batch_size(model, input_shape=(3, 224, 224), device='cuda', max_memory_gb=22.0):
    """
    Calculate ultra-conservative batch size that leaves significant memory buffer.
    Uses instance-aware optimization for different hardware profiles.
    For multi-GPU instances, calculates batch size per GPU.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Device to test on
        max_memory_gb: Maximum GPU memory in GB per GPU

    Returns:
        int: Ultra-conservative batch size per GPU
    """
    if not torch.cuda.is_available():
        return 4  # Conservative for CPU

    logger = get_unified_logger("batch_size_calc")

    # Detect number of GPUs for distributed training
    num_gpus = torch.cuda.device_count()
    logger.info(f"[BATCH_CALC] Detected {num_gpus} GPU(s) available")

    # For multi-GPU training, batch size is per GPU
    # The total effective batch size will be batch_size * num_gpus * gradient_accumulation_steps
    if num_gpus > 1:
        memory_per_gpu = max_memory_gb  # max_memory_gb is already per GPU
        logger.info(f"[BATCH_CALC] Multi-GPU training: calculating batch size per GPU ({memory_per_gpu:.1f}GB per GPU)")
    else:
        memory_per_gpu = max_memory_gb
        logger.info(f"[BATCH_CALC] Single GPU training: using full GPU memory ({memory_per_gpu:.1f}GB)")

    # Get instance profile for optimal resource utilization
    instance_profile = get_instance_resource_profile()
    optimal_fraction = instance_profile['optimal_batch_memory_fraction']

    # Use instance-optimized memory fraction per GPU
    available_for_batch_per_gpu = memory_per_gpu * optimal_fraction
    logger.info(f"[BATCH_CALC] Instance type: {instance_profile['instance_type']} ({instance_profile['gpu_memory_gb']:.1f}GB GPU, {instance_profile['cpu_cores']} CPU cores)")
    logger.info(f"[BATCH_CALC] Using {optimal_fraction:.0%} of GPU memory per GPU ({available_for_batch_per_gpu:.1f}GB) for batch processing")

    # Adjust memory estimate based on instance type
    if instance_profile['instance_type'] in ['high_end', 'mid_high_end']:
        estimated_mb_per_sample = 3.0  # Less conservative for high-end GPUs
    elif instance_profile['instance_type'] == 'mid_range':
        estimated_mb_per_sample = 4.0  # Moderate for mid-range
    else:
        estimated_mb_per_sample = 5.0  # Conservative for entry-level

    # Calculate maximum batch size per GPU
    max_batch_per_gpu = int((available_for_batch_per_gpu * 1024) / estimated_mb_per_sample)

    # Apply instance-aware safety factors
    if instance_profile['instance_type'] == 'high_end':
        conservative_batch = max(1, min(max_batch_per_gpu // 2, 16))  # Less conservative
    elif instance_profile['instance_type'] == 'mid_high_end':
        conservative_batch = max(1, min(max_batch_per_gpu // 3, 12))
    elif instance_profile['instance_type'] == 'mid_range':
        conservative_batch = max(1, min(max_batch_per_gpu // 4, 8))
    else:  # entry_level, low_end
        conservative_batch = max(1, min(max_batch_per_gpu // 6, 4))  # Very conservative

    logger.info(f"[BATCH_CALC] Estimated max batch per GPU: {max_batch_per_gpu}, Instance-optimized batch: {conservative_batch}")
    logger.info(f"[BATCH_CALC] Memory estimate: {estimated_mb_per_sample:.1f}MB per sample, {available_for_batch_per_gpu:.1f}GB available per GPU")
    if num_gpus > 1:
        total_effective_memory = available_for_batch_per_gpu * num_gpus
        logger.info(f"[BATCH_CALC] Total effective memory across {num_gpus} GPUs: {total_effective_memory:.1f}GB")

    return conservative_batch


def get_adaptive_gradient_accumulation_steps(batch_size, target_effective_batch=256):
    """
    Calculate gradient accumulation steps for effective batch size scaling
    """
    if batch_size >= target_effective_batch:
        return 1  # No accumulation needed

    accumulation_steps = max(1, target_effective_batch // batch_size)
    return min(accumulation_steps, 8)  # Cap at 8 to prevent too much accumulation


def create_oom_resilient_trainer(model, optimizer, criterion, device, max_retries=3):
    """
    Create a training wrapper that automatically handles OOM errors by reducing batch size
    """
    logger = get_unified_logger("oom_trainer")

    class OOMResilientTrainer:
        def __init__(self, model, optimizer, criterion, device, max_retries=3):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.max_retries = max_retries
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            self.oom_count = 0

        def train_step(self, inputs, targets, gradient_accumulation_steps=1):
            """Single training step with OOM resilience"""
            retry_count = 0
            current_accumulation = gradient_accumulation_steps

            while retry_count <= self.max_retries:
                try:
                    # Memory cleanup before step
                    if retry_count > 0:
                        aggressive_memory_cleanup()
                        logger.warning(f"🔄 OOM retry {retry_count}/{self.max_retries} - reducing accumulation to {current_accumulation}")

                    # Training step with mixed precision if available
                    if self.scaler and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)

                        # Scale loss for accumulation
                        self.scaler.scale(loss / current_accumulation).backward()
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        (loss / current_accumulation).backward()

                    return outputs, loss.item()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.oom_count += 1
                        retry_count += 1
                        current_accumulation = max(1, current_accumulation // 2)

                        if retry_count > self.max_retries:
                            logger.error(f"❌ Max OOM retries exceeded. Final error: {e}")
                            raise e
                    else:
                        # Not an OOM error, re-raise
                        raise e

            return None, None

        def optimizer_step(self):
            """Perform optimizer step with gradient clipping and scaler update"""
            if self.scaler and torch.cuda.is_available():
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard gradient clipping and step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()

    return OOMResilientTrainer(model, optimizer, criterion, device, max_retries)


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
              gradient_accumulation_steps=1, enable_oom_protection=True):
        """Full training with OneCycle LR and cyclical momentum"""
        
        logger = get_unified_logger()
        
        # Calculate adaptive gradient accumulation if not specified
        if gradient_accumulation_steps == 1:
            gradient_accumulation_steps = get_adaptive_gradient_accumulation_steps(batch_size)
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        logger.info("[START] Starting Full Training:")
        logger.info(f"   📚 Epochs: {epochs}")
        logger.info(f"   [SIZE] LR Range: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
        logger.info(f"   [WEIGHT]  Weight Decay: {weight_decay:.2e}")
        logger.info(f"   [BATCH] Batch Size: {batch_size}")
        logger.info(f"   [GRAD] Gradient Accumulation Steps: {gradient_accumulation_steps}")
        if gradient_accumulation_steps > 1:
            logger.info(f"   [EFFECTIVE] Effective Batch Size: {effective_batch_size}")
        logger.info(f"   [OOM] OOM Protection: {'Enabled' if enable_oom_protection else 'Disabled'}")
        
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
            train_loss, train_acc = self._train_epoch(optimizer, criterion, scheduler, scaler, performance_optimizer, gradient_accumulation_steps, enable_oom_protection)

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
    
    def _train_epoch(self, optimizer, criterion, scheduler, scaler=None, performance_optimizer=None, gradient_accumulation_steps=1, enable_oom_protection=True):
        """Train one epoch with optional mixed precision, performance optimization, gradient accumulation, and OOM protection"""
        logger = get_unified_logger()
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create OOM resilient trainer if protection is enabled and no performance optimizer
        if enable_oom_protection and not performance_optimizer:
            oom_trainer = create_oom_resilient_trainer(self.model, optimizer, criterion, self.device)
            accumulation_counter = 0
        else:
            oom_trainer = None
            accumulation_counter = 0
        
        # Use progress manager for clean progress tracking
        progress_manager.create_progress_bar("Training", len(self.train_loader))
        
        step_count = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs, targets = batch_data
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            step_count += 1
            
            try:
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
                elif oom_trainer:
                    # Use OOM resilient training
                    if accumulation_counter == 0:
                        optimizer.zero_grad()
                    
                    outputs, loss_value = oom_trainer.train_step(inputs, targets, gradient_accumulation_steps)
                    accumulation_counter += 1
                    
                    # Perform optimizer step after accumulation
                    if accumulation_counter >= gradient_accumulation_steps:
                        oom_trainer.optimizer_step()
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
                
                # Update metrics
                if performance_optimizer:
                    running_loss += loss
                else:
                    running_loss += loss_value if oom_trainer else loss.item()
                
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
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and not oom_trainer:
                    logger.error(f"❌ OOM in training step {step_count}: {e}")
                    logger.info("💡 Consider enabling OOM protection or reducing batch_size/gradient_accumulation_steps")
                    aggressive_memory_cleanup()
                    raise e
                else:
                    raise e
        
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
    logger.info(f"[PYTHON] Python version: {sys.version}")
    logger.info(f"[PYTORCH] PyTorch version: {torch.__version__}")
    logger.info(f"[SYSTEM] Working directory: {os.getcwd()}")
    logger.info(f"[FILE] Script path: {sys.argv[0]}")
    logger.info(f"[ARGS] Command line args: {sys.argv[1:] if len(sys.argv) > 1 else 'None'}")
    
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
    parser.add_argument('--skip-lr-test', action='store_true', help='Skip LR range test')
    parser.add_argument('--skip-wd-search', action='store_true', help='Skip weight decay search')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_unified_logger('imagenet_pipeline')
    
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
            
            # Use optimizer's batch size detection method with actual GPU memory
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                actual_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if num_gpus > 1:
                    # For multi-GPU training, divide memory per GPU for batch size calculation
                    max_memory_gb = (actual_gpu_memory_gb * 0.7) / num_gpus
                    logger.info(f"[BATCH] Multi-GPU ({num_gpus} GPUs): Using {max_memory_gb:.1f}GB per GPU ({max_memory_gb*num_gpus:.1f}GB total) of detected {actual_gpu_memory_gb:.1f}GB per GPU")
                else:
                    # Single GPU: use 70% of available memory
                    max_memory_gb = actual_gpu_memory_gb * 0.7
                    logger.info(f"[BATCH] Single GPU: Using {max_memory_gb:.1f}GB ({max_memory_gb/actual_gpu_memory_gb:.0%}) of detected {actual_gpu_memory_gb:.1f}GB GPU memory")
            else:
                max_memory_gb = 4.0  # Conservative for CPU
                logger.info(f"[BATCH] CPU mode: Using {max_memory_gb:.1f}GB memory limit")

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
        
        # Check for multi-GPU scenario and adjust batch size per GPU if needed
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # For multi-GPU distributed training, batch size should be per GPU
                per_gpu_batch_size = max(1, initial_batch_size // (num_gpus))
                logger.warning(f"[MULTI-GPU] Specified batch size {initial_batch_size} detected with {num_gpus} GPUs")
                logger.warning(f"[MULTI-GPU] Adjusting to {per_gpu_batch_size} per GPU (total effective: {per_gpu_batch_size * num_gpus})")
                logger.warning(f"[MULTI-GPU] This ensures proper memory distribution across GPUs")
                initial_batch_size = per_gpu_batch_size
    
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
        
        # Use ultra-conservative batch size calculation with actual GPU memory
        try:
            temp_model = create_model()
            if torch.cuda.is_available():
                actual_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                actual_gpu_memory_gb = 8.0  # Conservative CPU estimate

            ultra_conservative_batch = get_ultra_conservative_batch_size(temp_model, max_memory_gb=actual_gpu_memory_gb)
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
    
    # Enable gradient checkpointing based on instance profile and batch size
    instance_profile = get_instance_resource_profile()
    should_checkpoint = instance_profile['enable_gradient_checkpointing'] or optimal_batch_size <= 8

    if should_checkpoint:
        logger.info("[MEMORY] Enabling gradient checkpointing for memory efficiency")
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

    # Determine gradient accumulation steps based on batch size and instance profile
    instance_profile = get_instance_resource_profile()
    base_accumulation = instance_profile['gradient_accumulation_base']

    gradient_accumulation_steps = 1
    if optimal_batch_size <= 4:
        gradient_accumulation_steps = base_accumulation * 4  # Instance-aware scaling
        logger.info(f"[GRAD] Ultra-low batch size detected, using aggressive gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 8:
        gradient_accumulation_steps = base_accumulation * 2
        logger.info(f"[GRAD] Low batch size detected, using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 16:
        gradient_accumulation_steps = base_accumulation
        logger.info(f"[GRAD] Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
    elif optimal_batch_size <= 32:
        gradient_accumulation_steps = max(1, base_accumulation // 2)
        if gradient_accumulation_steps > 1:
            logger.info(f"[GRAD] Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {optimal_batch_size * gradient_accumulation_steps})")
        else:
            logger.info(f"[GRAD] No gradient accumulation needed (batch size: {optimal_batch_size})")
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Setup unified logger for error reporting if main logger fails
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
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"   {line}")
        raise
