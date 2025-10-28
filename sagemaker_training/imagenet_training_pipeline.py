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
from tqdm.auto import tqdm
import csv
import argparse
from datetime import datetime
import gc
import logging
import copy
from torch.cuda.amp import autocast, GradScaler
import psutil
import multiprocessing as mp
import math
from utils import is_main_process
from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_dataloaders
from training_performance_optimizer import TrainingPerformanceOptimizer
from logger_setup import get_unified_logger
import platform
import subprocess
import threading
import sys
try:
    import GPUtil
except ImportError:
    GPUtil = None

# Standard ImageNet Constants
IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000

def save_pipeline_status(stage_name, status_file):
    if is_main_process():
        status = {
            'last_completed_stage': stage_name,
            'timestamp': datetime.now().isoformat()
        }
        with open(status_file, 'w') as f:
            json.dump(status, f)

TQDM_DISABLE = os.environ.get("TQDM_DISABLE", "0") == "1"

def get_hardware_summary():
    summary = {}
    summary['system'] = platform.system()
    summary['node_name'] = platform.node()
    summary['release'] = platform.release()
    summary['version'] = platform.version()
    summary['machine'] = platform.machine()
    summary['processor'] = platform.processor()
    summary['cpu_cores'] = psutil.cpu_count(logical=False)
    summary['logical_cpus'] = psutil.cpu_count(logical=True)
    summary['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)

    # GPU details
    try:
        import torch
        if torch.cuda.is_available():
            summary['cuda_device_count'] = torch.cuda.device_count()
            gpus = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                }
                gpus.append(gpu_info)
            summary['gpus'] = gpus
        else:
            summary['gpus'] = 'No CUDA GPUs detected.'
    except Exception as e:
        summary['gpus'] = f'Error getting GPU info: {e}'

    # NVIDIA SMI output
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        summary['nvidia_smi'] = result.stdout.strip()
    except Exception as e:
        summary['nvidia_smi'] = f'Error running nvidia-smi: {e}'

    return summary


# Advanced non-blocking resource monitor
class ResourceMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.metrics = []
        self.running = False
        self.thread = None

    def _collect(self):
        while self.running:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'ram_gb': psutil.virtual_memory().used / (1024**3),
            }
            # GPU monitoring (multi-GPU)
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    metric['gpus'] = [{
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'mem_used': gpu.memoryUsed,
                        'mem_total': gpu.memoryTotal,
                        'mem_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    } for gpu in gpus]
                except Exception:
                    metric['gpus'] = []
            else:
                metric['gpus'] = []
            self.metrics.append(metric)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# -----------------------------
# CSV/JSON Logging Utilities
# -----------------------------
def log_metrics_csv(file_path, fieldnames, row):
    """Append a row of metrics to a CSV file."""
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def log_metrics_json(file_path, data):
    """Append a dict of metrics to a JSON file (list of dicts)."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    else:
        existing = []
    existing.append(data)
    with open(file_path, 'w') as f:
        json.dump(existing, f, indent=2)

def log_config_json(file_path, config):
    """Save config/hyperparameters to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
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
    except Exception:
        # Fallback to conservative setting
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        print("🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 (fallback)")
os.environ['SMDATAPARALLEL_OPTIMIZE_SDP'] = 'true'
os.environ['PYTHONUNBUFFERED'] = '1'

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
        except Exception:
            profile['memory_per_core_gb'] = 4.0  # Default assumption

    except Exception as e:
        print(f"⚠️  Warning: Could not detect instance profile: {e}")
        # Keep default conservative settings

    return profile

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
            self.scaler = GradScaler(enabled=torch.cuda.is_available())
            self.oom_count = 0
            self.enable_amp = self.scaler.enabled

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
                        with autocast(enabled=self.enable_amp):
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
    """Progress manager using tqdm for console progress bars and status updates."""
    def __init__(self):
        self.current_bar = None
        self.status = {}

    def create_progress_bar(self, desc, total):
        self.close_progress_bar()  # Close any previous bar
        from tqdm.auto import tqdm
        self.current_bar = tqdm(total=total, desc=desc, ncols=120, disable=TQDM_DISABLE)

    def update_progress(self, n, metrics=None):
        if self.current_bar:
            self.current_bar.n = n
            if metrics:
                formatted_metrics = {
                    k: f"{v:.4f}" if isinstance(v, float) else v 
                    for k, v in metrics.items()
                }
            
                self.current_bar.set_postfix(formatted_metrics, refresh=False)
                # Compose description from metrics dict
                #desc = self.current_bar.desc.split('|')[0].strip()
                #metrics_str = ' | '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                #self.current_bar.set_description(f"{desc} | {metrics_str}")
            self.current_bar.refresh()

    def close_progress_bar(self):
        if self.current_bar:
            self.current_bar.close()
            self.current_bar = None

    def create_status_updater(self, key, message):
        self.status[key] = message
        print(f"[STATUS] {message}")

    def update_status(self, key, message):
        self.status[key] = message
        print(f"[STATUS] {message}")

    def finalize_status(self, key):
        msg = self.status.get(key, None)
        if msg:
            print(f"[FINALIZED] {msg}")
        self.status.pop(key, None)

# Global progress manager instance
progress_manager = LiveProgressManager()


# ----------------------------------------------------------------------
# I. NUM_WORKERS OPTIMIZATION (Corrected)
# ----------------------------------------------------------------------

def optimize_num_workers(batch_size: int, available_memory_gb: float = None, cpu_count: int = None) -> int:
    """
    Optimizes num_workers for DataLoader based on system resources and batch size.
    Removes arbitrary caps and unknown instance profile dependencies.
    """
    logger = get_unified_logger("num_workers_optimizer")
    
    if cpu_count is None:
        cpu_count = mp.cpu_count()
    if available_memory_gb is None:
        # Use available memory, not total memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Base Limit: Use all but one core to keep the main process free.
    base_workers = max(1, cpu_count - 1)

    # Memory Limit: Assume a conservative 0.5GB (500MB) of RAM consumption per worker.
    MEMORY_PER_WORKER_GB = 0.5
    max_workers_by_memory = math.floor(available_memory_gb / MEMORY_PER_WORKER_GB)
    max_workers_by_memory = max(1, max_workers_by_memory)

    # Batch Size Adjustment: Large batches put more pressure on individual workers.
    if batch_size >= 256:
        batch_factor = 0.5
    elif batch_size >= 128:
        batch_factor = 0.7
    elif batch_size >= 64:
        batch_factor = 0.9
    else:
        batch_factor = 1.0

    # Optimal workers is the minimum of CPU potential and Memory limit, adjusted by batch size.
    optimal_workers = min(base_workers, max_workers_by_memory)
    final_workers = int(optimal_workers * batch_factor)
    
    # Do NOT cap at 8. Ensure at least 1.
    final_workers = max(1, final_workers)
    
    logger.info(f"💾 Workers | CPU Cores: {cpu_count}, Available RAM: {available_memory_gb:.1f}GB")
    logger.info(f"💾 Workers | Base (CPU): {base_workers}, Max (RAM): {max_workers_by_memory}")
    logger.info(f"💾 Workers | Optimal num_workers: {final_workers}")

    return final_workers

# ----------------------------------------------------------------------
# II. BATCH SIZE OPTIMIZATION (Corrected)
# ----------------------------------------------------------------------

class BatchSizeFinder:
    def __init__(self, model, optimizer, criterion, device, enable_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.enable_amp = torch.cuda.is_available() and enable_amp
        self.scaler = GradScaler(enabled=enable_amp)
        
    def _calculate_max_memory_gb(self, max_memory_gb_limit: float) -> float:
        """Determines the effective memory ceiling based on actual GPU VRAM (Corrected)."""
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            actual_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # The memory limit should be based on a single GPU's VRAM, not divided by num_gpus.
            # Use 70% as a safety buffer.
            max_memory_gb = actual_gpu_memory_gb * 0.7
            max_memory_gb = min(max_memory_gb, max_memory_gb_limit) # Use user's specified limit if lower
            
            logger.info(f"🔍 BATCH | GPU VRAM: {actual_gpu_memory_gb:.1f}GB. Effective limit: {max_memory_gb:.1f}GB.")
        else:
            max_memory_gb = max_memory_gb_limit
            logger.info(f"🔍 BATCH | CPU mode. Using specified limit: {max_memory_gb:.1f}GB.")
        return max_memory_gb

    def get_optimal_batch_size(self, max_memory_gb: float = 14.0, 
                               quick_mode: bool = False) -> int:
        """
        Find optimal batch size by iterating until CUDA OOM error (Corrected).
        """
        max_limit = self._calculate_max_memory_gb(max_memory_gb)
        
        base_batch_size = 32
        optimal_batch = base_batch_size
        
        # Test powers of 2 for GPU efficiency
        batch_sizes_to_test = [32, 64, 128, 256, 512, 1024, 2048, 4096] 

        logger.info(f"🔍 BATCH | Finding optimal batch size (max {max_limit:.1f}GB memory)")
        
        for batch_size in batch_sizes_to_test:
            if batch_size <= optimal_batch: # Skip smaller batches if already successful
                continue
                
            try:
                # Cleanup before new test
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                
                # Test memory usage with dummy batch (standard ImageNet size 224x224)
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                dummy_target = torch.randint(0, 1000, (batch_size,)).to(self.device)

                with autocast(enabled=self.enable_amp):
                    output = self.model(dummy_input)
                    loss = self.criterion(output, dummy_target)
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Check allocated memory AFTER backward pass
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                
                # Manual memory check is removed; rely on OOM or max_limit
                if memory_gb > max_limit:
                    logger.warning(f"  ❌ Batch size {batch_size}: {memory_gb:.1f}GB exceeds limit. Stopping.")
                    break
                    
                optimal_batch = batch_size
                logger.info(f"  ✅ Batch size {batch_size}: {memory_gb:.1f}GB")

                # Clear tensors and cache (critical for successful loop iteration)
                del dummy_input, dummy_target, output, loss
                if self.scaler:
                     self.scaler.unscale_(self.optimizer) # Necessary if using scaler
                self.optimizer.step() # Perform step to test optimizer state
                self.optimizer.zero_grad() 
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"  ❌ Batch size {batch_size} failed: CUDA Out of Memory.")
                    # Critical cleanup after OOM
                    #del dummy_input, dummy_target, output, loss
                    torch.cuda.empty_cache() 
                    break
                else:
                    logger.warning(f"  ❌ Batch size {batch_size} failed with non-OOM error: {e}")
                    break
            except NameError:
                 # Catch error if deletion fails on objects that weren't created
                 pass


        # --- Safety Factor and Power of 2 Adjustment (Corrected) ---
        safety_factor = 0.5 if quick_mode else 0.8
        safe_batch_size = int(optimal_batch * safety_factor)
        
        # Round down to the nearest Power of 2 (ensures max GPU efficiency)
        if safe_batch_size > 0:
            final_batch_size = max(1, 2 ** int(np.floor(np.log2(safe_batch_size))))
        else:
            final_batch_size = 1
            
        logger.info(f"🎯 BATCH | Optimal batch found: {optimal_batch}. Final training size: {final_batch_size}")
        
        return final_batch_size

# ----------------------------------------------------------------------
# III. LEARNING RATE FINDER (Corrected with State Restoration)
# ----------------------------------------------------------------------

class LRFinder:
    def suggest_lr(self):
        """
        Suggest min_lr and max_lr based on LR range test loss curve.
        Returns a dict: {'min_lr': ..., 'max_lr': ...}
        """
        lrs = self.history.get('lr', [])
        losses = self.history.get('loss', [])
        if not lrs or not losses:
            return {'min_lr': 1e-4, 'max_lr': 1e-2}
        import numpy as np
        losses = np.array(losses)
        lrs = np.array(lrs)
        min_idx = int(np.argmin(losses))
        min_lr = float(lrs[min_idx])
        # Heuristic: max_lr is where loss starts increasing rapidly after min
        # Find first index after min_idx where loss increases by >30%
        max_lr = min_lr
        for i in range(min_idx+1, len(losses)):
            if losses[i] > losses[min_idx] * 1.3:
                max_lr = float(lrs[i])
                break
        if max_lr == min_lr:
            max_lr = float(lrs[-1])
        return {'min_lr': min_lr, 'max_lr': max_lr}
    def plot(self):
        """
        Plot LR vs Loss curve and return matplotlib figure and min_lr suggestion.
        Does NOT call plt.show(), so it is non-blocking.
        """
        import matplotlib.pyplot as plt
        lrs = self.history.get('lr', [])
        losses = self.history.get('loss', [])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses, label='LR Range Test')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Smoothed Loss')
        ax.set_title('LR Range Test: LR vs Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Suggest min_lr as LR with lowest loss
        if losses:
            min_idx = int(np.argmin(losses))
            min_lr = lrs[min_idx] if min_idx < len(lrs) else None
        else:
            min_lr = None
        return fig, min_lr
    """Learning Rate Range Test Implementation with state restoration."""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def range_test(self, dataloader, start_lr=1e-7, end_lr=1, num_iter=100, smooth_factor=0.05):
        """Perform LR range test with state restoration."""
        logger = get_unified_logger("lr_range_test")
        # Setup logging files
        output_dir = './imagenet_pipeline_results'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'lr_range_log.csv')
        json_path = os.path.join(output_dir, 'lr_range_log.json')
        csv_fields = ['iteration', 'lr', 'smoothed_loss']
        
        # --- CRITICAL: Save Initial State ---
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        initial_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        
        # Calculate multiplicative factor
        lr_lambda = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
            
        self.model.train()
        losses = []
        lrs = []
        best_loss = float('inf')
        
        bar = tqdm(total=num_iter, desc="LR Range Test", unit="it", ncols=120, disable=TQDM_DISABLE)
        data_iter = iter(dataloader)

        for i in range(num_iter):
            # With .repeat() in WebDataset, StopIteration should never occur naturally
            # But keep error handling for safety
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                logger.error("Unexpected StopIteration - dataloader may be empty or corrupted")
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
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

            # Log metrics for this iteration
            log_metrics_csv(csv_path, csv_fields, {
                'iteration': i+1,
                'lr': current_lr,
                'smoothed_loss': smoothed_loss
            })
            log_metrics_json(json_path, {
                'iteration': i+1,
                'lr': current_lr,
                'smoothed_loss': smoothed_loss
            })
            
            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss or torch.isnan(loss):
                logger.warning(f"[ERROR] Stopping early at iteration {i}, loss exploded")
                break
                
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Memory cleanup
            del outputs, loss, inputs, targets
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_lambda
                
            # Update progress
            bar.n = i + 1
            bar.set_description(f"LR Range Test | LR: {current_lr:.2e} | Loss: {smoothed_loss:.4f}")
            bar.refresh()
            logger.info(f"lr_range_test | Iter {i+1}/{num_iter} | LR: {current_lr:.2e} | Loss: {smoothed_loss:.4f}")
            
        bar.close()
        
        # --- CRITICAL: Restore Initial State ---
        logger.info("[DEBUG] Restoring model and optimizer states...")
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = initial_lrs[i]

        self.history['lr'] = lrs
        self.history['loss'] = losses
        
        return lrs, losses


# ----------------------------------------------------------------------
# IV. WEIGHT DECAY FINDER (Skeleton)
# ----------------------------------------------------------------------

class HyperparameterOptimizer:
    """Grid/Random search for hyperparameters"""
    
    def __init__(self, model_fn, train_loader, val_loader, device, train_batches_per_epoch=None, val_batches_per_epoch=None):
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        
    def _quick_train(self, model, optimizer, criterion, scheduler, epochs):
        """Quick training for hyperparameter search"""
        train_losses = []
        val_losses = []
        val_accs = []
        logger = get_unified_logger("WeightDecaySearch - QuickTrain")
        # Setup logging files
        output_dir = './imagenet_pipeline_results'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'wd_quicktrain_log.csv')
        json_path = os.path.join(output_dir, 'wd_quicktrain_log.json')
        csv_fields = ['epoch', 'train_loss', 'val_loss', 'val_acc']
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            # Use progress manager for clean progress tracking
            total_batches = self.train_batches_per_epoch   # Calculate effective epoch size for the scheduler. This is critical.
            from tqdm.auto import tqdm
            bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="it", ncols=120, disable=TQDM_DISABLE)
            
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
                bar.n = batch_idx + 1
                bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/train_batches:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                bar.refresh()
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"weight_decay_search quick train | Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | Loss: {train_loss/train_batches:.4f}")
            
            bar.close()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0
            
            # Use progress manager for validation progress
            val_limit = self.val_batches_per_epoch  # Limit validation batches for speed
            from tqdm.auto import tqdm
            bar = tqdm(total=val_limit, desc=f"Validation {epoch+1}/{epochs}", unit="it", ncols=120, disable=TQDM_DISABLE)
            
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
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"weight_decay_search quick valid | Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | Loss: {val_loss/val_batches:.4f}")
                    # Limit validation batches for speed
                    if val_batches >= 50:
                        break

            bar.close()
            
            train_losses.append(train_loss / train_batches)
            val_losses.append(val_loss / val_batches)
            val_accs.append(100. * val_correct / val_total)

            # Log metrics for this epoch
            log_metrics_csv(csv_path, csv_fields, {
                'epoch': epoch+1,
                'train_loss': train_loss / train_batches if train_batches else 0,
                'val_loss': val_loss / val_batches if val_batches else 0,
                'val_acc': 100. * val_correct / val_total if val_total else 0
            })
            log_metrics_json(json_path, {
                'epoch': epoch+1,
                'train_loss': train_loss / train_batches if train_batches else 0,
                'val_loss': val_loss / val_batches if val_batches else 0,
                'val_acc': 100. * val_correct / val_total if val_total else 0
            })
            
            # Clean up GPU memory to prevent OOM accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        return train_losses, val_losses, val_accs 

    def weight_decay_search(self, lr_config, batch_size, wd_values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], epochs=5):
        """Search for optimal weight decay (Corrected, requires _quick_train)."""
        logger = get_unified_logger("WeightDecaySearch")
        
        results = []
        bar = tqdm(total=len(wd_values), desc="Weight Decay Search", unit="it", ncols=120, disable=TQDM_DISABLE)
        
        for idx, wd in enumerate(wd_values):
            logger.info(f"[ANALYSIS] Testing Weight Decay: {wd:.2e} ({idx+1}/{len(wd_values)})")
            
            # Create fresh model for each run (CRITICAL)
            model = self.model_fn().to(self.device)
            
            # Use SGD as it is common, momentum=0.9, nesterov=True is typical for CV
            optimizer = optim.SGD(model.parameters(), lr=lr_config['min_lr'], 
                                momentum=0.9, weight_decay=wd, nesterov=True)
            criterion = nn.CrossEntropyLoss()
            
            # OneCycle scheduler is essential for fast convergence in short runs
            scheduler = OneCycleLR(optimizer, max_lr=lr_config['max_lr'], 
                                epochs=epochs, steps_per_epoch=self.train_batches_per_epoch)
            
            # Train for a few epochs
            train_losses, val_losses, val_accs = self._quick_train(
                model, optimizer, criterion, scheduler, epochs)

            logger.info(f"weight_decay_search | Iter {idx+1}/{len(wd_values)} | wd: {wd:.2e} | max accuracy: {max(val_accs):.4f}")
            # Store results
            result = {
                'weight_decay': wd,
                'best_val_acc': max(val_accs),
            }
            results.append(result)
            
            # Logging and progress update...
            
            # Clean up GPU memory (CRITICAL)
            del model, optimizer, criterion, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        bar.close()
        
        # Find best weight decay based on best validation accuracy
        best_result = max(results, key=lambda x: x['best_val_acc'])
        logger.info(f"[COMPLETE] Best Weight Decay: {best_result['weight_decay']:.2e} "
                    f"(Val Acc: {best_result['best_val_acc']:.2f}%)")
        
        return results, best_result['weight_decay']



## Removed duplicate get_adaptive_gradient_accumulation_steps

# --- DUMMY AGGRESSIVE MEMORY CLEANUP (Needed for OOM handling) ---
## Removed duplicate aggressive_memory_cleanup

# ----------------------------------------------------------------------
# FULL TRAINER (CORRECTED)
# ----------------------------------------------------------------------

class FullTrainer:
    """Full training with OneCycleLR, AMP, Gradient Accumulation, and Checkpointing."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir, train_batches_per_epoch=None, val_batches_per_epoch=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.enable_amp = torch.cuda.is_available()
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'momentum': []
        }
        
    def train(self, 
              lr_config, 
              epochs, 
              batch_size, 
              weight_decay=1e-4, 
              save_checkpoints=True,
              start_epoch=0, 
              early_stopping_patience=10, 
              args=None, 
              gradient_accumulation_steps=1):
        """
        Full training run. Simplified by removing external performance_optimizer
        and internal OOM trainers, focusing on standard PyTorch best practices.
        """
        
        logger = get_unified_logger("FullTrainer")
        
        # Setup logging files (once per run)
        output_dir = self.save_dir if self.save_dir else './imagenet_pipeline_results'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'train_log.csv')
        json_path = os.path.join(output_dir, 'train_log.json')
        csv_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'momentum', 'timestamp']
        
        # --- 1. Resource and Hyperparameter Setup ---
        # Calculate adaptive gradient accumulation if not specified (uses dummy if not defined)
        if gradient_accumulation_steps == 1:
            gradient_accumulation_steps = get_adaptive_gradient_accumulation_steps(batch_size)
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Using base LRs as scaled LRs (Removed complex scaling logic for simplicity)
        scaled_min_lr = lr_config['min_lr']
        scaled_max_lr = lr_config['max_lr']
        
        # Setup optimizer and scheduler with scaled LRs
        optimizer = optim.SGD(self.model.parameters(), lr=scaled_min_lr,
                              momentum=0.85, weight_decay=weight_decay, nesterov=True)
        
        # --- CRITICAL CORRECTION: div_factor uses scaled LRs ---
        steps_per_epoch = self.train_batches_per_epoch // gradient_accumulation_steps
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=scaled_max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=scaled_max_lr / scaled_min_lr, # CORRECTED: Use scaled LRs
            final_div_factor=1000,
            base_momentum=0.85,
            max_momentum=0.95
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # --- 2. Performance Enhancements ---
        
        # Check for torch.compile and Mixed Precision (AMP) logic
        is_cuda_available = torch.cuda.is_available()
        
        # torch.compile (Requires PyTorch 2.0+)
        if hasattr(torch, 'compile') and is_cuda_available and args is not None and not getattr(args, 'no_compile', False):
             try:
                 self.model = torch.compile(self.model)
                 logger.info("[SPEED] torch.compile enabled")
             except Exception as e:
                 logger.warning(f"[SPEED] torch.compile failed: {e}")

        # Mixed Precision Training (AMP)
        scaler = None
        # Assuming AMP is desired unless explicitly disabled via args
        if is_cuda_available and args is not None and not getattr(args, 'no_amp', False):
            scaler = GradScaler(enabled=torch.cuda.is_available())
            logger.info("[SPEED] Mixed precision training enabled (AMP)")
        else:
            scaler = GradScaler(enabled=torch.cuda.is_available())
            logger.info("[SPEED] AMP disabled or not available, using full precision")
        
        # cuDNN benchmark
        if is_cuda_available:
            torch.backends.cudnn.benchmark = True
            
        logger.info(f"💾 Training with effective batch size: {effective_batch_size} (BS={batch_size} * GA={gradient_accumulation_steps})")

        # --- 3. Training Loop ---
        best_val_acc = 0.0
        patience_counter = 0
        
        bar = tqdm(total=epochs, desc="Full Training", unit="epoch", ncols=120, disable=TQDM_DISABLE)
        
        for epoch in range(start_epoch, epochs):
            # Training
            train_loss, train_acc = self._train_epoch(optimizer, criterion, scheduler, scaler, gradient_accumulation_steps)

            # Validation
            val_loss, val_acc = self._validate_epoch(criterion)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['momentum'].append(optimizer.param_groups[0]['momentum'])

            # Log metrics for this epoch with timestamp
            import time
            metric_row = {
                'epoch': epoch+1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'momentum': optimizer.param_groups[0]['momentum'],
                'timestamp': time.time()
            }
            log_metrics_csv(csv_path, csv_fields, metric_row)
            log_metrics_json(json_path, metric_row)

            # Update tqdm progress bar
            bar.n = epoch + 1
            bar.set_description(f"Full Training | E: {epoch+1}/{epochs} | T: {train_loss:.4f}/{train_acc:.2f}% | V: {val_loss:.4f}/{val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
            bar.refresh()
            logger.info(f"💾 Full Training | E: {epoch+1}/{epochs} | T: {train_loss:.4f}/{train_acc:.2f}% | V: {val_loss:.4f}/{val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Clean up GPU memory
            aggressive_memory_cleanup()

            # Save best model and early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_checkpoints:
                    self._save_checkpoint(epoch, val_acc, optimizer, scheduler)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.warning(f"[TIME] Early stopping after {patience_counter} epochs without improvement")
                break

        bar.close()
        logger.info(f"[COMPLETE] Training completed! Best Val Acc: {best_val_acc:.2f}%")
        return self.history
    
    # ----------------------------------------------------------------------
    # Corrected _train_epoch Method (Simplified and Robust)
    # ----------------------------------------------------------------------

    def _train_epoch(self, optimizer, criterion, scheduler, scaler=None, gradient_accumulation_steps=1):
        """Train one epoch with AMP, Gradient Accumulation, and standard PyTorch pattern."""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        accumulation_counter = 0
        logger = get_unified_logger("FullTrainer - training")
        
        train_batches = self.train_batches_per_epoch
        bar = tqdm(total=train_batches, desc="Training", unit="batch", ncols=120, disable=TQDM_DISABLE)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # --- CRITICAL: Zero grad only at the start of accumulation cycle ---
            if accumulation_counter == 0:
                optimizer.zero_grad()
            
            try:
                # Forward pass with AMP (autocast) if scaler is available
                with autocast(enabled=self.enable_amp):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass: Scale loss for accumulation
                scaled_loss = loss / gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                accumulation_counter += 1
                
                # --- CRITICAL: Optimizer Step (After Accumulation) ---
                if accumulation_counter % gradient_accumulation_steps == 0:
                    
                    if scaler:
                        # Unscale before clipping
                        scaler.unscale_(optimizer) 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update() # Update scaler
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                    # Step scheduler
                    scheduler.step()
                    accumulation_counter = 0 # Reset counter

                # --- Metric Update ---
                running_loss += loss.item() # Use un-scaled loss for metric
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress
                bar.set_description(f"Training | B: {batch_idx+1}/{train_batches} | L: {running_loss/(batch_idx+1):.4f} | A: {100.*correct/total:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
                bar.update(1)

                # Clean up memory
                del inputs, targets, outputs, loss, predicted
                if (batch_idx + 1) % 50 == 0:
                     aggressive_memory_cleanup()
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"💾 Full Training - Training | B: {batch_idx+1}/{train_batches} | L: {running_loss/(batch_idx+1):.4f} | A: {100.*correct/total:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
                     
            except RuntimeError as e:
                # Catch OOM errors if they still occur
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ OOM in training step {batch_idx}: {e}")
                    aggressive_memory_cleanup()
                    raise e
                else:
                    raise e
                    
        bar.close()
        # Final step check: If accumulation_counter > 0, the last batch was too small for a full step
        # This is expected and typically ignored in standard training loops.
        return running_loss / train_batches, 100. * correct / total
    
    # ----------------------------------------------------------------------
    # _validate_epoch and _save_checkpoint (Kept mostly as is)
    # ----------------------------------------------------------------------
    
    def _validate_epoch(self, criterion):
        """Validate one epoch with AMP support."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        logger = get_unified_logger("FullTrainer - validation")
        
        val_batches = self.val_batches_per_epoch
        bar = tqdm(total=val_batches, desc="Validation", unit="batch", ncols=120, disable=TQDM_DISABLE)
        
        with torch.no_grad():
            with autocast(enabled=self.enable_amp):
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    bar.set_description(f"Validation | B: {batch_idx+1}/{val_batches} | L: {running_loss/(batch_idx+1):.4f} | A: {100.*correct/total:.2f}%")
                    bar.update(1)

                    del inputs, targets, outputs, loss, predicted
                    if (batch_idx + 1) % 25 == 0:
                        aggressive_memory_cleanup()
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"💾 Full Training - Validation | B: {batch_idx+1}/{val_batches} | L: {running_loss/(batch_idx+1):.4f} | A: {100.*correct/total:.2f}%")

        bar.close()
        return running_loss / val_batches, 100. * correct / total
    
    def _save_checkpoint(self, epoch, val_acc, optimizer, scheduler, step=0):
        """Save model checkpoint to SageMaker's /opt/ml/checkpoints for spot resumption"""
        checkpoint_dir = '/opt/ml/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))

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

class TrainingResultPlotter:
    def __init__(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        self.pd = pd
        self.plt = plt

    def plot_train_log(self, output_dir):
        train_log_csv = os.path.join(output_dir, 'train_log.csv')
        if os.path.exists(train_log_csv):
            df = self.pd.read_csv(train_log_csv)
            self.plt.figure(figsize=(10, 6))
            self.plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            self.plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
            self.plt.xlabel('Epoch')
            self.plt.ylabel('Loss')
            self.plt.title('Training & Validation Loss')
            self.plt.legend()
            self.plt.grid(True)
            self.plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
            self.plt.close()

            self.plt.figure(figsize=(10, 6))
            self.plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
            self.plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
            self.plt.xlabel('Epoch')
            self.plt.ylabel('Accuracy (%)')
            self.plt.title('Training & Validation Accuracy')
            self.plt.legend()
            self.plt.grid(True)
            self.plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
            self.plt.close()
        else:
            print('No train_log.csv found for plotting.')

    def plot_lr_range_curve(self, output_dir):
        lr_curve_csv = os.path.join(output_dir, 'lr_range_curve.csv')
        if os.path.exists(lr_curve_csv):
            df = self.pd.read_csv(lr_curve_csv)
            self.plt.figure(figsize=(10, 6))
            self.plt.plot(df['lr'], df['smoothed_loss'])
            self.plt.xscale('log')
            self.plt.xlabel('Learning Rate')
            self.plt.ylabel('Smoothed Loss')
            self.plt.title('LR Range Test Curve')
            self.plt.grid(True)
            self.plt.savefig(os.path.join(output_dir, 'lr_range_curve_plot.png'))
            self.plt.close()
        else:
            print('No lr_range_curve.csv found for plotting.')

    def plot_weight_decay_search(self, output_dir):
        wd_csv = os.path.join(output_dir, 'weight_decay_search.csv')
        if os.path.exists(wd_csv):
            df = self.pd.read_csv(wd_csv)
            self.plt.figure(figsize=(10, 6))
            self.plt.plot(df['weight_decay'], df['best_val_acc'], marker='o')
            self.plt.xscale('log')
            self.plt.xlabel('Weight Decay')
            self.plt.ylabel('Best Validation Accuracy (%)')
            self.plt.title('Weight Decay Search Results')
            self.plt.grid(True)
            self.plt.savefig(os.path.join(output_dir, 'weight_decay_search_plot.png'))
            self.plt.close()
        else:
            print('No weight_decay_search.csv found for plotting.')

    def plot_all(self, output_dir):
        self.plot_train_log(output_dir)
        self.plot_lr_range_curve(output_dir)
        self.plot_weight_decay_search(output_dir)


def main():
    # --- Step 1: Define checkpoint directory and pipeline status file ---
    # ...existing code...
    global os, logger, torch
    checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
    status_file = os.path.join(checkpoint_dir, 'pipeline_status.json')

    current_stage = 'START'
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            current_stage = status.get('last_completed_stage', 'START')
            global logger
            logger.info(f"🔄 Resuming pipeline. Last completed stage: {current_stage}")
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline status: {e}. Starting from scratch.")

    """Main training pipeline"""
    # Set up unified logging first thing
    logger = get_unified_logger("imagenet_training_pipeline")

    # =============================================================================
    # SAGEMAKER TRAINING STARTED - SIMPLE STATUS LOG
    # =============================================================================
    if (is_main_process()):
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
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick-mode', action='store_true', default=False, help='Enable quick mode for faster testing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no-amp', action='store_true', default=False, help='Disable mixed precision training')
    parser.add_argument('--no-compile', action='store_true', default=False, help='Disable torch.compile optimization')
    parser.add_argument('--lightweight-augs', action='store_true', default=False, help='Use lightweight augmentations for maximum speed')
    parser.add_argument('--skip-lr-test', action='store_true', default=False, help='Skip LR range test')
    parser.add_argument('--skip-wd-search', action='store_true', default=False, help='Skip weight decay search')
    parser.add_argument('--world-size', type=int, default=1, help='Number of distributed processes (GPUs)')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for DDP')
    args = parser.parse_args()
    
    # Step 2: Start resource monitor before training
    if (is_main_process()):
        monitor = ResourceMonitor(interval=5.0)
        monitor.start()
    
    # -----------------------------
    # Log config and environment info
    # -----------------------------
    output_dir = args.output if hasattr(args, 'output') else './imagenet_pipeline_results'
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'run_config.json')
    run_config = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'working_directory': os.getcwd(),
        'script_path': sys.argv[0],
        'command_line_args': sys.argv[1:] if len(sys.argv) > 1 else 'None',
        'hyperparameters': {
            'train_data': args.train,
            'val_data': args.val,
            'output_dir': args.output,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'quick_mode': args.quick_mode,
            'no_amp': args.no_amp,
            'no_compile': args.no_compile,
            'lightweight_augs': args.lightweight_augs,
            'skip_lr_test': args.skip_lr_test,
            'skip_wd_search': args.skip_wd_search
        }
    }
    log_config_json(config_path, run_config)
    
    
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
        # DDP setup - Get values from environment variables (SageMaker/NVIDIA/PyTorch)
        world_size = int(os.environ.get('WORLD_SIZE', getattr(args, 'world_size', 1)))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', getattr(args, 'local_rank', 0))))
        logger.info(f"[DDP] world_size: {world_size}, local_rank: {local_rank}")
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
    logger.info("[DEBUG] DEBUG: No batch size specified, starting batch size detection")
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
            enable_amp=True,
            enable_profiling=False  # Disable profiling for batch size detection
        )
        
        # Use optimizer's batch size detection method with actual GPU memory
        if torch.cuda.is_available():
            actual_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Set the memory limit for a single GPU process (90% heuristic)
            max_memory_gb = actual_gpu_memory_gb * 0.90
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                logger.info(f"[BATCH] Multi-GPU ({num_gpus} GPUs): Using {max_memory_gb:.1f}GB limit per GPU (Total VRAM: {actual_gpu_memory_gb*num_gpus:.1f}GB)")
            else:
                logger.info(f"[BATCH] Single GPU: Using {max_memory_gb:.1f}GB limit out of {actual_gpu_memory_gb:.1f}GB VRAM")
        else:
            max_memory_gb = 4.0  # Conservative for CPU
            logger.info(f"[BATCH] CPU mode: Using {max_memory_gb:.1f}GB memory limit")

        optimal_batch_size = temp_performance_optimizer.get_optimal_batch_size(max_memory_gb=max_memory_gb)
        
        # 1. Determine Safety Factor
        safety_factor = 0.5 if args.quick_mode else 0.8
        logger.info(f"[BATCH] Applying safety factor of {safety_factor}")

        # 2. Apply Safety Factor
        safe_batch_size = int(optimal_batch_size * safety_factor)

        # 3. Round down to the nearest Power of 2 for GPU efficiency, ensuring a minimum of 1
        if safe_batch_size > 0:
            # This finds the largest power of 2 <= safe_batch_size
            training_batch_size = max(1, 2 ** int(np.floor(np.log2(safe_batch_size))))
        else:
            training_batch_size = 1
            
        initial_batch_size = training_batch_size

        logger.info(f"[COMPLETE] Optimal batch size (CPU/GPU if exists): {initial_batch_size} (optimizer: {optimal_batch_size}, safety: {safety_factor})")

        # Clean up temporary resources
        #del temp_model, temp_optimizer, temp_criterion, temp_performance_optimizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"[ERROR] DEBUG: Error in batch size detection: {e}")
        # Fallback to default batch size
        initial_batch_size = 32
        logger.warning(f"[FALLBACK] Using default batch size: {initial_batch_size}")
    
    args.batch_size = initial_batch_size
    
    # Optimize num_workers for balanced CPU/GPU utilization and memory usage
    logger.info("[OPTIMIZE] Optimizing num_workers for DataLoader...")
    optimized_num_workers_i = optimize_num_workers(initial_batch_size)
    logger.info(f"[OPTIMIZE] Using optimized num_workers: {optimized_num_workers_i} (batch_size: {initial_batch_size})")
    
    # Monitor GPU utilization to validate optimization
    if torch.cuda.is_available():
        gpu_stats = monitor_gpu_utilization(duration_seconds=2)
        logger.info(f"[GPU] Initial GPU utilization: {gpu_stats['avg_utilization']:.1f}% (bottleneck: {gpu_stats['is_bottleneck']})")
        if gpu_stats['is_bottleneck']:
            logger.warning("[GPU] GPU utilization low - data loading may be bottleneck, num_workers optimization should help")
    
    # Override args.num_workers with optimized value
    args.num_workers = optimized_num_workers_i
    
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
        train_loader, val_loader, train_batches_per_epoch, val_batches_per_epoch = get_imagenet_dataloaders(
            train=args.train, val=args.val, batch_size=initial_batch_size, num_workers=args.num_workers, 
            lightweight_augs=args.lightweight_augs)
        # DDP: Use DistributedSampler if multi-GPU -> will not support in webloader
        #if torch.cuda.is_available() and getattr(args, 'world_size', 1) > 1:
        #    from torch.utils.data.distributed import DistributedSampler
        #    train_loader.sampler = DistributedSampler(train_loader.dataset, num_replicas=args.world_size, rank=args.local_rank)
        #    val_loader.sampler = DistributedSampler(val_loader.dataset, num_replicas=args.world_size, rank=args.local_rank)
        #    logger.info("[DDP] Using DistributedSampler for train and val loaders")
        progress_manager.update_progress(2, {'step': 'Loading validation dataset'})
    except Exception as e:
        progress_manager.close_progress_bar()
        logger.error(f"Dataset loading failed: {e}")
        raise
    
    progress_manager.close_progress_bar()
    # For WebLoader, use the calculated batch counts and known dataset sizes
    num_train_batches = train_batches_per_epoch
    num_val_batches = val_batches_per_epoch
    num_train_samples = IMAGENET_TRAIN_SIZE
    num_val_samples = IMAGENET_VAL_SIZE
    logger.info(f"Dataset loaded - Train batches: {num_train_batches}, Val batches: {num_val_batches}")
    logger.info(f"Dataset loaded - Train samples: {num_train_samples}, Val samples: {num_val_samples}")
    
    # Not required for webloader
    # Initialize Training Performance Optimizer for data loading optimization
    #logger.info("[OPTIMIZER] Initializing TrainingPerformanceOptimizer for data loading optimization...")
    #try:
    #    # Create temporary model and optimizer for optimizer initialization
    #    temp_model = create_model().to(device)
    #    temp_optimizer = optim.SGD(temp_model.parameters(), lr=1e-3, momentum=0.9)
    #    temp_criterion = nn.CrossEntropyLoss()
    #    
    #    # Create optimizer instance
    #    performance_optimizer = TrainingPerformanceOptimizer(
    #        model=temp_model,
    #        optimizer=temp_optimizer,
    #        criterion=temp_criterion,
    #        train_loader=train_loader,
    #        val_loader=val_loader,
    #        device=device,
    #        batch_size=args.batch_size,
    #        num_workers=args.num_workers,
    #        enable_amp=not args.no_amp,
    #        enable_profiling=True
    #    )
    #    
    #    # Optimize data loading
    #    logger.info("[OPTIMIZER] Optimizing data loading pipeline...")
    #    optimized_train_loader, optimized_val_loader = performance_optimizer.optimize_data_loading(target_workers=16)
    #    
    #    # Replace original loaders with optimized ones
    #    train_loader = optimized_train_loader
    #    if optimized_val_loader is not None:
    #        val_loader = optimized_val_loader
    #    logger.info("[OPTIMIZER] ✅ Data loading optimization complete")
    #    
    #    # Clean up temporary resources
    #    del temp_model, temp_optimizer, temp_criterion
    #    
    #except Exception as e:
    #    logger.warning(f"[OPTIMIZER] ⚠️ Failed to initialize performance optimizer: {e}")
    #    logger.warning("[OPTIMIZER] Continuing with standard data loading...")
    #    performance_optimizer = None
           
    
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
        if current_stage == 'START':
            # All processes run LR range test simultaneously to maintain NCCL synchronization
            lrs, losses = lr_finder.range_test(lr_test_loader, num_iter=num_iter)

            # Only main process plots and suggests LR config
            if is_main_process():
                # Plot results
                fig, min_lr = lr_finder.plot()
                fig.savefig(os.path.join(args.output, 'lr_range_test.png'))
                global plt
                plt.close(fig)
                # Get suggestions
                lr_config = lr_finder.suggest_lr()
            else:
                # Non-main processes get lr_config via broadcasting
                lr_config = lr_finder.suggest_lr()

            # Broadcast lr_config from rank 0 to ensure consistency across all processes
            if hasattr(args, 'world_size') and args.world_size > 1:
                import torch.distributed as dist
                if is_main_process():
                    lr_config_str = json.dumps(lr_config)
                    lr_config_bytes = lr_config_str.encode('utf-8')
                    lr_config_size = len(lr_config_bytes)
                else:
                    lr_config_size = 0

                # Broadcast size first
                lr_config_size_tensor = torch.tensor([lr_config_size], dtype=torch.int32, device=device)
                dist.broadcast(lr_config_size_tensor, src=0)
                lr_config_size = lr_config_size_tensor.item()

                # Broadcast data
                if is_main_process():
                    lr_config_tensor = torch.frombuffer(lr_config_bytes, dtype=torch.uint8).to(device)
                else:
                    lr_config_tensor = torch.zeros(lr_config_size, dtype=torch.uint8, device=device)

                dist.broadcast(lr_config_tensor, src=0)

                # Decode on non-main processes
                if not is_main_process():
                    lr_config_bytes = bytes(lr_config_tensor.cpu().numpy())
                    lr_config_str = lr_config_bytes.decode('utf-8')
                    lr_config = json.loads(lr_config_str)
        else:
             # Load lr_config from checkpoint_dir
            lr_config_path = os.path.join(checkpoint_dir, 'lr_config.json')
            with open(lr_config_path, 'r') as f:
                lr_config = json.load(f)

        

        progress_manager.update_status(lr_step_key,
            f"[OK] STEP 1: LR Range Test Complete - Min: {lr_config['min_lr']:.2e}, Max: {lr_config['max_lr']:.2e}")
        progress_manager.finalize_status(lr_step_key)

        # Save results (only main process to avoid conflicts)
        if is_main_process():
            with open(os.path.join(args.output, 'lr_config.json'), 'w') as f:
                json.dump({k: float(v) for k, v in lr_config.items()}, f, indent=2)

            # Save LR range test curve to CSV/JSON for analysis
            lr_curve_csv = os.path.join(args.output, 'lr_range_curve.csv')
            lr_curve_json = os.path.join(args.output, 'lr_range_curve.json')
            import csv
            import time
            with open(lr_curve_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['iteration', 'lr', 'smoothed_loss', 'timestamp'])
                writer.writeheader()
                for i, (lr, loss) in enumerate(zip(lrs, losses)):
                    writer.writerow({'iteration': i+1, 'lr': lr, 'smoothed_loss': loss, 'timestamp': time.time()})
            with open(lr_curve_json, 'w') as f:
                json.dump([
                    {'iteration': i+1, 'lr': float(lr), 'smoothed_loss': float(loss), 'timestamp': time.time()}
                    for i, (lr, loss) in enumerate(zip(lrs, losses))
                ], f, indent=2)
            
            lr_config_path = os.path.join(checkpoint_dir, 'lr_config.json')
            with open(lr_config_path, 'w') as f:
                json.dump(lr_config, f)
            logger.info(f"✅ Saved LR config to {lr_config_path}")
            
            save_pipeline_status('LR_TEST_COMPLETE', status_file)    
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
    logger.info(f"[OK] STEP 2 and 3: LR bounds selected: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
    
    # STEP 4: Batch Size Already Optimized
    optimal_batch_size = initial_batch_size
    logger.info(f"[OK] STEP 4: Using optimized batch size: {optimal_batch_size}")
    
    # STEP 5: Weight Decay Search
    best_weight_decay = 1e-4  # Default
    if not args.skip_wd_search:
        wd_step_key = "weight_decay_search"
        progress_manager.create_status_updater(wd_step_key, "[WEIGHT] STEP 5: Weight Decay Search - Starting...")

        wd_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] if not args.quick_mode else [1e-4, 5e-4]
        search_epochs = 3 if args.quick_mode else 5

        if current_stage in ['START', 'LR_TEST_COMPLETE']:
            # Create HyperparameterOptimizer for each process (each needs its own instance)
            optimizer = HyperparameterOptimizer(
                model_fn=create_model,  # Function to create fresh models
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                train_batches_per_epoch=train_batches_per_epoch,
                val_batches_per_epoch=val_batches_per_epoch
            )

            # Parallel weight decay search: Each GPU tests different weight decay values simultaneously
            # Each process runs independently and saves results to file - no distributed coordination needed

            # Get distributed training parameters from environment variables (SageMaker/NVIDIA/PyTorch)
            world_size = int(os.environ.get('WORLD_SIZE', getattr(args, 'world_size', 1)))
            local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', getattr(args, 'local_rank', 0))))

            # Divide weight decay values among processes (each GPU gets subset)
            wd_values_per_process = len(wd_values) // world_size
            remainder = len(wd_values) % world_size

            # Assign values to each process (some get extra if not evenly divisible)
            start_idx = local_rank * wd_values_per_process + min(local_rank, remainder)
            end_idx = start_idx + wd_values_per_process + (1 if local_rank < remainder else 0)
            my_wd_values = wd_values[start_idx:end_idx]

            logger.info(f"[WEIGHT] Process {local_rank} assigned {len(my_wd_values)} weight decay values: {[f'{wd:.2e}' for wd in my_wd_values]}")

            # Each process runs its weight decay search independently (no DDP conflicts since each has its own model)
            progress_manager.update_status(wd_step_key, f"[WEIGHT] STEP 5: Process {local_rank} testing {len(my_wd_values)} weight decay values for {search_epochs} epochs each...")
            my_wd_results, my_best_weight_decay = optimizer.weight_decay_search(
                lr_config, optimal_batch_size, my_wd_values, epochs=search_epochs)
            logger.info(f"[WEIGHT] Process {local_rank} completed weight decay search - Best WD: {my_best_weight_decay:.2e}")

            # Save results to file for this process
            process_results_file = os.path.join(checkpoint_dir, f'wd_results_rank_{local_rank}.json')
            with open(process_results_file, 'w') as f:
                json.dump({
                    'wd_results': my_wd_results,
                    'best_weight_decay': my_best_weight_decay,
                    'process_rank': local_rank
                }, f)
            logger.info(f"[WEIGHT] Process {local_rank} saved results to {process_results_file}")

            # Wait for all processes to complete (barrier)
            # Use file-based synchronization since distributed process group may not be initialized yet
            import time
            all_results_files = [os.path.join(checkpoint_dir, f'wd_results_rank_{i}.json') for i in range(world_size)]
            while not all(os.path.exists(f) for f in all_results_files):
                time.sleep(1)  # Wait 1 second and check again

            # Process 0 collects and combines all results
            if is_main_process():
                all_wd_results = []
                for i in range(world_size):
                    results_file = os.path.join(checkpoint_dir, f'wd_results_rank_{i}.json')
                    with open(results_file, 'r') as f:
                        process_results = json.load(f)
                        all_wd_results.extend(process_results['wd_results'])

                # Find best weight decay across all processes
                best_result = max(all_wd_results, key=lambda x: x['best_val_acc'])
                best_weight_decay = best_result['weight_decay']
                wd_results = all_wd_results

                logger.info(f"[WEIGHT] Combined results from all {world_size} processes. Best WD: {best_weight_decay:.2e} (acc: {best_result['best_val_acc']:.4f})")
            else:
                # Non-main processes get results from main process via file
                wd_results = my_wd_results  # Fallback
                best_weight_decay = my_best_weight_decay  # Fallback

                # Wait for main process to write combined results
                combined_results_file = os.path.join(checkpoint_dir, 'wd_results_combined.json')
                while not os.path.exists(combined_results_file):
                    time.sleep(1)

                with open(combined_results_file, 'r') as f:
                    combined_data = json.load(f)
                    wd_results = combined_data['wd_results']
                    best_weight_decay = combined_data['best_weight_decay']

            # Main process saves combined results
            if is_main_process():
                combined_results_file = os.path.join(checkpoint_dir, 'wd_results_combined.json')
                with open(combined_results_file, 'w') as f:
                    json.dump({
                        'wd_results': wd_results,
                        'best_weight_decay': best_weight_decay
                    }, f)
        else:
            # Load best_weight_decay from checkpoint_dir
            wd_results_path = os.path.join(checkpoint_dir, 'wd_results.json')
            with open(wd_results_path, 'r') as f:
                wd_data = json.load(f)
                wd_results = wd_data['wd_results']
                best_weight_decay = wd_data['best_weight_decay']
        
        # Save results (only main process)
        if is_main_process():
            with open(os.path.join(args.output, 'weight_decay_search.json'), 'w') as f:
                json.dump(wd_results, f, indent=2)

            # Save weight decay search results to CSV for analysis
            wd_csv = os.path.join(args.output, 'weight_decay_search.csv')
            import csv
            import time
            with open(wd_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['weight_decay', 'best_val_acc', 'timestamp'])
                writer.writeheader()
                for result in wd_results:
                    writer.writerow({'weight_decay': result['weight_decay'], 'best_val_acc': result['best_val_acc'], 'timestamp': time.time()})

            progress_manager.update_status(wd_step_key, f"[OK] STEP 5: Weight Decay Search Complete - Best WD: {best_weight_decay:.2e}")
            progress_manager.finalize_status(wd_step_key)
            
            wd_results_path = os.path.join(checkpoint_dir, 'wd_results.json')
            with open(wd_results_path, 'w') as f:
                json.dump({
                    'wd_results': wd_results,
                    'best_weight_decay': best_weight_decay
                }, f)
            logger.info(f"✅ Saved WD results and best_weight_decay to {wd_results_path}")
            
            save_pipeline_status('WD_SEARCH_COMPLETE', status_file)
        
        # Check for memory fragmentation after weight decay search
        check_memory_fragmentation()
        
    else:
        wd_skip_key = "wd_skip"
        progress_manager.create_status_updater(wd_skip_key, "⏭️ STEP 5: Skipping weight decay search, using default 1e-4")
        progress_manager.finalize_status(wd_skip_key)
    
    # Aggressive memory cleanup before full training to prevent fragmentation
    logger.info("[MEMORY] Performing aggressive memory cleanup before full training...")
    aggressive_memory_cleanup()
    
    # Final memory check and cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_detailed_memory_usage("pre_training_final_check")
    
    # STEP 6: Full Training
    training_epochs = 20 if args.quick_mode else args.epochs
    
    # Log memory usage before starting full training
    log_detailed_memory_usage("before_full_training")
    
    full_train_key = "full_training"
    progress_manager.create_status_updater(full_train_key,
        f"[START] STEP 6: Full OneCycle Training - Starting {training_epochs} epochs...")

    if is_main_process():
        logger.info("="*60)
        logger.info("[START] STEP 6: Full OneCycle Training")
        logger.info("="*60)
        logger.info(f"[MEMORY] Final batch size: {args.batch_size}")
        # Note: gradient_accumulation_steps and effective batch size will be logged after calculation
        if args.batch_size <= 8:
            logger.info("[MEMORY] Gradient checkpointing: ENABLED")
        else:
            logger.info("[MEMORY] Gradient checkpointing: DISABLED")

    model = create_model().to(device)
    # DDP: Wrap model if multi-GPU
    if torch.cuda.is_available() and getattr(args, 'world_size', 1) > 1:
        import torch.nn.parallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        logger.info("[DDP] Model wrapped with DistributedDataParallel")
    
    # Log memory usage after model creation
    log_detailed_memory_usage("after_model_creation")
    
    # Enable gradient checkpointing based on instance profile and batch size
    instance_profile = get_instance_resource_profile()
    should_checkpoint = instance_profile['enable_gradient_checkpointing'] or optimal_batch_size <= 8

    if should_checkpoint:
        logger.info("[MEMORY] Enabling gradient checkpointing for memory efficiency")
        try:
            # Import checkpoint utilities
            # Removed unused import checkpoint_sequential

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
            
    model_ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    start_epoch = 0
    if os.path.exists(model_ckpt_path):
        checkpoint = torch.load(model_ckpt_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0) + 1
        # Load model, optimizer, scheduler state as needed:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"➡️ Resuming Full Training from Epoch {start_epoch}")
    else:
        logger.info("🟢 Starting Full Training from Epoch 0")
    
    trainer = FullTrainer(model, train_loader, val_loader, device, args.output, train_batches_per_epoch, val_batches_per_epoch)

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

    # Now log the gradient accumulation details after calculation
    logger.info(f"[MEMORY] Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"[MEMORY] Effective batch size: {optimal_batch_size * gradient_accumulation_steps}")
    if optimal_batch_size <= 8:
        logger.info("[MEMORY] Gradient checkpointing: ENABLED")
    else:
        logger.info("[MEMORY] Gradient checkpointing: DISABLED")
        
    if current_stage in ['START', 'LR_TEST_COMPLETE', 'WD_SEARCH_COMPLETE']:
        history = trainer.train(
            lr_config=lr_config,
            epochs=training_epochs,
            batch_size=optimal_batch_size,
            weight_decay=best_weight_decay,
            save_checkpoints=True,
            early_stopping_patience=15 if not args.quick_mode else 5,
            args=args,
            start_epoch=start_epoch,
            #performance_optimizer=performance_optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    
    progress_manager.update_status(full_train_key, f"[OK] STEP 6: Full Training Complete - Best Val Acc: {max(history['val_acc']):.2f}%")
    progress_manager.finalize_status(full_train_key)
    
    # Performance Optimization Summary
    #if performance_optimizer:
    #    logger.info("="*60)
    #    logger.info("[OPTIMIZER] Performance Optimization Summary")
    #    logger.info("="*60)
    #    try:
    #        optimization_summary = performance_optimizer.get_optimization_summary()
    #        logger.info(optimization_summary)
    #        
    #        # Save optimization summary to file
    #        summary_file = os.path.join(args.output, 'optimization_summary.txt')
    #        with open(summary_file, 'w') as f:
    #            f.write(optimization_summary)
    #        logger.info(f"[SAVE] Optimization summary saved to: {summary_file}")
    #    except Exception as e:
    #        logger.warning(f"[OPTIMIZER] Failed to generate optimization summary: {e}")
   
    if (is_main_process()): 
        # STEP 7: Results Analysis and Plotting
        logger.info("="*60)
        logger.info("[ANALYSIS] STEP 7: Results Analysis")
        logger.info("="*60)
   
        # Create progress bar for results analysis
        analysis_steps = 4  # Plot creation, saving plot, saving JSON, final summary
        progress_manager.create_progress_bar("Results Analysis", analysis_steps)
    
    
        # ...existing training pipeline code...
        # Call TrainingResultPlotter after training completes
        plotter = TrainingResultPlotter()
        plotter.plot_all(args.output)
    
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
        import time
        final_results = {
            'lr_config': lr_config,
            'batch_size': optimal_batch_size,
            'weight_decay': best_weight_decay,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc']),
            'total_epochs': len(history['train_acc']),
            'timestamp': time.time()
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

        # Step 2: Stop resource monitor and save metrics after training
        monitor.stop()
        resource_metrics_path = os.path.join(args.output, 'resource_metrics.json')
        monitor.save(resource_metrics_path)
        logger.info(f"[RESOURCE MONITOR] Resource utilization metrics saved to: {resource_metrics_path}")
    
        # Step 3: Plot resource utilization after training
        try:
            import matplotlib.pyplot as plt
            ## removed duplicate import

            with open(resource_metrics_path) as f:
                data = json.load(f)

            timestamps = [m['timestamp'] - data[0]['timestamp'] for m in data]
            cpu = [m['cpu_percent'] for m in data]
            ram = [m['ram_gb'] for m in data]
            gpu_load = [m['gpus'][0]['load'] if m['gpus'] and len(m['gpus']) > 0 else 0 for m in data]

            plt.figure(figsize=(12,6))
            plt.plot(timestamps, cpu, label='CPU %')
            plt.plot(timestamps, ram, label='RAM (GB)')
            plt.plot(timestamps, gpu_load, label='GPU Load %')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization')
            plt.title('Resource Utilization Over Training')
            plot_path = os.path.join(args.output, 'resource_utilization.png')
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"[RESOURCE MONITOR] Resource utilization plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"[RESOURCE MONITOR] Could not plot resource utilization: {e}")

        # Log machine summary at the end of pipeline training
        import time
        machine_summary = get_hardware_summary()
        machine_summary['timestamp'] = time.time()
        logger.info("[MACHINE SUMMARY] Hardware and GPU Details:")
        for k, v in machine_summary.items():
            if k == 'gpus' and isinstance(v, list):
                for idx, gpu in enumerate(v):
                    logger.info(f"    GPU {idx}: {gpu['name']} ({gpu['memory_gb']} GB)")
            elif k == 'nvidia_smi':
                logger.info("    NVIDIA SMI Output:")
                for line in v.split('\n'):
                    logger.info(f"        {line}")
            else:
                logger.info(f"    {k}: {v}")
    
        final_results_path = os.path.join(checkpoint_dir, 'final_results.json')
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f)
        logger.info(f"✅ Saved final results to {final_results_path}")
    save_pipeline_status('FULL_TRAINING_COMPLETE', status_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if (is_main_process()):
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
            logger.error("[ERROR] Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error(f"   {line}")
            raise
