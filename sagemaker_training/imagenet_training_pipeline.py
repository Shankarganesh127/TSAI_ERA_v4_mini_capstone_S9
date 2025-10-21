#!/usr/bin/env python3
"""
Complete ImageNet Training Pipeline
7-Step Systematic Approach:
1) LR Range Test → 2) Pick LR bounds → 3) Set OneCycle LR + cyclical momentum 
→ 4) Choose batch size → 5) Tune weight-decay & regularizers → 6) Full OneCycle training → 7) Monitor & iterate
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_dataloaders
from ilsvrc_dataset import get_ilsvrc_dataloaders
from logger_setup import setup_logger, log_system_info, log_training_config, get_logger


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
        logger = get_logger()
        logger.info(f"🔍 Starting LR Range Test: {start_lr:.2e} → {end_lr:.2e}")
        
        # Calculate multiplicative factor
        lr_lambda = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
            
        self.model.train()
        losses = []
        lrs = []
        best_loss = float('inf')
        
        pbar = tqdm(total=num_iter, desc="LR Range Test")
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
                logger.warning(f"💥 Stopping early at iteration {i}, loss exploded")
                break
                
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_lambda
                
            pbar.set_postfix({
                'LR': f'{current_lr:.2e}',
                'Loss': f'{smoothed_loss:.3f}'
            })
            pbar.update()
            
        pbar.close()
        
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
        logger = get_logger()
        model.train()  # Use training mode for realistic memory usage
        batch_size = 1
        criterion = nn.CrossEntropyLoss()
        
        logger.info("🔍 Finding maximum batch size (training mode)...")
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
                
                logger.info(f"✅ Batch size {batch_size} works (train mode)")
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"❌ Batch size {batch_size} failed (OOM)")
                    max_working_batch_size = batch_size // 2
                    logger.info(f"🎯 Maximum batch size: {max_working_batch_size}")
                    return max_working_batch_size
                else:
                    raise e
                    
        return max_batch_size // 2


class HyperparameterOptimizer:
    """Grid/Random search for hyperparameters"""
    
    def __init__(self, model_fn, train_loader, val_loader, device):
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def weight_decay_search(self, lr_config, batch_size, wd_values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], epochs=5):
        """Search for optimal weight decay"""
        logger = get_logger()
        logger.info(f"🔍 Weight Decay Search: {wd_values}")
        
        results = []
        
        for wd in wd_values:
            logger.info(f"📊 Testing Weight Decay: {wd:.2e}")
            
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
            
            logger.info(f"📈 Results - Val Acc: {result['final_val_acc']:.2f}%, "
                  f"Val Loss: {result['final_val_loss']:.3f}")
        
        # Find best weight decay
        best_result = max(results, key=lambda x: x['best_val_acc'])
        logger.info(f"🎯 Best Weight Decay: {best_result['weight_decay']:.2e} "
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
            
            for inputs, targets in tqdm(self.train_loader, 
                                       desc=f'Epoch {epoch+1}/{epochs}', 
                                       leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Limit training batches for speed
                if train_batches >= 100:
                    break
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_batches += 1
                    
                    # Limit validation batches for speed
                    if val_batches >= 50:
                        break
            
            train_losses.append(train_loss / train_batches)
            val_losses.append(val_loss / val_batches)
            val_accs.append(100. * val_correct / val_total)
        
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
              save_checkpoints=True, early_stopping_patience=10):
        """Full training with OneCycle LR and cyclical momentum"""
        
        logger = get_logger()
        logger.info("🚀 Starting Full Training:")
        logger.info(f"   📚 Epochs: {epochs}")
        logger.info(f"   📏 LR Range: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
        logger.info(f"   ⚖️  Weight Decay: {weight_decay:.2e}")
        logger.info(f"   📦 Batch Size: {batch_size}")
        
        # Setup optimizer and scheduler
        optimizer = optim.SGD(self.model.parameters(), lr=lr_config['min_lr'],
                            momentum=0.85, weight_decay=weight_decay, nesterov=True)
        
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr_config['max_lr'],
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            div_factor=lr_config['max_lr'] / lr_config['min_lr'],
            final_div_factor=1000,
            base_momentum=0.85,
            max_momentum=0.95
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"🔄 Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self._train_epoch(optimizer, criterion, scheduler)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(criterion)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['momentum'].append(optimizer.param_groups[0]['momentum'])
            
            # Logging
            logger.info(f"📊 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"📊 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            logger.info(f"📈 LR: {optimizer.param_groups[0]['lr']:.2e}, "
                  f"Momentum: {optimizer.param_groups[0]['momentum']:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_checkpoints:
                    self._save_checkpoint(epoch, val_acc, optimizer, scheduler)
                logger.info(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"⏰ Early stopping after {patience_counter} epochs without improvement")
                break
                
        logger.info(f"🎯 Training completed! Best Val Acc: {best_val_acc:.2f}%")
        return self.history
    
    def _train_epoch(self, optimizer, criterion, scheduler):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(pbar.n+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def _validate_epoch(self, criterion):
        """Validate one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(pbar.n+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
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
    logger = get_logger()
    logger.info(f"🔍 Checking dataset format for: {data_path}")
    
    # Case 1: Check if data_path points directly to ILSVRC root
    ilsvrc_root_indicators = [
        os.path.join(data_path, "Data", "CLS-LOC"),
        os.path.join(data_path, "ImageSets", "CLS-LOC"),
        os.path.join(data_path, "ImageSets", "CLS-LOC", "val.txt")
    ]
    
    if all(os.path.exists(path) for path in ilsvrc_root_indicators):
        logger.info("✅ Detected ILSVRC format (root directory)")
        return 'ilsvrc'
    
    # Case 2: Check if data_path points to CLS-LOC subdirectory
    # Look for parent ILSVRC structure
    if data_path.endswith("Data/CLS-LOC") or data_path.endswith("Data\\CLS-LOC"):
        # Go up two levels to find ILSVRC root
        potential_root = os.path.dirname(os.path.dirname(data_path))
        imagesets_path = os.path.join(potential_root, "ImageSets", "CLS-LOC", "val.txt")
        if os.path.exists(imagesets_path):
            logger.info("✅ Detected ILSVRC format (CLS-LOC subdirectory)")
            return 'ilsvrc'
    
    # Case 3: Check if we have flat validation directory (ILSVRC-style)
    val_dir = os.path.join(data_path, "val")
    if os.path.exists(val_dir):
        # Check if validation directory has subdirectories (standard) or flat files (ILSVRC)
        val_contents = os.listdir(val_dir)
        if val_contents:
            first_item = os.path.join(val_dir, val_contents[0])
            if os.path.isfile(first_item) and first_item.lower().endswith(('.jpg', '.jpeg')):
                logger.info("✅ Detected ILSVRC format (flat validation directory)")
                return 'ilsvrc'
    
    # Case 4: Check for standard ImageNet format
    standard_paths = [
        os.path.join(data_path, "train"),
        os.path.join(data_path, "val")
    ]
    
    if all(os.path.exists(path) for path in standard_paths):
        logger.info("✅ Detected standard ImageNet format")
        return 'imagenet'
    
    # Default to ILSVRC if we can't determine
    logger.warning("⚠️  Could not determine format, defaulting to ILSVRC")
    return 'ilsvrc'


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='ImageNet Training Pipeline')
    parser.add_argument('--train', type=str, required=True, help='ImageNet training dataset path')
    parser.add_argument('--val', type=str, required=True, help='ImageNet validation dataset path')
    parser.add_argument('--output', type=str, default='./imagenet_pipeline_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detect if not specified)')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs for full training')
    parser.add_argument('--skip-lr-test', action='store_true', help='Skip LR range test')
    parser.add_argument('--skip-wd-search', action='store_true', help='Skip weight decay search')
    parser.add_argument('--quick-mode', action='store_true', help='Quick mode with fewer iterations')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('imagenet_pipeline')
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Detect dataset format
    #dataset_format = detect_dataset_format(args.data)
    #logger.info(f"📂 Detected dataset format: {dataset_format.upper()}")
    
    # Model factory
    def create_model():
        return resnet50_imagenet(num_classes=1000, pretrained=False)
    
    # STEP 0: Batch Size Detection (if not specified)
    if args.batch_size is None:
        logger.info("="*60)
        logger.info("🔧 STEP 0: Batch Size Detection")
        logger.info("="*60)
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"🖥️  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total")
        
        # Create a temporary model to test batch sizes
        temp_model = create_model().to(device)
        max_batch_size = BatchSizeFinder.find_max_batch_size(temp_model, (3, 224, 224), device)
        
        # Use different safety factors based on mode
        if args.quick_mode:
            safety_factor = 0.25  # Very conservative for quick mode (training uses more memory than inference)
            logger.info("🚀 Quick mode: Using very conservative batch size for training stability")
        else:
            safety_factor = 0.5  # Conservative safety factor (training uses ~2x memory of inference)
        
        initial_batch_size = int(max_batch_size * safety_factor)
        # Ensure it's a power of 2 and at least 1
        initial_batch_size = max(1, 2 ** int(np.log2(initial_batch_size))) if initial_batch_size > 0 else 32
        
        logger.info(f"🎯 Optimal batch size: {initial_batch_size} (max: {max_batch_size}, safety: {safety_factor})")
        
        # Clean up temporary model
        del temp_model
        torch.cuda.empty_cache()
    else:
        initial_batch_size = args.batch_size
        logger.info(f"📏 Using specified batch size: {initial_batch_size}")
    
    # Load data
    logger.info("📂 Loading ImageNet dataset...")
    
    #if dataset_format == 'ilsvrc':
    #    logger.info("Using ILSVRC dataset loader (handles flat validation directory)")
    #    train_loader, val_loader = get_ilsvrc_dataloaders(
    #        args.data, batch_size=initial_batch_size, num_workers=4)
    #else:
    logger.info("Using standard ImageNet dataset loader")
    train_loader, val_loader = get_imagenet_dataloaders(
    train = args.train, val = args.val, batch_size=initial_batch_size, num_workers=4)
    
    logger.info(f"📊 Dataset loaded - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # STEP 1: LR Range Test
    lr_config = None
    if not args.skip_lr_test:
        logger.info("="*60)
        logger.info("🔍 STEP 1: LR Range Test")
        logger.info("="*60)
        
        model = create_model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        lr_finder = LRFinder(model, optimizer, criterion, device)
        
        num_iter = 100 if args.quick_mode else 200
        
        logger.info(f"🔍 Running LR range test with optimized batch size {initial_batch_size}")
        lrs, losses = lr_finder.range_test(train_loader, num_iter=num_iter)
        
        # Plot results
        fig, min_lr = lr_finder.plot()
        fig.savefig(os.path.join(args.output, 'lr_range_test.png'))
        plt.close(fig)
        
        # Get suggestions
        lr_config = lr_finder.suggest_lr()
        
        logger.info("📈 LR Range Test Results:")
        logger.info(f"   Min LR: {lr_config['min_lr']:.2e}")
        logger.info(f"   Max LR: {lr_config['max_lr']:.2e}")
        logger.info(f"   Steepest decline LR: {lr_config['steepest_decline_lr']:.2e}")
        
        # Save results
        with open(os.path.join(args.output, 'lr_config.json'), 'w') as f:
            json.dump({k: float(v) for k, v in lr_config.items()}, f, indent=2)
    else:
        # Default LR config
        lr_config = {'min_lr': 1e-3, 'max_lr': 0.1}
        logger.info("⏭️  Skipping LR Range Test, using default config")
    
    # STEP 2 & 3: Already incorporated in lr_config
    logger.info(f"✅ LR bounds selected: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
    
    # STEP 4: Batch Size Already Optimized
    optimal_batch_size = initial_batch_size
    logger.info(f"✅ Using optimized batch size: {optimal_batch_size}")
    
    # STEP 5: Weight Decay Search
    best_weight_decay = 1e-4  # Default
    if not args.skip_wd_search:
        logger.info("="*60)
        logger.info("⚖️  STEP 5: Weight Decay Search")
        logger.info("="*60)
        
        optimizer = HyperparameterOptimizer(create_model, train_loader, val_loader, device)
        
        wd_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] if not args.quick_mode else [1e-4, 5e-4]
        search_epochs = 3 if args.quick_mode else 5
        
        wd_results, best_weight_decay = optimizer.weight_decay_search(
            lr_config, optimal_batch_size, wd_values, epochs=search_epochs)
        
        # Save results
        with open(os.path.join(args.output, 'weight_decay_search.json'), 'w') as f:
            json.dump(wd_results, f, indent=2)
            
        logger.info(f"🎯 Best weight decay: {best_weight_decay:.2e}")
    else:
        logger.info("⏭️  Skipping weight decay search, using default 1e-4")
    
    # STEP 6: Full Training
    logger.info("="*60)
    logger.info("🚀 STEP 6: Full OneCycle Training")
    logger.info("="*60)
    
    model = create_model().to(device)
    trainer = FullTrainer(model, train_loader, val_loader, device, args.output)
    
    training_epochs = 20 if args.quick_mode else args.epochs
    history = trainer.train(
        lr_config=lr_config,
        epochs=training_epochs,
        batch_size=optimal_batch_size,
        weight_decay=best_weight_decay,
        save_checkpoints=True,
        early_stopping_patience=15 if not args.quick_mode else 5
    )
    
    # STEP 7: Results Analysis and Plotting
    logger.info("="*60)
    logger.info("📊 STEP 7: Results Analysis")
    logger.info("="*60)
    
    # Plot training curves
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
    plt.savefig(os.path.join(args.output, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final results
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
    
    logger.info("🎉 Pipeline Complete!")
    logger.info("📊 Final Results:")
    logger.info(f"   Best Validation Accuracy: {final_results['best_val_acc']:.2f}%")
    logger.info(f"   Final Training Accuracy: {final_results['final_train_acc']:.2f}%")
    logger.info(f"   Final Validation Accuracy: {final_results['final_val_acc']:.2f}%")
    logger.info(f"   Optimal Batch Size: {optimal_batch_size}")
    logger.info(f"   Best Weight Decay: {best_weight_decay:.2e}")
    logger.info(f"   LR Range: {lr_config['min_lr']:.2e} → {lr_config['max_lr']:.2e}")
    logger.info(f"📁 Results saved to: {args.output}")


if __name__ == '__main__':
    main()