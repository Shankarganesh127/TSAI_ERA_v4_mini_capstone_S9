#!/usr/bin/env python3
"""
Training Integration Script for Model Replacement

This script ensures that model replacement happens on every epoch
regardless of how the underlying training code is structured.
"""

import os
import sys
import importlib
import threading
import time
from pathlib import Path

from model_saver import EpochModelSaver
from sagemaker_logging import setup_sagemaker_logger


class TrainingMonitorThread:
    """Monitor training and handle model replacement automatically"""
    
    def __init__(self, output_dir, config_file=None):
        self.logger = setup_sagemaker_logger(__name__)
        self.output_dir = Path(output_dir)
        self.model_saver = EpochModelSaver(output_dir, config_file)
        self.running = False
        self.thread = None
        
        # Tracking
        self.last_epoch = 0
        self.last_model_time = 0
        
    def start_monitoring(self):
        """Start monitoring thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            self.logger.info("🔍 Model replacement monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.logger.info("🛑 Model replacement monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        while self.running:
            try:
                self._check_for_new_models()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.warning(f"Monitoring error: {e}")
                time.sleep(30)
    
    def _check_for_new_models(self):
        """Check for new model files and handle replacement"""
        
        # Look for model files in common locations
        search_paths = [
            self.output_dir,
            self.output_dir / "checkpoints",
            self.output_dir / "models",
            Path("/opt/ml/model"),
            Path("/opt/ml/checkpoints")
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # Look for PyTorch model files
            model_files = list(search_path.glob("*.pth")) + list(search_path.glob("*.pt"))
            
            for model_file in model_files:
                self._process_model_file(model_file)
    
    def _process_model_file(self, model_file):
        """Process a detected model file"""
        
        try:
            # Check if this is a new model (by modification time)
            mtime = model_file.stat().st_mtime
            
            if mtime > self.last_model_time:
                self.last_model_time = mtime
                
                # Try to extract epoch info from filename or model content
                epoch = self._extract_epoch_info(model_file)
                
                if epoch > self.last_epoch:
                    self.last_epoch = epoch
                    self._handle_model_replacement(model_file, epoch)
        
        except Exception as e:
            self.logger.debug(f"Error processing model file {model_file}: {e}")
    
    def _extract_epoch_info(self, model_file):
        """Extract epoch information from model file"""
        
        try:
            # Try to get epoch from filename
            import re
            epoch_match = re.search(r'epoch[_-]?(\d+)', model_file.name, re.IGNORECASE)
            if epoch_match:
                return int(epoch_match.group(1))
            
            # Try to load model and get epoch from state dict
            import torch
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    return checkpoint['epoch']
            except Exception:
                pass
            
            # Fallback: use file modification time as proxy for epoch
            return int(time.time()) % 1000
            
        except Exception:
            return 0
    
    def _handle_model_replacement(self, model_file, epoch):
        """Handle model replacement logic"""
        
        try:
            import torch
            
            # Load the model
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Extract information
            if isinstance(checkpoint, dict):
                accuracy = checkpoint.get('accuracy', 0.0)
                loss = checkpoint.get('loss', 0.0)
                lr = checkpoint.get('learning_rate', checkpoint.get('lr', 0.0))
            else:
                # If it's just the model state dict
                accuracy = 0.0
                loss = 0.0
                lr = 0.0
            
            # Create a dummy model for saving (we're just organizing existing models)
            self.logger.info(f"🔄 Processing model replacement for epoch {epoch}")
            
            # Copy to our organized structure with replacement
            self._organize_and_replace_model(model_file, epoch, accuracy, loss, lr)
            
        except Exception as e:
            self.logger.warning(f"Failed to handle model replacement: {e}")
    
    def _organize_and_replace_model(self, source_model, epoch, accuracy, loss, lr):
        """Organize model into our replacement structure"""
        
        try:
            import shutil
            import torch
            
            # Ensure models directory exists
            models_dir = self.model_saver.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and enhance the checkpoint
            checkpoint = torch.load(source_model, map_location='cpu')
            
            # Enhance with our metadata
            if isinstance(checkpoint, dict):
                checkpoint.update({
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': loss,
                    'learning_rate': lr,
                    'sagemaker_enhanced': True,
                    'replacement_timestamp': time.time()
                })
            else:
                # Convert to proper checkpoint format
                checkpoint = {
                    'model_state_dict': checkpoint,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': loss,
                    'learning_rate': lr,
                    'sagemaker_enhanced': True,
                    'replacement_timestamp': time.time()
                }
            
            # Replace current model
            current_model_path = models_dir / "model_current.pth"
            
            # Remove old current model
            if current_model_path.exists():
                current_model_path.unlink()
                self.logger.debug(f"🗑️ Removed previous current model")
            
            # Save new current model
            torch.save(checkpoint, current_model_path)
            
            # Update best model if this is better
            best_model_path = models_dir / "model_best.pth"
            update_best = False
            
            if not best_model_path.exists():
                update_best = True
            else:
                try:
                    best_checkpoint = torch.load(best_model_path, map_location='cpu')
                    best_accuracy = best_checkpoint.get('accuracy', 0.0)
                    if accuracy > best_accuracy:
                        update_best = True
                except Exception:
                    update_best = True
            
            if update_best:
                torch.save(checkpoint, best_model_path)
                self.logger.info(f"🏆 Best model updated: epoch {epoch} (acc: {accuracy:.4f})")
            
            # Create symlink for SageMaker
            sagemaker_model = models_dir / "model.pth"
            if sagemaker_model.exists() or sagemaker_model.is_symlink():
                sagemaker_model.unlink()
            
            try:
                sagemaker_model.symlink_to("model_current.pth")
            except OSError:
                shutil.copy2(current_model_path, sagemaker_model)
            
            self.logger.info(f"✅ Model replacement completed: epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to organize and replace model: {e}")


def setup_model_replacement_monitoring(output_dir, config_file=None):
    """Setup automatic model replacement monitoring"""
    
    monitor = TrainingMonitorThread(output_dir, config_file)
    monitor.start_monitoring()
    
    # Register cleanup
    import atexit
    atexit.register(monitor.stop_monitoring)
    
    return monitor


def patch_training_functions():
    """Monkey patch common PyTorch training functions to add model replacement"""
    
    try:
        import torch
        
        # Store original save function
        original_save = torch.save
        
        def enhanced_torch_save(obj, f, *args, **kwargs):
            # Call original save
            result = original_save(obj, f, *args, **kwargs)
            
            # Check if this looks like a model checkpoint
            if isinstance(f, (str, Path)) and str(f).endswith(('.pth', '.pt')):
                try:
                    # Try to trigger our monitoring
                    output_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
                    if hasattr(patch_training_functions, '_monitor'):
                        monitor = patch_training_functions._monitor
                        if hasattr(monitor, '_check_for_new_models'):
                            monitor._check_for_new_models()
                except Exception:
                    pass
            
            return result
        
        # Replace torch.save
        torch.save = enhanced_torch_save
        
        return enhanced_torch_save
        
    except Exception as e:
        logger = setup_sagemaker_logger(__name__)
        logger.warning(f"Could not patch training functions: {e}")
        return None


if __name__ == '__main__':
    # Setup monitoring for current environment
    output_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    config_file = os.environ.get('SM_MODEL_CONFIG', None)
    
    logger = setup_sagemaker_logger(__name__)
    logger.info("🚀 Starting training integration with model replacement")
    
    # Setup monitoring
    monitor = setup_model_replacement_monitoring(output_dir, config_file)
    patch_training_functions._monitor = monitor
    
    # Patch training functions
    enhanced_save = patch_training_functions()
    
    if enhanced_save:
        logger.info("✅ Training functions patched for model replacement")
    else:
        logger.warning("⚠️ Could not patch training functions")
    
    # Keep monitoring active
    try:
        while monitor.running:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping training integration")
        monitor.stop_monitoring()