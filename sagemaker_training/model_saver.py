#!/usr/bin/env python3
"""
Model Saving Utility for SageMaker Training

Handles model saving and replacement on each epoch for SageMaker training.
Integrates with existing training pipeline without code modifications.
"""

import os
import sys
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from sagemaker_logging import setup_sagemaker_logger
except ImportError:
    # Fallback logging
    import logging
    def setup_sagemaker_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


class EpochModelSaver:
    """Handles model saving and replacement for each epoch"""
    
    def __init__(self, output_dir, config_file=None):
        self.logger = setup_sagemaker_logger(__name__)
        self.output_dir = Path(output_dir)
        self.config = self._load_config(config_file)
        
        # Setup directories
        self.models_dir = self.output_dir / "models"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.current_model_path = self.models_dir / "model_current.pth"
        self.best_model_path = self.models_dir / "model_best.pth"
        self.final_model_path = self.models_dir / "model_final.pth"
        
        # Tracking
        self.best_accuracy = 0.0
        self.epoch_count = 0
        
        self.logger.info(f"🔧 Model saver initialized")
        self.logger.info(f"📁 Models directory: {self.models_dir}")
        self.logger.info(f"🔄 Replace mode: {self.config['replace_previous_model']}")
    
    def _load_config(self, config_file):
        """Load model saving configuration"""
        default_config = {
            "save_model_every_epoch": True,
            "replace_previous_model": True,
            "save_format": "pytorch",
            "keep_best_model": True,
            "save_optimizer_state": True,
            "compress_checkpoints": False
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def save_epoch_model(self, model, optimizer, epoch, accuracy, loss, lr=None):
        """Save model for current epoch, replacing previous epoch model"""
        
        self.epoch_count = epoch
        
        # Prepare model state
        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat(),
            'sagemaker_info': {
                'job_name': os.environ.get('SM_TRAINING_JOB_NAME', 'unknown'),
                'instance_type': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'unknown'),
                'output_dir': str(self.output_dir)
            }
        }
        
        if self.config["save_optimizer_state"] and optimizer:
            model_state['optimizer_state_dict'] = optimizer.state_dict()
        
        try:
            # Save current model (replaces previous epoch)
            if self.config["replace_previous_model"]:
                self._save_current_model(model_state, epoch, accuracy)
            
            # Save best model if this is the best so far
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self._save_best_model(model_state, epoch, accuracy)
            
            # Save checkpoint with epoch number (optional, for debugging)
            if not self.config["replace_previous_model"]:
                self._save_epoch_checkpoint(model_state, epoch)
            
            self.logger.info(f"💾 Epoch {epoch}: Model saved successfully")
            self.logger.info(f"📊 Current: Acc={accuracy:.4f}, Loss={loss:.4f}")
            if accuracy > self.best_accuracy:
                self.logger.info(f"🏆 New best accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model for epoch {epoch}: {e}")
            return False
    
    def _save_current_model(self, model_state, epoch, accuracy):
        """Save current model, replacing the previous one"""
        
        # Remove previous current model if it exists
        if self.current_model_path.exists():
            try:
                self.current_model_path.unlink()
                self.logger.debug(f"🗑️ Removed previous current model")
            except Exception as e:
                self.logger.warning(f"Could not remove previous model: {e}")
        
        # Save new current model
        torch.save(model_state, self.current_model_path)
        
        # Create a symbolic link or copy for SageMaker output
        sagemaker_model_path = self.models_dir / "model.pth"
        if sagemaker_model_path.exists():
            sagemaker_model_path.unlink()
        
        try:
            # Try to create symlink (faster)
            sagemaker_model_path.symlink_to(self.current_model_path.name)
        except OSError:
            # Fall back to copy if symlink fails
            shutil.copy2(self.current_model_path, sagemaker_model_path)
        
        self.logger.info(f"🔄 Current model replaced: epoch {epoch} (acc: {accuracy:.4f})")
    
    def _save_best_model(self, model_state, epoch, accuracy):
        """Save best model so far"""
        
        torch.save(model_state, self.best_model_path)
        self.logger.info(f"🏆 Best model updated: epoch {epoch} (acc: {accuracy:.4f})")
    
    def _save_epoch_checkpoint(self, model_state, epoch):
        """Save epoch-specific checkpoint (when not replacing)"""
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(model_state, checkpoint_path)
    
    def finalize_training(self, model, optimizer, final_epoch, final_accuracy, final_loss):
        """Save final model and create training summary"""
        
        try:
            # Save final model
            final_state = {
                'epoch': final_epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': final_accuracy,
                'loss': final_loss,
                'best_accuracy': self.best_accuracy,
                'total_epochs': self.epoch_count,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            if optimizer:
                final_state['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(final_state, self.final_model_path)
            
            # Create SageMaker model.tar.gz for deployment
            self._create_sagemaker_model_archive()
            
            # Generate training summary
            self._generate_training_summary(final_state)
            
            self.logger.info(f"🎉 Training finalized: Final model saved")
            self.logger.info(f"📊 Final accuracy: {final_accuracy:.4f}, Best: {self.best_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to finalize training: {e}")
            return False
    
    def _create_sagemaker_model_archive(self):
        """Create model.tar.gz for SageMaker deployment"""
        
        try:
            import tarfile
            
            # Use best model for deployment
            source_model = self.best_model_path if self.best_model_path.exists() else self.current_model_path
            
            if not source_model.exists():
                self.logger.warning("No model file found for archiving")
                return
            
            # Create tar.gz archive
            archive_path = self.output_dir / "model.tar.gz"
            
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(source_model, arcname="model.pth")
                
                # Add inference script if it exists
                inference_script = self.output_dir.parent / "inference.py"
                if inference_script.exists():
                    tar.add(inference_script, arcname="inference.py")
            
            self.logger.info(f"📦 SageMaker model archive created: {archive_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not create model archive: {e}")
    
    def _generate_training_summary(self, final_state):
        """Generate comprehensive training summary"""
        
        summary = {
            "training_summary": {
                "completed": True,
                "final_epoch": final_state['epoch'],
                "total_epochs": self.epoch_count,
                "final_accuracy": final_state['accuracy'],
                "best_accuracy": self.best_accuracy,
                "final_loss": final_state['loss']
            },
            "model_files": {
                "current_model": str(self.current_model_path),
                "best_model": str(self.best_model_path),
                "final_model": str(self.final_model_path),
                "archive": str(self.output_dir / "model.tar.gz")
            },
            "sagemaker_integration": {
                "job_name": os.environ.get('SM_TRAINING_JOB_NAME', 'unknown'),
                "output_dir": str(self.output_dir),
                "model_replacement_used": self.config["replace_previous_model"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.output_dir / "model_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Training summary saved: {summary_file}")


def create_model_saver_hook(output_dir, config_file=None):
    """Create a model saver instance for integration with training loops"""
    
    return EpochModelSaver(output_dir, config_file)


def save_model_checkpoint(model, optimizer, epoch, accuracy, loss, output_dir, lr=None, config_file=None):
    """Standalone function to save model checkpoint with replacement"""
    
    saver = EpochModelSaver(output_dir, config_file)
    return saver.save_epoch_model(model, optimizer, epoch, accuracy, loss, lr)


# Integration with existing training code
def monkey_patch_training_save(training_module):
    """Monkey patch existing training code to add model replacement saving"""
    
    original_save = getattr(training_module, 'save_checkpoint', None)
    
    def enhanced_save_checkpoint(*args, **kwargs):
        # Call original save if it exists
        if original_save:
            result = original_save(*args, **kwargs)
        else:
            result = True
        
        # Add our model replacement logic
        try:
            # Extract common arguments
            output_dir = kwargs.get('output_dir') or '/opt/ml/model'
            config_file = kwargs.get('config_file') or os.path.join(output_dir, 'model_save_config.json')
            
            if len(args) >= 4:
                model, optimizer, epoch, accuracy = args[:4]
                loss = args[4] if len(args) > 4 else 0.0
                lr = kwargs.get('lr')
                
                saver = EpochModelSaver(output_dir, config_file)
                saver.save_epoch_model(model, optimizer, epoch, accuracy, loss, lr)
        
        except Exception as e:
            print(f"Enhanced model saving failed: {e}")
        
        return result
    
    # Replace the save function
    setattr(training_module, 'save_checkpoint', enhanced_save_checkpoint)
    return enhanced_save_checkpoint


if __name__ == '__main__':
    # Test the model saver
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model saver')
    parser.add_argument('--output-dir', default='/tmp/model_test', help='Output directory')
    parser.add_argument('--config-file', help='Config file path')
    
    args = parser.parse_args()
    
    # Create test saver
    saver = EpochModelSaver(args.output_dir, args.config_file)
    
    print("Model saver test completed successfully!")
    print(f"Models will be saved to: {saver.models_dir}")
    print(f"Replace mode: {saver.config['replace_previous_model']}")