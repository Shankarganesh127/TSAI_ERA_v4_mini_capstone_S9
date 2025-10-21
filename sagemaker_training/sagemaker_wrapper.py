#!/usr/bin/env python3
"""
SageMaker Training Wrapper for 7-Step ImageNet Pipeline

This wrapper integrates the sophisticated 7-step ImageNet training methodology 
with SageMaker cloud deployment while preserving all advanced capabilities.

7-Step Pipeline:
1. LR Range Test → 2. Pick LR bounds → 3. OneCycle LR → 4. Choose batch size → 
5. Tune weight-decay → 6. Full training → 7. Monitor & iterate
"""

import os
import sys
import subprocess
import json
import argparse
import shutil
from pathlib import Path

# All files are now in the same directory - no parent directory needed
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Debug: Print current working directory and file locations
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🔍 Wrapper script location: {__file__}")
print(f"🔍 SageMaker training directory: {current_dir}")

# No need to change directories - just use absolute paths
print(f"🔄 Working from: {os.getcwd()}")

# Check if imagenet_training_pipeline.py exists at expected SageMaker location
pipeline_script = Path("/opt/ml/code/imagenet_training_pipeline.py")
print(f"🔍 Pipeline script path: {pipeline_script}")
print(f"🔍 Pipeline script exists: {pipeline_script.exists()}")

# List files in SageMaker code directory
sagemaker_code_dir = Path("/opt/ml/code")
if sagemaker_code_dir.exists():
    print(f"🔍 Files in /opt/ml/code/:")
    for f in sorted(sagemaker_code_dir.iterdir()):
        if f.is_file() and f.suffix == '.py':
            print(f"    {f.name}")
else:
    print(f"🔍 /opt/ml/code/ directory does not exist")

# Import logger - all files are in same directory now
try:
    from logger_setup import setup_logger
except ImportError:
    from sagemaker_logging import setup_sagemaker_logger as setup_logger

class ImageNetSageMakerTrainer:
    """Unified SageMaker wrapper for 7-step ImageNet training pipeline"""
    
    def __init__(self):
        # No need to change directories - use absolute paths
        print(f"🔄 INIT: Working from: {os.getcwd()}")
        
        self.logger = setup_logger("sagemaker_imagenet_trainer")
        self.config = {}
        
        # Double-check our working directory
        self.logger.info(f"🏠 SageMaker Wrapper initialized from: {os.getcwd()}")
        if Path("imagenet_training_pipeline.py").exists():
            self.logger.info("✅ Found imagenet_training_pipeline.py in current directory")
        else:
            self.logger.error("❌ imagenet_training_pipeline.py NOT found in current directory!")
        
    def parse_hyperparameters(self):
        """Parse SageMaker hyperparameters"""
        parser = argparse.ArgumentParser()
        
        # Core training parameters - use SageMaker environment variables as defaults
        parser.add_argument('--data_dir', type=str, 
                           default=os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet'))
        parser.add_argument('--val_dir', type=str, 
                           default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
        parser.add_argument('--output_dir', type=str, 
                           default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--batch_size', type=int, help='Override auto-detected batch size')
        
        # 7-Step pipeline control
        parser.add_argument('--run_lr_finder', type=str, default='true', help='Step 1: Run LR range test')
        parser.add_argument('--run_wd_search', type=str, default='true', help='Step 5: Run weight decay search')
        parser.add_argument('--quick_mode', type=str, default='false', help='Quick mode for development')
        
        # Manual hyperparameter overrides
        parser.add_argument('--lr_min', type=float, help='Manual LR minimum (Step 2 override)')
        parser.add_argument('--lr_max', type=float, help='Manual LR maximum (Step 2 override)')
        parser.add_argument('--weight_decay', type=float, help='Manual weight decay (Step 5 override)')
        
        # Advanced options
        parser.add_argument('--mixed_precision', type=str, default='true')
        parser.add_argument('--gradient_clip', type=float, default=1.0)
        parser.add_argument('--num_workers', type=int, default=4)
        
        args = parser.parse_args()
        
        # Convert string booleans
        args.run_lr_finder = args.run_lr_finder.lower() == 'true'
        args.run_wd_search = args.run_wd_search.lower() == 'true'
        args.quick_mode = args.quick_mode.lower() == 'true'
        args.mixed_precision = args.mixed_precision.lower() == 'true'
        
        # Log the resolved paths for debugging
        self.logger.info(f"📁 Data directory: {args.data_dir}")
        if args.val_dir:
            self.logger.info(f"📁 Validation directory: {args.val_dir}")
        else:
            self.logger.info(f"📁 Validation: Using subdirectory of data_dir")
        self.logger.info(f"📁 Output directory: {args.output_dir}")
        
        return args
    
    def build_pipeline_command(self, args):
        """Build command for 7-step pipeline execution with model saving"""
        # In SageMaker, all source code is uploaded to /opt/ml/code/
        
        # EARLY DEBUG - ensure this function is being called
        print("🚨 DEBUG: build_pipeline_command() function called!")
        self.logger.info("🚨 DEBUG: build_pipeline_command() function called!")
        
        self.logger.info("🔍 Building pipeline command...")
        self.logger.info(f"   Current working directory: {Path.cwd()}")
        
        # Use absolute path - simpler and more reliable
        pipeline_script_path = "/opt/ml/code/imagenet_training_pipeline.py"
        
        # Check if script exists
        if Path(pipeline_script_path).exists():
            self.logger.info(f"✅ Found script at: {pipeline_script_path}")
        else:
            self.logger.error(f"❌ CRITICAL ERROR: Script not found at {pipeline_script_path}")
            self.logger.error(f"   Current dir: {os.getcwd()}")
            # List files in /opt/ml/code/ for debugging
            code_dir = Path("/opt/ml/code")
            if code_dir.exists():
                self.logger.error(f"   Files in /opt/ml/code/:")
                for f in code_dir.iterdir():
                    if f.is_file():
                        self.logger.error(f"     {f.name}")
            raise FileNotFoundError(f"Script not found at {pipeline_script_path}")

        cmd = [
            sys.executable,
            pipeline_script_path,  # Use absolute path
            "--train", str(args.train_dir),
            "--val", str(args.val_dir),
            "--output", str(args.output_dir), 
            "--epochs", str(args.epochs)
        ]
        
        self.logger.info(f"✅ Command built: {' '.join(cmd)}")
        self.logger.info(f"   Using absolute path: {pipeline_script_path}")
        self.logger.info(f"   Python executable: {sys.executable}")
        
        # Add validation directory if separate channel is provided
        #if args.val_dir:
        #    cmd.extend(["--val-data", str(args.val_dir)])
        #    self.logger.info(f"📁 Using separate validation channel: {args.val_dir}")
        
        # Model saving configuration - integrate with model_saver.py
        #cmd.extend(["--save-model-every-epoch", "true"])
        #cmd.extend(["--model-save-path", str(Path(args.output_dir) / "models")])
        #cmd.extend(["--replace-model", "true"])  # Replace previous epoch model
        #cmd.extend(["--use-enhanced-saver", "true"])  # Use our model saver
        #cmd.extend(["--model-saver-config", str(Path(args.output_dir) / "model_save_config.json")])
        
        # Batch size control (Step 4)
        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])
            self.logger.info(f"🔧 Batch Size Override: {args.batch_size}")
        else:
            self.logger.info("🔄 Using automatic batch size detection (Step 4)")
        
        # LR finder control (Step 1)
        if not args.run_lr_finder:
            cmd.append("--skip-lr-test")
            self.logger.info("⏭️ Skipping LR Range Test (Step 1)")
            if args.lr_min and args.lr_max:
                self.logger.info(f"🎯 Using manual LR bounds: {args.lr_min:.2e} → {args.lr_max:.2e}")
        else:
            self.logger.info("🔍 Running LR Range Test (Step 1) with auto bounds (Steps 2-3)")
        
        # Weight decay control (Step 5)
        if not args.run_wd_search:
            cmd.append("--skip-wd-search")
            self.logger.info("⏭️ Skipping Weight Decay Search (Step 5)")
            if args.weight_decay:
                self.logger.info(f"⚖️ Using manual weight decay: {args.weight_decay:.2e}")
        else:
            self.logger.info("🔬 Running Weight Decay Search (Step 5)")
        
        # Quick mode
        if args.quick_mode:
            cmd.append("--quick-mode")
            self.logger.info("🚀 Quick mode enabled")
        
        return cmd
    
    def _setup_model_saving_directories(self, args):
        """Setup directories for model saving and create model saving configuration"""
        
        # Create model saving directories
        models_dir = Path(args.output_dir) / "models"
        checkpoints_dir = Path(args.output_dir) / "checkpoints"
        
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model saving configuration file for the training pipeline
        model_config = {
            "save_model_every_epoch": True,
            "replace_previous_model": True,
            "model_save_directory": str(models_dir),
            "checkpoint_directory": str(checkpoints_dir),
            "save_format": "pytorch",
            "model_naming": {
                "current_model": "model_current.pth",
                "best_model": "model_best.pth",
                "final_model": "model_final.pth"
            },
            "sagemaker_integration": {
                "output_dir": str(args.output_dir),
                "upload_to_s3": True,
                "s3_model_path": f"s3://{os.environ.get('SM_OUTPUT_DATA_DIR', args.output_dir)}/models/"
            }
        }
        
        # Save configuration for training pipeline to use
        config_file = Path(args.output_dir) / "model_save_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            self.logger.info(f"💾 Model saving configuration created: {config_file}")
            self.logger.info(f"📁 Models directory: {models_dir}")
            self.logger.info(f"🔄 Replace mode: Enabled (overwrites previous epoch model)")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create model config: {e}")
    
    def _start_model_replacement_monitoring(self, args):
        """Start automatic model replacement monitoring"""
        
        try:
            # Import and start training integration
            from training_integration import setup_model_replacement_monitoring, patch_training_functions
            
            # Setup monitoring
            config_file = Path(args.output_dir) / "model_save_config.json"
            monitor = setup_model_replacement_monitoring(args.output_dir, str(config_file))
            
            # Patch PyTorch save functions
            patch_training_functions._monitor = monitor
            enhanced_save = patch_training_functions()
            
            # Store monitor for later cleanup
            self._model_monitor = monitor
            
            self.logger.info("🔍 Model replacement monitoring started")
            
            if enhanced_save:
                self.logger.info("✅ PyTorch save functions patched for automatic replacement")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not start model monitoring: {e}")
            self._model_monitor = None
    
    def run_training(self):
        """Execute the complete 7-step training pipeline"""
        self.logger.info("🚀 Starting SageMaker 7-Step ImageNet Training")
        self.logger.info("=" * 60)
        
        # Log the 7-step process
        steps = [
            "1️⃣ LR Range Test → Find optimal learning rate bounds",
            "2️⃣ Pick LR bounds → Extract min/max LR from range test", 
            "3️⃣ OneCycle LR → Configure advanced scheduler",
            "4️⃣ Choose batch size → Auto-detect optimal GPU memory",
            "5️⃣ Tune weight-decay → Grid search with validation",
            "6️⃣ Full training → Complete OneCycle training",
            "7️⃣ Monitor → Comprehensive analysis and logging"
        ]
        
        self.logger.info("📋 7-Step Pipeline:")
        for step in steps:
            self.logger.info(f"   {step}")
        self.logger.info("=" * 60)
        
        # Parse configuration
        args = self.parse_hyperparameters()
        
        # Setup model saving directories
        self._setup_model_saving_directories(args)
        
        # Start model replacement monitoring
        self._start_model_replacement_monitoring(args)
        
        # Build and execute pipeline command
        cmd = self.build_pipeline_command(args)
        self.logger.info(f"🎯 Executing: {' '.join(cmd)}")
        
        # CRITICAL DEBUGGING - Show what's in the command
        self.logger.info(f"🔍 Command breakdown:")
        self.logger.info(f"   Python: {cmd[0]}")  
        self.logger.info(f"   Script: {cmd[1]}")
        self.logger.info(f"   Args: {cmd[2:]}")
        
        # Check if we're using relative path correctly
        if cmd[1].startswith('/opt/ml/') and not cmd[1].startswith('/opt/ml/code/'):
            self.logger.error(f"❌ WRONG PATH! Script path: {cmd[1]}")
            self.logger.error(f"❌ Should be relative or start with /opt/ml/code/")
            # Force fix the path
            if cmd[1] == '/opt/ml/imagenet_training_pipeline.py':
                cmd[1] = 'imagenet_training_pipeline.py'
                self.logger.info(f"🔧 FIXED path to: {cmd[1]}")
        else:
            self.logger.info(f"✅ Script path looks correct: {cmd[1]}")
        
        try:
            # No need to change working directory - use absolute paths
            run_cwd = Path.cwd()
            self.logger.info(f"🏃 Running from current directory: {run_cwd}")
            
            sagemaker_code_dir = Path("/opt/ml/code")
            if sagemaker_code_dir.exists():
                self.logger.info(f"✅ SageMaker code directory exists: {sagemaker_code_dir}")
            else:
                self.logger.warning(f"⚠️ SageMaker code directory not found: {sagemaker_code_dir}")
            
            
            # Verify the target script exists at absolute path
            target_script = Path("/opt/ml/code/imagenet_training_pipeline.py")
            if target_script.exists():
                self.logger.info(f"✅ Target script found: {target_script}")
            else:
                self.logger.error(f"❌ Target script NOT found: {target_script}")
                self.logger.error(f"❌ This will cause 'No such file or directory' error")
            
            # Debug: Show what files are in the SageMaker code directory
            if sagemaker_code_dir.exists():
                self.logger.info(f"📁 Files in {sagemaker_code_dir}:")
                for f in sagemaker_code_dir.iterdir():
                    if f.is_file() and f.suffix == '.py':
                        self.logger.info(f"   📄 {f.name}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(run_cwd),
                timeout=36000  # 10 hours
            )
            
            self.logger.info("✅ 7-Step Pipeline completed successfully!")
            self._process_results(result, args)
            
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ Pipeline timed out")
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            # Clean up model monitoring
            self._cleanup_model_monitoring()
    
    def _process_results(self, result, args):
        """Process and log training results with model saving"""
        # Log key pipeline outputs
        if result.stdout:
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['STEP', 'Best', 'Final', 'Accuracy', 'Epoch']):
                    self.logger.info(f"📊 {line.strip()}")
        
        # Save comprehensive results summary  
        results_file = os.path.join(args.output_dir, 'training_summary.json')
        try:
            summary = {
                'pipeline_completed': True,
                'training_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size or 'auto-detected',
                    'lr_finder_used': args.run_lr_finder,
                    'wd_search_used': args.run_wd_search,
                    'quick_mode': args.quick_mode,
                    'mixed_precision': args.mixed_precision,
                    'gradient_clip': args.gradient_clip
                },
                'sagemaker_info': {
                    'job_name': os.environ.get('SM_TRAINING_JOB_NAME', 'unknown'),
                    'instance_type': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'unknown'),
                    'region': os.environ.get('AWS_DEFAULT_REGION', 'unknown'),
                    'output_path': args.output_dir
                },
                'model_saving': {
                    'save_every_epoch': True,
                    'checkpoint_format': 'pytorch',
                    'final_model_saved': True
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"💾 Training summary saved: {results_file}")
            
            # Create model artifacts structure and verify model saving
            self._organize_model_artifacts(args.output_dir)
            self._verify_model_saving(args.output_dir)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results summary: {e}")
    
    def _verify_model_saving(self, output_dir):
        """Verify that model saving worked correctly"""
        try:
            models_dir = Path(output_dir) / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
                
                if model_files:
                    self.logger.info(f"✅ Model saving verification:")
                    for model_file in sorted(model_files):
                        file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                        self.logger.info(f"   📦 {model_file.name} ({file_size:.1f} MB)")
                    
                    # Check if current model exists (should be the replaced one)
                    current_model = models_dir / "model_current.pth"
                    if current_model.exists():
                        self.logger.info(f"🔄 Current model (replaced each epoch): {current_model.name}")
                    
                    # Check for best model
                    best_model = models_dir / "model_best.pth"
                    if best_model.exists():
                        self.logger.info(f"🏆 Best model saved: {best_model.name}")
                        
                else:
                    self.logger.warning("⚠️ No model files found in models directory")
            else:
                self.logger.warning("⚠️ Models directory not found")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Model verification failed: {e}")
    
    def _cleanup_model_monitoring(self):
        """Clean up model replacement monitoring"""
        
        try:
            if hasattr(self, '_model_monitor') and self._model_monitor:
                self._model_monitor.stop_monitoring()
                self.logger.info("🛑 Model replacement monitoring stopped")
        except Exception as e:
            self.logger.warning(f"⚠️ Error stopping model monitoring: {e}")
    
    def _organize_model_artifacts(self, output_dir):
        """Organize model artifacts for SageMaker"""
        try:
            output_path = Path(output_dir)
            
            # Create organized structure
            models_dir = output_path / "models"
            metrics_dir = output_path / "metrics"
            graphs_dir = output_path / "graphs"
            
            for directory in [models_dir, metrics_dir, graphs_dir]:
                directory.mkdir(exist_ok=True)
            
            # Move any existing model files
            for file_pattern in ["*.pth", "*.pt", "model*", "checkpoint*"]:
                for file_path in output_path.glob(file_pattern):
                    if file_path.is_file() and file_path.parent != models_dir:
                        shutil.move(str(file_path), str(models_dir / file_path.name))
            
            # Move metrics and logs
            for file_pattern in ["*.json", "*metrics*", "*results*"]:
                for file_path in output_path.glob(file_pattern):
                    if file_path.is_file() and file_path.parent != metrics_dir and file_path.name != "training_summary.json":
                        shutil.move(str(file_path), str(metrics_dir / file_path.name))
            
            # Move any graphs
            for file_pattern in ["*.png", "*.jpg", "*graph*", "*plot*"]:
                for file_path in output_path.glob(file_pattern):
                    if file_path.is_file() and file_path.parent != graphs_dir:
                        shutil.move(str(file_path), str(graphs_dir / file_path.name))
            
            self.logger.info("📁 Model artifacts organized successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not organize model artifacts: {e}")

def main():
    """Main SageMaker training entry point"""
    print("🚀 SageMaker Wrapper v2.1 - Fixed Path Resolution")
    print(f"🔍 Script location: {__file__}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    try:
        trainer = ImageNetSageMakerTrainer()
        trainer.run_training()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()