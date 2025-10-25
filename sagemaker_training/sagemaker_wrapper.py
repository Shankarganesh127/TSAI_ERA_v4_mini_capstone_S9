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

# Import unified logger - all files are in same directory now
try:
    from logger_setup import setup_unified_logger, get_unified_logger
except ImportError:
    from logger_setup import setup_unified_logger, get_unified_logger

class ImageNetSageMakerTrainer:
    """Unified SageMaker wrapper for 7-step ImageNet training pipeline"""
    
    def __init__(self):
        # Initialize logger attributes first to ensure they're always available
        self.logger = None
        self.unified_logger = None
        self.subprocess_logger = None
        
        try:
            # Set up logging
            self.unified_logger = setup_unified_logger()
            self.logger = get_unified_logger("sagemaker_wrapper")
            self.subprocess_logger = get_unified_logger("subprocess_monitor")
            
            # Set memory fragmentation fix BEFORE any PyTorch operations
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
                if self.logger:
                    self.logger.info("🔧 Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 to prevent memory fragmentation")
            
            # No need to change directories - use absolute paths
            self.config = {}
            if self.logger:
                self.logger.info(f"🔄 INIT: Working from: {os.getcwd()}")
                self.logger.info("="*80)
                self.logger.info("🚀 SAGEMAKER WRAPPER INITIALIZATION")
                self.logger.info("="*80)
                self.logger.info(f"🏠 SageMaker Wrapper initialized from: {os.getcwd()}")
                self.logger.info(f"📝 Unified log file: {getattr(self.unified_logger, 'unified_log_path', 'N/A')}")
                if Path("imagenet_training_pipeline.py").exists():
                    self.logger.info("✅ Found imagenet_training_pipeline.py in current directory")
                else:
                    self.logger.error("❌ imagenet_training_pipeline.py NOT found in current directory!")
                    
        except Exception as e:
            # Fallback logger setup if unified logging fails
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger("sagemaker_wrapper_fallback")
            self.logger.error(f"Failed to setup unified logger, using fallback: {e}")
            self.unified_logger = self.logger
            self.subprocess_logger = self.logger
        
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
        # Check Python executable
        python_executable = sys.executable
        if not Path(python_executable).exists():
            self.logger.error(f"❌ Python executable not found: {python_executable}")
            raise FileNotFoundError(f"Python executable not found: {python_executable}")

        # Check train and val data directories
        train_path = Path(args.data_dir)
        val_path = Path(args.val_dir)
        if not train_path.exists():
            self.logger.error(f"❌ Training data directory not found: {train_path}")
            raise FileNotFoundError(f"Training data directory not found: {train_path}")
        if not val_path.exists():
            self.logger.error(f"❌ Validation data directory not found: {val_path}")
            raise FileNotFoundError(f"Validation data directory not found: {val_path}")
        """Build command for 7-step pipeline execution with model saving"""
        # In SageMaker, all source code is uploaded to /opt/ml/code/
        
        # EARLY DEBUG - ensure this function is being called
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
            "--train", str(args.data_dir),
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
        #if args.batch_size:
        #    cmd.extend(["--batch-size", str(args.batch_size)])
        #    self.logger.info(f"🔧 Batch Size Override: {args.batch_size}")
        #else:
        #    self.logger.info("🔄 Using automatic batch size detection (Step 4)")
        
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
        if not self.logger:
            raise RuntimeError("Logger not initialized")
            
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
        
        # Log all hyperparameters
        self.logger.info("🔧 Hyperparameters Configuration:")
        self.logger.info(f"   data_dir: {args.data_dir}")
        self.logger.info(f"   val_dir: {args.val_dir}")
        self.logger.info(f"   output_dir: {args.output_dir}")
        self.logger.info(f"   epochs: {args.epochs}")
        self.logger.info(f"   run_lr_finder: {args.run_lr_finder}")
        self.logger.info(f"   run_wd_search: {args.run_wd_search}")
        self.logger.info(f"   quick_mode: {args.quick_mode}")
        self.logger.info(f"   mixed_precision: {args.mixed_precision}")
        self.logger.info(f"   gradient_clip: {args.gradient_clip}")
        if args.lr_min and args.lr_max:
            self.logger.info(f"   manual_lr_bounds: {args.lr_min:.2e} → {args.lr_max:.2e}")
        if args.weight_decay:
            self.logger.info(f"   manual_weight_decay: {args.weight_decay:.2e}")
        self.logger.info("=" * 60)
        
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
            from datetime import datetime
            self.logger.info("=" * 80)
            self.logger.info("🚀 SAGEMAKER WRAPPER CALLING IMAGENET_TRAINING_PIPELINE.PY")
            self.logger.info("=" * 80)
            self.logger.info(f"📞 Caller: {__file__}")
            self.logger.info(f"🎯 Target Script: {cmd[1]}")
            self.logger.info(f"🐍 Python Executable: {cmd[0]}")
            self.logger.info(f"📋 Full Command: {' '.join(cmd)}")
            self.logger.info(f"💻 Working Directory: {run_cwd}")
            self.logger.info(f"⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"❌ Exception during SageMaker code/script checks: {e}")
            self.logger.info("🎬 SUBPROCESS OUTPUT STREAMING BELOW:")
            self.logger.info("=" * 80)
            sys.stdout.flush()
            
        # Detailed logging to unified log file
        self.logger.info("="*60)
        self.logger.info("� SUBPROCESS: Launching imagenet_training_pipeline.py")
        self.logger.info("="*60)
        self.logger.info(f"📞 Caller script: {__file__}")
        self.logger.info(f"🎯 Target script: {cmd[1]}")
        self.logger.info(f"🐍 Python executable: {cmd[0]}")
        self.logger.info(f"🎯 Full command: {' '.join(cmd)}")
        self.logger.info(f"💻 Working directory: {run_cwd}")
        if len(cmd) > 2:
            self.logger.info(f"⚙️  Script arguments: {' '.join(map(str, cmd[2:]))}")
            
        # Log environment variables that might affect execution
        self.subprocess_logger.info("🌍 Subprocess environment variables:")
        for key in ['PYTHONPATH', 'PATH', 'CUDA_VISIBLE_DEVICES', 'SM_MODEL_DIR', 'SM_CHANNEL_TRAINING']:
            value = os.environ.get(key, 'Not set')
            self.subprocess_logger.info(f"   {key}={value}")
            
        # Log process startup details
        start_time = datetime.now()
        self.subprocess_logger.info(f"⏰ Subprocess start time: {start_time}")
        self.logger.info("🔥 Starting subprocess execution...")

        # Stream subprocess output in real-time instead of capturing
        # Set environment to disable tqdm in subprocess to prevent progress bar spam
        subprocess_env = os.environ.copy()
        subprocess_env['TQDM_DISABLE'] = '0'  # 0 for Enable tqdm in subprocess
        subprocess_env['PYTHONUNBUFFERED'] = '1'  # Ensure immediate output
            
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            cwd=str(run_cwd),
            bufsize=1,  # Line buffered
            env=subprocess_env,  # Use modified environment
            universal_newlines=True
        )
            
        # Log process creation details
        self.subprocess_logger.info(f"🆔 Process ID: {process.pid}")
        self.subprocess_logger.info(f"📝 Process created successfully")
        self.logger.info(f"🚀 Subprocess started with PID: {process.pid}")
            
        # Stream output line by line - simplified without progress bar parsing
        last_progress_line = ""
        line_counter = 0
            
        # Log process creation details
        self.subprocess_logger.info(f"🆔 Process ID: {process.pid}")
        self.subprocess_logger.info(f"📝 Process created successfully")
        self.logger.info(f"🚀 Subprocess started with PID: {process.pid}")

        # Stream output line by line and log to both loggers
        line_counter = 0
        try:
            for line in process.stdout:
                line_counter += 1
                line = line.rstrip()
                # Log every line to both loggers
                self.logger.info(f"[PIPELINE] {line}")
                self.subprocess_logger.info(f"[PIPELINE] {line}")
        except Exception as stream_exc:
            self.logger.error(f"❌ Error streaming subprocess output: {stream_exc}")
            self.subprocess_logger.error(f"❌ Error streaming subprocess output: {stream_exc}")
        finally:
            try:
                return_code = process.wait()
                end_time = datetime.now()
                duration = end_time - start_time
            except Exception as wait_exc:
                self.logger.error(f"❌ Error waiting for subprocess: {wait_exc}")
                self.subprocess_logger.error(f"❌ Error waiting for subprocess: {wait_exc}")

        # =============================================================================
        # SUBPROCESS COMPLETED
        # =============================================================================
        self.logger.info("=" * 80)
        self.logger.info("✅ IMAGENET_TRAINING_PIPELINE.PY SUBPROCESS COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Return Code: {return_code}")
        self.logger.info(f"⏰ Completion Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"⌛ Total Duration: {duration}")
        self.logger.info(f"📝 Total output lines: {line_counter}")
        self.logger.info("=" * 80)

        # Comprehensive subprocess completion logging
        self.subprocess_logger.info("="*60)
        self.subprocess_logger.info("🏁 SUBPROCESS EXECUTION COMPLETED")
        self.subprocess_logger.info("="*60)
        self.subprocess_logger.info(f"🆔 Process ID: {process.pid}")
        self.subprocess_logger.info(f"📊 Exit code: {return_code}")
        self.subprocess_logger.info(f"⏰ Start time: {start_time}")
        self.subprocess_logger.info(f"⏰ End time: {end_time}")
        self.subprocess_logger.info(f"⌛ Total duration: {duration}")
        self.subprocess_logger.info(f"📝 Total output lines captured: {line_counter}")

        if return_code == 0:
            self.logger.info("✅ Subprocess completed successfully")
            self.subprocess_logger.info("✅ Process exited normally")
        else:
            self.logger.error(f"❌ Subprocess failed with return code: {return_code}")
            self.subprocess_logger.error(f"❌ Process failed with exit code: {return_code}")

        # Check if subprocess failed
        if return_code != 0:
            self.logger.error(f"❌ Pipeline failed with return code: {return_code}")
            raise subprocess.CalledProcessError(return_code, cmd)

        self.logger.info("✅ 7-Step Pipeline completed successfully!")

        # Create a mock result object for compatibility with _process_results
        class MockResult:
            def __init__(self, returncode):
                self.returncode = returncode
                self.stdout = ""  # Output was already streamed
                self.stderr = ""

        self._process_results(MockResult(return_code), args)
        # Clean up model monitoring (progress bars now handled in pipeline)
        self._cleanup_model_monitoring()
    
        # Save comprehensive results summary  
        results_file = os.path.join(args.output_dir, 'training_summary.json')
        # Always save logs to the specified absolute path
        local_logs_dir = '/home/sagemaker-user/TSAI_ERA_v4_mini_capstone_S9/sagemaker_training/logs'
        os.makedirs(local_logs_dir, exist_ok=True)
        log_file_path = os.path.join(local_logs_dir, 'training_log.txt')
        try:
            summary = {
                # ...existing code...
            }
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"💾 Training summary saved: {results_file}")
            self._organize_model_artifacts(args.output_dir)
            self._verify_model_saving(args.output_dir)
            # Save logs to text file in specified logs directory
            if hasattr(self.logger, 'get_log_contents'):
                log_contents = self.logger.get_log_contents()
                with open(log_file_path, 'w', encoding='utf-8') as logf:
                    logf.write(log_contents)
                self.logger.info(f"📝 Training log saved to: {log_file_path}")
            else:
                # Fallback: Write a message if logger does not support get_log_contents
                with open(log_file_path, 'w', encoding='utf-8') as logf:
                    logf.write('Logger does not support get_log_contents. Please check console output.')
                self.logger.info(f"📝 Training log saved to: {log_file_path} (fallback mode)")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results summary or logs: {e}")

    def _process_results(self, result, args):
        """Process and log training results with model saving"""
        # Log key pipeline outputs
        if result.stdout:
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['STEP', 'Best', 'Final', 'Accuracy', 'Epoch']):
                    self.logger.info(f"📊 {line.strip()}")
        
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
    import logger_setup
    logger = logger_setup.get_unified_logger("sagemaker_wrapper_main")
    logger.info("🚀 SageMaker Wrapper v2.1 - Fixed Path Resolution")
    logger.info(f"🔍 Script location: {__file__}")
    logger.info(f"🔍 Current working directory: {os.getcwd()}")
    try:
        trainer = ImageNetSageMakerTrainer()
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            raise RuntimeError("ImageNetSageMakerTrainer logger not properly initialized")
        trainer.run_training()
        logger.info("🎉 Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"   {line}")
        sys.exit(1)

if __name__ == "__main__":
    if (int(os.environ.get('LOCAL_RANK', 0)) == 0):
        main()