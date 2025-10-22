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

# Try to import tqdm for live progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

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

# Import unified logger - all files are in same directory now
try:
    from logger_setup import setup_unified_logger, get_unified_logger
except ImportError:
    from sagemaker_logging import setup_sagemaker_logger as setup_unified_logger
    from sagemaker_logging import setup_unified_logger, get_unified_logger

class ImageNetSageMakerTrainer:
    """Unified SageMaker wrapper for 7-step ImageNet training pipeline"""
    
    def __init__(self):
        # No need to change directories - use absolute paths
        print(f"🔄 INIT: Working from: {os.getcwd()}")
        
        # Set up unified logging for all components
        self.unified_logger = setup_unified_logger()
        self.logger = get_unified_logger("sagemaker_wrapper")
        
        # Create subprocess logger for detailed logging
        self.subprocess_logger = get_unified_logger("subprocess_monitor")
        
        self.config = {}
        
        # Double-check our working directory
        self.logger.info("="*80)
        self.logger.info("🚀 SAGEMAKER WRAPPER INITIALIZATION")
        self.logger.info("="*80)
        self.logger.info(f"🏠 SageMaker Wrapper initialized from: {os.getcwd()}")
        self.logger.info(f"📝 Unified log file: {getattr(self.unified_logger, 'unified_log_path', 'N/A')}")
        
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

            # =============================================================================
            # SUBPROCESS CALL TO IMAGENET_TRAINING_PIPELINE.PY
            # =============================================================================
            from datetime import datetime
            
            # Console output for immediate visibility
            print("=" * 80)
            print("🚀 SAGEMAKER WRAPPER CALLING IMAGENET_TRAINING_PIPELINE.PY")
            print("=" * 80)
            print(f"📞 Caller: {__file__}")
            print(f"🎯 Target Script: {cmd[1]}")
            print(f"🐍 Python Executable: {cmd[0]}")
            print(f"📋 Full Command: {' '.join(cmd)}")
            print(f"💻 Working Directory: {run_cwd}")
            print(f"⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print("🎬 SUBPROCESS OUTPUT STREAMING BELOW:")
            print("=" * 80)
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
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                cwd=str(run_cwd),
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Log process creation details
            self.subprocess_logger.info(f"🆔 Process ID: {process.pid}")
            self.subprocess_logger.info(f"📝 Process created successfully")
            self.logger.info(f"🚀 Subprocess started with PID: {process.pid}")
            
            # Stream output line by line with single updating progress bar using tqdm
            last_progress_line = ""
            progress_counter = 0
            line_counter = 0
            last_progress_percentage = -1
            current_step = "Unknown"
            
            # Initialize tqdm progress bar for live updates
            progress_bar = None
            current_progress_data = {"percentage": 0, "current": 0, "total": 100, "metrics": ""}
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    line_counter += 1
                    
                    # Clean up control characters (like \r) that cause formatting issues
                    clean_line = line.replace('\r', '').replace('\x15', '').strip()
                    if not clean_line:
                        continue
                    
                    # Log all output to file for complete record (use clean line)
                    self.subprocess_logger.debug(f"SUBPROCESS_OUTPUT[{line_counter:06d}]: {clean_line}")
                    
                    # Detect step changes for progress tracking
                    if any(step_indicator in clean_line for step_indicator in ["Starting Step", "STEP", "LR Range Test", "Weight Decay", "Training", "Validation"]):
                        if "LR Range Test" in clean_line:
                            current_step = "LR Range Test"
                            # Close previous progress bar if exists
                            if progress_bar:
                                progress_bar.close()
                            # Create new progress bar for LR Range Test
                            progress_bar = self._create_live_progress_bar(current_step, 200)  # Typical LR test iterations
                        elif "Weight Decay" in clean_line:
                            current_step = "Weight Decay Search"
                            if progress_bar:
                                progress_bar.close()
                            progress_bar = self._create_live_progress_bar(current_step, 5)  # Typical WD values to test
                        elif "Full Training" in clean_line or "Training" in clean_line:
                            current_step = "Training"
                            if progress_bar:
                                progress_bar.close()
                            progress_bar = self._create_live_progress_bar(current_step, 100)  # Epochs
                        elif "Validation" in clean_line:
                            current_step = "Validation"
                    
                    # Handle progress bar updates with live tqdm
                    if self._is_progress_bar_line(clean_line):
                        # Parse progress information
                        progress_info = self._parse_progress_info(clean_line)
                        if progress_info and progress_bar:
                            # Update the live progress bar
                            self._update_live_progress_bar(progress_bar, progress_info, current_step)
                        
                        # Only log major milestones to prevent spam
                        current_percentage = self._extract_percentage(clean_line)
                        if current_percentage and self._is_progress_milestone(clean_line):
                            self.logger.info(f"📊 Progress milestone: {current_step} - {current_percentage}%")
                        
                        last_progress_line = clean_line
                    else:
                        # Regular log lines - always show and log
                        print(clean_line)
                        sys.stdout.flush()
                        # Log important non-progress lines
                        if any(keyword in clean_line.lower() for keyword in ['error', 'warning', 'step', 'epoch', 'starting', 'completed', 'failed', 'success']):
                            self.logger.info(f"SUBPROCESS: {clean_line}")
                        last_progress_line = clean_line
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            end_time = datetime.now()
            duration = end_time - start_time
            
            # =============================================================================
            # SUBPROCESS COMPLETED
            # =============================================================================
            print("=" * 80)
            print("✅ IMAGENET_TRAINING_PIPELINE.PY SUBPROCESS COMPLETED")
            print("=" * 80)
            print(f"📊 Return Code: {return_code}")
            print(f"⏰ Completion Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⌛ Total Duration: {duration}")
            print(f"📝 Total output lines: {line_counter}")
            print("=" * 80)
            sys.stdout.flush()
            
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
            self.subprocess_logger.info(f"📊 Progress updates shown: {progress_counter}")
            
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
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.logger.error(f"Return code: {e.returncode}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during pipeline execution: {e}")
            raise
        finally:
            # Clean up progress bar if still active
            if progress_bar:
                progress_bar.close()
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
    
    def _is_progress_bar_line(self, line):
        """Check if a line contains a progress bar (tqdm output)"""
        # Look for common progress bar indicators
        progress_indicators = ['|', '%', 'it/s', 's/it', '[', ']', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
        
        # Also check for common progress bar patterns
        has_percentage = any(c.isdigit() for c in line) and '%' in line
        has_progress_chars = any(char in line for char in progress_indicators)
        has_rate = 'it/s' in line or 's/it' in line
        
        # Common progress bar prefixes in your training
        progress_prefixes = ['LR Range Test:', 'Training:', 'Validation:', 'Epoch', 'Testing:']
        has_progress_prefix = any(prefix in line for prefix in progress_prefixes)
        
        return (has_percentage and has_progress_chars) or (has_rate and has_progress_prefix)
    
    def _is_progress_milestone(self, line):
        """Check if this is an important progress milestone to always show"""
        # Show progress at certain percentages
        milestones = ['0%', '10%', '25%', '50%', '75%', '90%', '100%']
        return any(milestone in line for milestone in milestones)
    
    def _extract_percentage(self, line):
        """Extract percentage from progress bar line"""
        import re
        # Look for patterns like "42%" or "42.5%"
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
        if percentage_match:
            try:
                return int(float(percentage_match.group(1)))
            except ValueError:
                return None
        return None
    
    def _parse_progress_info(self, line):
        """Parse detailed progress information from tqdm progress bar"""
        import re
        
        info = {}
        
        # Extract percentage
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
        if percentage_match:
            info['percentage'] = int(float(percentage_match.group(1)))
        
        # Extract current/total (e.g., "50/200")
        progress_match = re.search(r'(\d+)/(\d+)', line)
        if progress_match:
            info['current'] = int(progress_match.group(1))
            info['total'] = int(progress_match.group(2))
        
        # Extract rate (e.g., "1.41s/it" or "6.02it/s")
        rate_match = re.search(r'(\d+\.\d+(?:s/it|it/s))', line)
        if rate_match:
            info['rate'] = rate_match.group(1)
        
        # Extract ETA (e.g., "04:35")
        eta_match = re.search(r'<(\d+:\d+)', line)
        if eta_match:
            info['eta'] = eta_match.group(1)
        
        # Extract metrics (LR, Loss, Acc, etc.)
        metrics = []
        lr_match = re.search(r'LR=([^,\]]+)', line)
        if lr_match:
            metrics.append(f"LR={lr_match.group(1)}")
        
        loss_match = re.search(r'Loss=([^,\]]+)', line)
        if loss_match:
            metrics.append(f"Loss={loss_match.group(1)}")
        
        acc_match = re.search(r'Acc=([^,\]]+)', line)
        if acc_match:
            metrics.append(f"Acc={acc_match.group(1)}")
        
        if metrics:
            info['metrics'] = ", ".join(metrics)
        
        return info if info else None
    
    def _create_live_progress_bar(self, step_name, total_steps):
        """Create a new tqdm progress bar for live updates"""
        if not TQDM_AVAILABLE:
            self.logger.warning("tqdm not available, falling back to simple progress display")
            return None
            
        try:
            return tqdm(
                total=total_steps,
                desc=f"🔄 {step_name}",
                unit="it",
                ncols=100,
                position=0,
                leave=False,  # Don't leave the progress bar after completion - this prevents multiple bars
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]'
            )
        except Exception as e:
            self.logger.warning(f"Failed to create tqdm progress bar: {e}")
            return None
    
    def _update_live_progress_bar(self, progress_bar, progress_info, current_step):
        """Update the live progress bar with new information"""
        if not progress_bar:
            return
        
        try:
            # Extract current progress value
            current_value = 0
            if 'percentage' in progress_info:
                current_value = int(progress_info['percentage'] * progress_bar.total / 100)
            elif 'current' in progress_info and 'total' in progress_info:
                current_value = progress_info['current']
            
            # Update progress bar
            progress_bar.n = current_value
            
            # Add additional info to description if available
            desc = f"🔄 {current_step}"
            if 'loss' in progress_info:
                desc += f" | Loss: {progress_info['loss']:.4f}"
            if 'accuracy' in progress_info:
                desc += f" | Acc: {progress_info['accuracy']:.2f}%"
            if 'lr' in progress_info:
                desc += f" | LR: {progress_info['lr']:.6f}"
            
            progress_bar.set_description(desc)
            progress_bar.refresh()
            
        except Exception as e:
            self.logger.debug(f"Progress bar update error: {e}")
    
    def _display_single_progress_bar(self, progress_info):
        """Display a single, consolidated progress bar (fallback if tqdm not available)"""
        step = progress_info.get('step', 'Training')
        percentage = progress_info.get('percentage', 0)
        current = progress_info.get('current', 0)
        total = progress_info.get('total', 100)
        rate = progress_info.get('rate', '')
        eta = progress_info.get('eta', '')
        metrics = progress_info.get('metrics', '')
        
        # Create progress bar visual
        bar_width = 20
        filled = int((percentage / 100) * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Format the single progress line
        progress_line = f"🔄 {step}: {percentage:3d}% |{bar}| {current}/{total}"
        
        if rate:
            progress_line += f" [{rate}]"
        if eta:
            progress_line += f" ETA: {eta}"
        if metrics:
            progress_line += f" | {metrics}"
        
        print(f"\r{progress_line:<120}", end='')
        sys.stdout.flush()
        
        # Add newline for milestones to preserve in logs
        if percentage % 25 == 0 or percentage in [10, 50, 75, 90, 100]:
            print()  # Add newline for milestone
            sys.stdout.flush()

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