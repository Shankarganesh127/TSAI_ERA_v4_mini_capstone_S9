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
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Try to import from parent directory first, fallback to local
try:
    from logger_setup import setup_logger
except ImportError:
    # Fallback to local SageMaker logging
    from sagemaker_logging import setup_sagemaker_logger as setup_logger

class ImageNetSageMakerTrainer:
    """Unified SageMaker wrapper for 7-step ImageNet training pipeline"""
    
    def __init__(self):
        self.logger = setup_logger("sagemaker_imagenet_trainer")
        self.config = {}
        
    def parse_hyperparameters(self):
        """Parse SageMaker hyperparameters"""
        parser = argparse.ArgumentParser()
        
        # Core training parameters
        parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/imagenet')
        parser.add_argument('--output_dir', type=str, default='/opt/ml/model')
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
        
        return args
    
    def build_pipeline_command(self, args):
        """Build command for 7-step pipeline execution"""
        cmd = [
            sys.executable,
            os.path.join(parent_dir, "imagenet_training_pipeline.py"),
            "--data", str(args.data_dir),
            "--output", str(args.output_dir), 
            "--epochs", str(args.epochs)
        ]
        
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
        
        # Build and execute pipeline command
        cmd = self.build_pipeline_command(args)
        self.logger.info(f"🎯 Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd=parent_dir,
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
    
    def _process_results(self, result, args):
        """Process and log training results"""
        # Log key pipeline outputs
        if result.stdout:
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['STEP', 'Best', 'Final', 'Accuracy']):
                    self.logger.info(f"📊 {line.strip()}")
        
        # Save results summary
        results_file = os.path.join(args.output_dir, 'training_summary.json')
        try:
            summary = {
                'pipeline_completed': True,
                'epochs': args.epochs,
                'batch_size': args.batch_size or 'auto-detected',
                'lr_finder_used': args.run_lr_finder,
                'wd_search_used': args.run_wd_search,
                'quick_mode': args.quick_mode
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results summary: {e}")

def main():
    """Main SageMaker training entry point"""
    try:
        trainer = ImageNetSageMakerTrainer()
        trainer.run_training()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()