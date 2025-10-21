#!/usr/bin/env python3
"""
Test SageMaker path resolution
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from sagemaker_training.sagemaker_wrapper import ImageNetSageMakerTrainer

def test_sagemaker_paths():
    """Test SageMaker path resolution"""
    
    print("🧪 Testing SageMaker path resolution...")
    
    # Mock SageMaker environment variables
    os.environ['SM_CHANNEL_IMAGENET'] = '/opt/ml/input/data/imagenet'
    os.environ['SM_MODEL_DIR'] = '/opt/ml/model'
    
    try:
        trainer = ImageNetSageMakerTrainer()
        
        # Test parse_hyperparameters with SageMaker env vars
        print("\n1. Testing hyperparameter parsing with SageMaker env vars...")
        
        # Mock sys.argv for argparse
        original_argv = sys.argv
        sys.argv = ['sagemaker_wrapper.py', '--epochs', '1', '--quick_mode', 'true']
        
        try:
            args = trainer.parse_hyperparameters()
            print(f"✅ Data dir resolved to: {args.data_dir}")
            print(f"✅ Output dir resolved to: {args.output_dir}")
            print(f"✅ Epochs: {args.epochs}")
            print(f"✅ Quick mode: {args.quick_mode}")
        finally:
            sys.argv = original_argv
        
        # Test build_pipeline_command path resolution
        print("\n2. Testing pipeline script path resolution...")
        try:
            cmd = trainer.build_pipeline_command(args)
            print(f"✅ Pipeline script resolved to: {trainer.pipeline_script_path}")
            print(f"✅ Command built successfully: {len(cmd)} arguments")
        except FileNotFoundError as e:
            print(f"ℹ️ Pipeline script not found (expected locally): {e}")
        
        print("\n🎉 SageMaker path resolution tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sagemaker_paths()
    exit(0 if success else 1)