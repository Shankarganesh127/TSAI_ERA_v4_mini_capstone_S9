#!/usr/bin/env python3
"""
Test SageMaker Environment Variables Handling
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def test_sagemaker_env_vars():
    """Test how SageMaker environment variables work"""
    
    print("🧪 Testing SageMaker Environment Variables...")
    
    # Show current environment variables
    print("\n📋 Current SageMaker Environment Variables:")
    sagemaker_vars = {k: v for k, v in os.environ.items() if k.startswith('SM_')}
    
    if sagemaker_vars:
        for key, value in sorted(sagemaker_vars.items()):
            print(f"   {key}={value}")
    else:
        print("   ❌ No SageMaker environment variables found (not running on SageMaker)")
    
    # Test single channel configuration
    print("\n🔍 Testing Single Channel Configuration:")
    print("   Setting: data_inputs = {'imagenet': train_input}")
    print("   Result: SM_CHANNEL_IMAGENET → /opt/ml/input/data/imagenet")
    
    train_data_path = os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet')
    print(f"   ✅ Train data path: {train_data_path}")
    
    # Test multi-channel configuration  
    print("\n🔍 Testing Multi-Channel Configuration:")
    print("   Setting: data_inputs = {'imagenet': train_input, 'validation': val_input}")
    print("   Result: SM_CHANNEL_IMAGENET → /opt/ml/input/data/imagenet")
    print("   Result: SM_CHANNEL_VALIDATION → /opt/ml/input/data/validation")
    
    train_data_path = os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet')
    val_data_path = os.environ.get('SM_CHANNEL_VALIDATION', None)
    
    print(f"   ✅ Train data path: {train_data_path}")
    print(f"   ✅ Validation data path: {val_data_path or 'Not configured (single channel mode)'}")
    
    # Test hyperparameters
    print("\n🔍 Testing Hyperparameter Environment Variables:")
    hyperparams = {
        'SM_HP_EPOCHS': os.environ.get('SM_HP_EPOCHS', 'Not set'),
        'SM_HP_NUM_WORKERS': os.environ.get('SM_HP_NUM_WORKERS', 'Not set'),
        'SM_HP_QUICK_MODE': os.environ.get('SM_HP_QUICK_MODE', 'Not set'),
        'SM_HP_RUN_LR_FINDER': os.environ.get('SM_HP_RUN_LR_FINDER', 'Not set'),
        'SM_HP_RUN_WD_SEARCH': os.environ.get('SM_HP_RUN_WD_SEARCH', 'Not set')
    }
    
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")
    
    # Test model and output paths
    print("\n🔍 Testing Model and Output Paths:")
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    
    print(f"   Model output: {model_dir}")
    print(f"   Data output: {output_dir}")
    
    # Test instance information
    print("\n🔍 Testing Instance Information:")
    instance_info = {
        'SM_CURRENT_INSTANCE_TYPE': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'Not set'),
        'SM_NUM_GPUS': os.environ.get('SM_NUM_GPUS', 'Not set'),
        'SM_NUM_CPUS': os.environ.get('SM_NUM_CPUS', 'Not set'),
        'SM_TRAINING_JOB_NAME': os.environ.get('SM_TRAINING_JOB_NAME', 'Not set')
    }
    
    for key, value in instance_info.items():
        print(f"   {key}: {value}")
    
    # Simulate how sagemaker_wrapper.py uses environment variables
    print("\n🔧 Testing sagemaker_wrapper.py Environment Variable Usage:")
    try:
        from sagemaker_training.sagemaker_wrapper import ImageNetSageMakerTrainer
        
        # Mock command line arguments
        original_argv = sys.argv
        sys.argv = ['sagemaker_wrapper.py', '--epochs', '1']
        
        try:
            trainer = ImageNetSageMakerTrainer()
            args = trainer.parse_hyperparameters()
            
            print(f"   ✅ Parsed data_dir: {args.data_dir}")
            print(f"   ✅ Parsed val_dir: {args.val_dir}")
            print(f"   ✅ Parsed output_dir: {args.output_dir}")
            print(f"   ✅ Parsed epochs: {args.epochs}")
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"   ❌ Error testing wrapper: {e}")
    
    print("\n🎉 Environment variable testing completed!")
    
    # Summary
    print("\n📝 Summary:")
    print("   - Environment variables are set AUTOMATICALLY by SageMaker")
    print("   - Channel names in data_inputs become SM_CHANNEL_{NAME}")
    print("   - Hyperparameters become SM_HP_{NAME}")
    print("   - Standard paths are always available (SM_MODEL_DIR, etc.)")
    print("   - Your code reads these variables to find data and output locations")

if __name__ == "__main__":
    test_sagemaker_env_vars()