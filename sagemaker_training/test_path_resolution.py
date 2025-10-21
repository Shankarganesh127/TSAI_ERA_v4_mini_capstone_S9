#!/usr/bin/env python3
"""
Test path resolution logic for sagemaker_wrapper.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def test_path_resolution():
    """Test the path resolution logic"""
    
    print("🧪 Testing SageMaker path resolution logic...")
    
    # Mock SageMaker environment
    print("\n🔍 Testing path resolution scenarios:")
    
    # Scenario 1: SageMaker container structure
    print("\n1. SageMaker Container Scenario:")
    print("   Wrapper at: /opt/ml/code/sagemaker_training/sagemaker_wrapper.py")
    print("   Pipeline at: /opt/ml/code/imagenet_training_pipeline.py")
    
    # Calculate parent_dir as it would be in SageMaker
    mock_wrapper_path = Path("/opt/ml/code/sagemaker_training/sagemaker_wrapper.py")
    mock_parent_dir = mock_wrapper_path.parent.parent
    
    print(f"   Calculated parent_dir: {mock_parent_dir}")
    print(f"   Expected pipeline path: {mock_parent_dir / 'imagenet_training_pipeline.py'}")
    
    # Scenario 2: Current local structure
    print("\n2. Local Development Scenario:")
    print(f"   Current wrapper: {__file__}")
    print(f"   Current parent_dir: {parent_dir}")
    print(f"   Local pipeline path: {parent_dir / 'imagenet_training_pipeline.py'}")
    print(f"   Local pipeline exists: {(parent_dir / 'imagenet_training_pipeline.py').exists()}")
    
    # Test the actual wrapper logic
    print("\n3. Testing actual wrapper logic:")
    try:
        from sagemaker_training.sagemaker_wrapper import ImageNetSageMakerTrainer
        
        # Mock command line arguments
        original_argv = sys.argv
        sys.argv = ['sagemaker_wrapper.py', '--epochs', '1']
        
        try:
            trainer = ImageNetSageMakerTrainer()
            args = trainer.parse_hyperparameters()
            
            # This will trigger the path resolution logic
            print("   🔍 Testing build_pipeline_command...")
            cmd = trainer.build_pipeline_command(args)
            
            print(f"   ✅ Command built successfully")
            print(f"   ✅ Pipeline script resolved to: {trainer.pipeline_script_path}")
            print(f"   ✅ Command: {' '.join(cmd[:5])}...")  # Show first 5 parts
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"   ❌ Error testing wrapper: {e}")
    
    print("\n🎉 Path resolution testing completed!")

if __name__ == "__main__":
    test_path_resolution()