#!/usr/bin/env python3
"""
Test script to verify the SageMaker wrapper path resolution works correctly
"""

import sys
import os
from pathlib import Path

# Simulate the SageMaker environment structure
def test_wrapper_paths():
    """Test the path resolution logic from sagemaker_wrapper.py"""
    
    print("🧪 Testing SageMaker wrapper path resolution...")
    
    # Get the current working directory structure
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    print(f"📂 Current script location: {__file__}")
    print(f"📂 Current directory: {current_dir}")
    print(f"📂 Parent directory: {parent_dir}")
    
    # Check if imagenet_training_pipeline.py exists
    pipeline_script = parent_dir / "imagenet_training_pipeline.py"
    print(f"📂 Pipeline script path: {pipeline_script}")
    print(f"✅ Pipeline script exists: {pipeline_script.exists()}")
    
    if pipeline_script.exists():
        print("✅ Path resolution should work correctly in SageMaker")
    else:
        print("❌ Path resolution issue - need to debug further")
    
    # List Python files in parent directory
    if parent_dir.exists():
        print(f"\n📂 Python files in parent directory:")
        py_files = [f for f in parent_dir.iterdir() if f.is_file() and f.suffix == '.py']
        for f in sorted(py_files):
            print(f"    - {f.name}")
            
        print(f"\n📊 Total Python files found: {len(py_files)}")
    
    # Simulate SageMaker structure
    print(f"\n🔍 Simulating SageMaker directory structure:")
    print(f"   Local: {current_dir} -> {parent_dir}")
    print(f"   SageMaker: /opt/ml/code/sagemaker_training -> /opt/ml/code")
    print(f"   Expected pipeline location: /opt/ml/code/imagenet_training_pipeline.py")
    
    return pipeline_script.exists()

if __name__ == "__main__":
    success = test_wrapper_paths()
    print(f"\n🎯 Test result: {'✅ PASS' if success else '❌ FAIL'}")
    exit(0 if success else 1)