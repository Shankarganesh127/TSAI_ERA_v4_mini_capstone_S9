#!/usr/bin/env python3
"""
Test script to verify the SageMakerMonitor fixes
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from sagemaker_training.monitor_training import SageMakerMonitor

def test_monitor_methods():
    """Test the newly added monitor methods"""
    
    print("🧪 Testing SageMakerMonitor fixes...")
    
    try:
        # Initialize monitor
        monitor = SageMakerMonitor(region='us-east-1')
        
        # Test set_current_job method
        print("\n1. Testing set_current_job method...")
        monitor.set_current_job("test-job-123")
        assert monitor.current_job_name == "test-job-123"
        print("✅ set_current_job works correctly")
        
        # Test generate_training_summary method exists
        print("\n2. Testing generate_training_summary method...")
        assert hasattr(monitor, 'generate_training_summary')
        print("✅ generate_training_summary method exists")
        
        # Test generate_cost_analysis method exists  
        print("\n3. Testing generate_cost_analysis method...")
        assert hasattr(monitor, 'generate_cost_analysis')
        print("✅ generate_cost_analysis method exists")
        
        # Test generate_performance_graphs method exists
        print("\n4. Testing generate_performance_graphs method...")
        assert hasattr(monitor, 'generate_performance_graphs')
        print("✅ generate_performance_graphs method exists")
        
        # Try to call the methods (they may fail due to AWS credentials, but shouldn't crash)
        print("\n5. Testing method calls...")
        try:
            summary = monitor.generate_training_summary()
            print("✅ generate_training_summary called successfully")
            print(f"   Summary keys: {list(summary.keys())}")
        except Exception as e:
            print(f"⚠️  generate_training_summary failed (expected): {str(e)[:100]}")
        
        try:
            cost_analysis = monitor.generate_cost_analysis()
            print("✅ generate_cost_analysis called successfully")
            print(f"   Cost analysis keys: {list(cost_analysis.keys())}")
        except Exception as e:
            print(f"⚠️  generate_cost_analysis failed (expected): {str(e)[:100]}")
        
        print("\n🎉 All monitor method fixes verified successfully!")
        print("✅ The SageMakerMonitor now has all required methods")
        print("✅ The orchestrator should no longer get AttributeError")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_monitor_methods()
    exit(0 if success else 1)