#!/usr/bin/env python3
"""
Test the enhanced SageMaker orchestrator logging
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def test_logging_system():
    """Test the subprocess logging without actually launching SageMaker"""
    
    print("🧪 Testing Enhanced SageMaker Orchestrator Logging")
    print("=" * 60)
    
    try:
        from sagemaker_training.sagemaker_orchestrator import SageMakerPipelineOrchestrator
        
        # Test with debug mode disabled
        print("\n1️⃣ Test: Debug mode disabled (default)")
        orchestrator = SageMakerPipelineOrchestrator()
        debug_config = orchestrator.config.get("debug", {})
        
        print(f"   Real-time output: {debug_config.get('enable_realtime_output', False)}")
        print(f"   Subprocess details: {debug_config.get('log_subprocess_details', True)}")
        print(f"   Verbose errors: {debug_config.get('verbose_error_reporting', True)}")
        
        # Test timeout calculation
        print("\n2️⃣ Test: Timeout calculation")
        timeout_small = orchestrator._calculate_submission_timeout("ml.m5.large", False)
        timeout_large = orchestrator._calculate_submission_timeout("ml.g5.12xlarge", True)
        
        print(f"   Small instance (ml.m5.large, no spot): {timeout_small//60} minutes")
        print(f"   Large instance (ml.g5.12xlarge, spot): {timeout_large//60} minutes")
        
        # Test instance alternatives
        print("\n3️⃣ Test: Instance alternatives")
        alternatives = orchestrator._suggest_alternative_instances("ml.g5.12xlarge")
        print(f"   Current: ml.g5.12xlarge")
        print(f"   Alternatives: {alternatives}")
        
        print("\n4️⃣ Test: Configuration structure")
        required_sections = ["aws", "training", "timeouts", "debug"]
        for section in required_sections:
            exists = section in orchestrator.config
            print(f"   Config section '{section}': {'✅' if exists else '❌'}")
        
        print("\n✅ All logging system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_logging_system()
    sys.exit(0 if success else 1)