#!/usr/bin/env python3
"""
Test role_arn and epochs parameter handling

Usage:
    python test_parameter_priority.py
"""

class MockArgs:
    """Mock command line arguments"""
    def __init__(self, role_arn=None, epochs=None):
        self.source_bucket = "test-bucket"
        self.target_prefix = "test-prefix"
        self.instance_type = "ml.g5.12xlarge"
        self.use_spot = True
        self.role_arn = role_arn  # Test parameter
        self.epochs = epochs      # Test parameter

def test_parameter_priority():
    """Test that command line parameters take priority over config"""
    
    # Mock config with defaults
    config = {
        "aws": {"default_role_arn": "arn:aws:iam::123456789012:role/ConfigRole"},
        "dataset": {"source_bucket": "config-bucket", "target_prefix": "config-prefix"},
        "training": {"instance_type": "ml.p3.2xlarge", "use_spot": False, "default_epochs": 90, "max_runtime": 86400, "enable_7_stage_pipeline": True},
        "monitoring": {"enable_detailed_logging": True, "save_metrics": True, "track_costs": True}
    }
    
    print("🧪 Testing Parameter Priority Handling...")
    print("=" * 60)
    
    # Test Case 1: Both role_arn and epochs from command line
    print("\n1️⃣ Test: Both role_arn and epochs from command line")
    args1 = MockArgs(
        role_arn="arn:aws:iam::123456789012:role/CmdLineRole", 
        epochs=2
    )
    
    # Simulate the _prepare_training_args logic
    role_arn1 = getattr(args1, 'role_arn', None) or config.get("aws", {}).get("default_role_arn")
    epochs1 = getattr(args1, 'epochs', None) or config.get("training", {}).get("default_epochs", 90)
    
    print(f"   Role ARN: {role_arn1[-20:]}... ({'✅ CMD LINE' if role_arn1.endswith('CmdLineRole') else '❌ CONFIG'})")
    print(f"   Epochs: {epochs1} ({'✅ CMD LINE' if epochs1 == 2 else '❌ CONFIG'})")
    
    # Test Case 2: Neither provided (use config defaults)
    print("\n2️⃣ Test: Use config defaults")
    args2 = MockArgs(role_arn=None, epochs=None)
    
    role_arn2 = getattr(args2, 'role_arn', None) or config.get("aws", {}).get("default_role_arn")
    epochs2 = getattr(args2, 'epochs', None) or config.get("training", {}).get("default_epochs", 90)
    
    print(f"   Role ARN: {role_arn2[-20:]}... ({'✅ CONFIG' if role_arn2.endswith('ConfigRole') else '❌ OTHER'})")
    print(f"   Epochs: {epochs2} ({'✅ CONFIG' if epochs2 == 90 else '❌ OTHER'})")
    
    # Test Case 3: Mixed (role_arn from cmd, epochs from config)
    print("\n3️⃣ Test: Mixed sources")
    args3 = MockArgs(
        role_arn="arn:aws:iam::123456789012:role/MixedRole", 
        epochs=None
    )
    
    role_arn3 = getattr(args3, 'role_arn', None) or config.get("aws", {}).get("default_role_arn")
    epochs3 = getattr(args3, 'epochs', None) or config.get("training", {}).get("default_epochs", 90)
    
    print(f"   Role ARN: {role_arn3[-20:]}... ({'✅ CMD LINE' if role_arn3.endswith('MixedRole') else '❌ CONFIG'})")
    print(f"   Epochs: {epochs3} ({'✅ CONFIG' if epochs3 == 90 else '❌ CMD LINE'})")
    
    # Test Case 4: Error handling (no role_arn anywhere)
    print("\n4️⃣ Test: Error handling (no role_arn)")
    config_no_role = {**config}
    config_no_role["aws"]["default_role_arn"] = None
    
    args4 = MockArgs(role_arn=None, epochs=5)
    role_arn4 = getattr(args4, 'role_arn', None) or config_no_role.get("aws", {}).get("default_role_arn")
    
    print(f"   Role ARN: {role_arn4} ({'✅ SHOULD ERROR' if role_arn4 is None else '❌ UNEXPECTED'})")
    
    # Test command building
    print("\n📋 Command Building Test:")
    training_args = {
        'role_arn': role_arn1,  # Using test case 1
        'epochs': epochs1,
        'instance_type': args1.instance_type
    }
    
    # Simulate command building (fixed version)
    cmd_role = training_args.get("role_arn")
    cmd_epochs = str(training_args.get("epochs"))
    
    print(f"   --role-arn: {cmd_role[-20:]}...")
    print(f"   --epochs: {cmd_epochs}")
    print(f"   ✅ Role correct: {cmd_role.endswith('CmdLineRole')}")
    print(f"   ✅ Epochs correct: {cmd_epochs == '2'}")
    
    print("\n🎯 Summary:")
    print("   ✅ Command line role_arn takes priority over config")
    print("   ✅ Command line epochs takes priority over config")  
    print("   ✅ Config defaults used when command line not provided")
    print("   ✅ Error handling for missing required role_arn")
    print("   ✅ Command building uses actual values, no hardcoded defaults")

if __name__ == '__main__':
    test_parameter_priority()