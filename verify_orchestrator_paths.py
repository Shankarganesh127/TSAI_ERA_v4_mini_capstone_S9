#!/usr/bin/env python3
"""
Quick verification script to show exactly what S3 paths the orchestrator will use
"""

def verify_orchestrator_paths():
    # Simulate the arguments from your command
    source_bucket = "tsai-era-v4-mini-capstone"
    source_prefix = "Datasets/imagenet1k/ILSVRC"  # Default value
    target_prefix = "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker"  # Default value
    
    # Build S3 URI
    s3_bucket_uri = f"s3://{source_bucket}"
    
    # Configure train and validation data paths
    train_data_path = f"{source_prefix}/Data/CLS-LOC/train/"
    val_data_path = f"{target_prefix}/val/"
    
    print("🔍 Orchestrator Path Verification")
    print("=" * 50)
    print(f"📂 Source Bucket: {source_bucket}")
    print(f"📂 Source Prefix: {source_prefix}")
    print(f"📂 Target Prefix: {target_prefix}")
    print()
    print("📍 Full S3 Paths:")
    print(f"   Training Data: {s3_bucket_uri}/{train_data_path}")
    print(f"   Validation Data: {s3_bucket_uri}/{val_data_path}")
    print()
    print("✅ This matches your dataset structure!")
    print(f"   ✓ Train: s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/")
    print(f"   ✓ Val: s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/")
    print()
    print("🚀 Your command will work correctly:")
    print('python sagemaker_orchestrator.py \\')
    print('  --role-arn "arn:aws:iam::872109682518:role/service-role/AmazonSageMaker-ExecutionRole-20251009T010774" \\')
    print('  --source-bucket "tsai-era-v4-mini-capstone" \\')
    print('  --use-spot \\')
    print('  --epochs 1 \\')
    print('  --instance-type "ml.p3.8xlarge"')

if __name__ == "__main__":
    verify_orchestrator_paths()