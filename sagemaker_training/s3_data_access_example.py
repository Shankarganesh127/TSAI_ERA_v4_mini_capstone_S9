#!/usr/bin/env python3
"""
Example: Verify S3 Data Access in SageMaker Training

This script demonstrates how to access and validate your S3 ImageNet dataset
within a SageMaker training container.
"""

import os
import sys
from pathlib import Path

def verify_s3_data_access():
    """Verify S3 data is properly mounted in SageMaker container"""
    
    print("🔍 Verifying S3 data access in SageMaker training container...")
    print("=" * 60)
    
    # Standard SageMaker data paths
    data_dir = os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    print(f"📂 Data directory: {data_dir}")
    print(f"🏋️  Training directory: {train_dir}")
    print(f"✅ Validation directory: {val_dir}")
    print()
    
    # Check if directories exist
    directories_status = {
        'Data root': data_dir,
        'Training data': train_dir,
        'Validation data': val_dir
    }
    
    for name, path in directories_status.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path} (EXISTS)")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
    
    print()
    
    # Check training data structure
    if os.path.exists(train_dir):
        print("🔍 Checking training data structure...")
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        print(f"📊 Found {len(train_classes)} training classes")
        
        if train_classes:
            print(f"📝 Sample classes: {train_classes[:5]}...")
            
            # Check sample class folder
            sample_class = train_classes[0]
            sample_class_path = os.path.join(train_dir, sample_class)
            images = [f for f in os.listdir(sample_class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"🖼️  Images in {sample_class}: {len(images)}")
        print()
    
    # Check validation data structure
    if os.path.exists(val_dir):
        print("🔍 Checking validation data structure...")
        val_classes = [d for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
        print(f"📊 Found {len(val_classes)} validation classes")
        
        if val_classes:
            print(f"📝 Sample classes: {val_classes[:5]}...")
            
            # Check sample class folder
            sample_class = val_classes[0]
            sample_class_path = os.path.join(val_dir, sample_class)
            images = [f for f in os.listdir(sample_class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"🖼️  Images in {sample_class}: {len(images)}")
        print()
    
    # Check metadata
    metadata_path = os.path.join(data_dir, 'metadata')
    if os.path.exists(metadata_path):
        print("🔍 Checking metadata...")
        metadata_files = os.listdir(metadata_path)
        print(f"📋 Metadata files: {metadata_files}")
        print()
    
    # Environment variables check
    print("🔍 SageMaker environment variables:")
    sagemaker_vars = {k: v for k, v in os.environ.items() if 'SM_' in k}
    for key, value in sorted(sagemaker_vars.items()):
        print(f"   {key}: {value}")
    
    print()
    print("✅ S3 data access verification completed!")

def get_s3_data_paths():
    """Get standard S3 data paths for your training code"""
    
    # SageMaker automatically sets this environment variable
    data_dir = os.environ.get('SM_CHANNEL_IMAGENET', '/opt/ml/input/data/imagenet')
    
    paths = {
        'data_root': data_dir,
        'train_dir': os.path.join(data_dir, 'train'),
        'val_dir': os.path.join(data_dir, 'val'),
        'metadata_dir': os.path.join(data_dir, 'metadata'),
        'manifest_file': os.path.join(data_dir, 'manifest.json')
    }
    
    return paths

def example_data_loading():
    """Example of how to load data in your training code"""
    
    print("📚 Example: How to load S3 data in your training code")
    print("=" * 50)
    
    # Get data paths
    paths = get_s3_data_paths()
    
    print("1. Import required libraries:")
    print("""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
""")
    
    print("2. Define data paths:")
    print(f"""
# SageMaker automatically mounts your S3 data here:
data_root = '{paths['data_root']}'
train_dir = '{paths['train_dir']}'
val_dir = '{paths['val_dir']}'
""")
    
    print("3. Create data transforms:")
    print("""
# Standard ImageNet transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
""")
    
    print("4. Create datasets:")
    print("""
# Create datasets using torchvision.datasets.ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
""")
    
    print("5. Create data loaders:")
    print("""
# Create data loaders for training
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
""")
    
    print("6. Use in training loop:")
    print("""
# Training loop example
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Your training logic here
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
""")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_s3_data_access()
    elif len(sys.argv) > 1 and sys.argv[1] == 'example':
        example_data_loading()
    else:
        print("Usage:")
        print("  python s3_data_access_example.py verify   # Verify data access")
        print("  python s3_data_access_example.py example  # Show code examples")
        print()
        verify_s3_data_access()
        print()
        example_data_loading()