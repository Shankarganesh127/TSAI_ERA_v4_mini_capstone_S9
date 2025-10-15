#!/usr/bin/env python3
"""
Quick test script to verify ImageNet training setup
"""

import torch
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_transforms


def test_model():
    """Test model creation and forward pass"""
    print("Testing ResNet50 model...")
    
    # Create model
    model = resnet50_imagenet(num_classes=1000, pretrained=False)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test with pretrained weights
    print("\nTesting with pretrained weights...")
    try:
        model_pretrained = resnet50_imagenet(num_classes=1000, pretrained=True)
        print("✓ Pretrained model loaded successfully")
    except Exception as e:
        print(f"⚠ Pretrained model loading failed: {e}")
        print("  This is normal if torchvision is not installed")


def test_transforms():
    """Test data transforms"""
    print("\nTesting data transforms...")
    
    try:
        train_transform, val_transform = get_imagenet_transforms()
        print("✓ Transforms created successfully")
        
        # Create dummy image
        dummy_image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
        from PIL import Image
        pil_image = Image.fromarray(dummy_image.numpy())
        
        # Apply transforms
        train_tensor = train_transform(pil_image)
        val_tensor = val_transform(pil_image)
        
        print(f"✓ Transform application successful")
        print(f"  Train tensor shape: {train_tensor.shape}")
        print(f"  Val tensor shape: {val_tensor.shape}")
        print(f"  Train tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
        
    except ImportError as e:
        print(f"⚠ Transform test skipped: {e}")
        print("  Install PIL: pip install Pillow")


def test_device():
    """Test device availability"""
    print("\nTesting device availability...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU memory
        try:
            model = resnet50_imagenet().to(device)
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            print("✓ GPU inference test successful")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
            
    else:
        print("⚠ CUDA not available - training will be slow on CPU")
    
    print(f"✓ PyTorch version: {torch.__version__}")


def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'), 
        ('tqdm', 'Progress bars'),
        ('PIL', 'Image processing')
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {description} ({package})")
        except ImportError:
            print(f"✗ {description} ({package}) - Install with: pip install {package}")


def main():
    parser = argparse.ArgumentParser(description='Test ImageNet training setup')
    parser.add_argument('--skip-model', action='store_true', help='Skip model testing')
    parser.add_argument('--skip-transforms', action='store_true', help='Skip transform testing')
    
    args = parser.parse_args()
    
    print("ImageNet Training Setup Test")
    print("=" * 40)
    
    # Test dependencies first
    test_dependencies()
    
    # Test device
    test_device()
    
    # Test model
    if not args.skip_model:
        test_model()
    
    # Test transforms
    if not args.skip_transforms:
        test_transforms()
    
    print("\n" + "=" * 40)
    print("Test completed! If all tests passed, you're ready to train.")
    print("\nNext steps:")
    print("1. Prepare ImageNet dataset")
    print("2. Run: python train_imagenet.py --data-dir /path/to/imagenet")


if __name__ == '__main__':
    main()