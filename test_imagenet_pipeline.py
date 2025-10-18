#!/usr/bin/env python3
"""
ImageNet Pipeline Test Script
Quick validation that all components work correctly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from imagenet_training_pipeline import LRFinder, BatchSizeFinder, HyperparameterOptimizer, FullTrainer

def create_dummy_data(batch_size=32, num_batches=10):
    """Create dummy data for testing"""
    print("📦 Creating dummy ImageNet data...")
    
    class DummyDataset:
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Random ImageNet-like data: 224x224 RGB images, 1000 classes
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 1000, (1,)).item()
            return image, label
    
    dataset = DummyDataset(num_samples=batch_size * num_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, dataloader  # Use same for train/val

def create_dummy_model():
    """Create a small ResNet-like model for testing"""
    print("🏗️  Creating dummy model...")
    
    class DummyResNet(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return DummyResNet()

def test_lr_finder():
    """Test LR Range Test functionality"""
    print("\n" + "="*50)
    print("🔍 Testing LR Finder")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_dummy_model().to(device)
    train_loader, _ = create_dummy_data(batch_size=16, num_batches=5)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    try:
        lrs, losses = lr_finder.range_test(train_loader, num_iter=20)
        lr_config = lr_finder.suggest_lr()
        
        print(f"✅ LR Finder completed successfully")
        print(f"   Found {len(lrs)} LR points")
        print(f"   Suggested max LR: {lr_config['max_lr']:.2e}")
        print(f"   Suggested min LR: {lr_config['min_lr']:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ LR Finder failed: {e}")
        return False

def test_batch_size_finder():
    """Test Batch Size Finder functionality"""
    print("\n" + "="*50)
    print("📦 Testing Batch Size Finder")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("⏭️  Skipping batch size test (no CUDA)")
        return True
    
    device = torch.device('cuda')
    model = create_dummy_model().to(device)
    
    try:
        max_batch_size = BatchSizeFinder.find_max_batch_size(
            model, (3, 224, 224), device, max_batch_size=128)
        
        print(f"✅ Batch Size Finder completed successfully")
        print(f"   Maximum batch size: {max_batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch Size Finder failed: {e}")
        return False

def test_hyperparameter_optimizer():
    """Test Hyperparameter Optimizer functionality"""
    print("\n" + "="*50)
    print("⚖️  Testing Hyperparameter Optimizer")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_dummy_data(batch_size=16, num_batches=3)
    
    def model_fn():
        return create_dummy_model()
    
    optimizer = HyperparameterOptimizer(model_fn, train_loader, val_loader, device)
    lr_config = {'min_lr': 1e-3, 'max_lr': 0.1}
    
    try:
        results, best_wd = optimizer.weight_decay_search(
            lr_config, batch_size=16, wd_values=[1e-4, 5e-4], epochs=1)
        
        print(f"✅ Hyperparameter Optimizer completed successfully")
        print(f"   Tested {len(results)} weight decay values")
        print(f"   Best weight decay: {best_wd:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hyperparameter Optimizer failed: {e}")
        return False

def test_full_trainer():
    """Test Full Trainer functionality"""
    print("\n" + "="*50)
    print("🚀 Testing Full Trainer")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_dummy_model().to(device)
    train_loader, val_loader = create_dummy_data(batch_size=16, num_batches=3)
    
    trainer = FullTrainer(model, train_loader, val_loader, device, './test_output')
    lr_config = {'min_lr': 1e-3, 'max_lr': 0.1}
    
    try:
        history = trainer.train(
            lr_config=lr_config,
            epochs=2,  # Very short test
            batch_size=16,
            weight_decay=1e-4,
            save_checkpoints=False,
            early_stopping_patience=1
        )
        
        print(f"✅ Full Trainer completed successfully")
        print(f"   Trained for {len(history['train_loss'])} epochs")
        print(f"   Final train acc: {history['train_acc'][-1]:.2f}%")
        print(f"   Final val acc: {history['val_acc'][-1]:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Full Trainer failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ImageNet Training Pipeline Test Suite")
    print("="*60)
    
    # System info
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"🐍 PyTorch: {torch.__version__}")
    
    # Run tests
    tests = [
        ("LR Finder", test_lr_finder),
        ("Batch Size Finder", test_batch_size_finder),
        ("Hyperparameter Optimizer", test_hyperparameter_optimizer),
        ("Full Trainer", test_full_trainer)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("🏁 Test Results Summary")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\n🚀 Quick start commands:")
        print("  ./run_imagenet_pipeline.sh --data /path/to/imagenet --mode test")
        print("  uv run python imagenet_training_pipeline.py --data /path/to/imagenet --quick-mode")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)