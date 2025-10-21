#!/usr/bin/env python3
"""
Minimal ImageNet-1K dataset loader
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from logger_setup import get_logger


def get_imagenet_transforms(input_size=224):
    """
    Get standard ImageNet data transforms
    
    Args:
        input_size: Input image size (default: 224)
    """
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_imagenet_dataloaders(data_dir, batch_size=32, num_workers=4, pin_memory=True):
    """
    Create ImageNet-1K data loaders
    
    Args:
        data_dir: Path to ImageNet dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    
    train_transform, val_transform = get_imagenet_transforms()
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ImageNet data directory not found: {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")
    
    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Log dataset information if called directly (not from other modules)
    try:
        logger = get_logger("imagenet_dataset")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Number of classes: {len(train_dataset.classes)}")
    except Exception:
        # Fallback for when logger is not available
        pass
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def get_tiny_imagenet_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Alternative: Create Tiny ImageNet data loaders (200 classes, 64x64 images)
    This is much smaller and more manageable for testing
    """
    
    # Transforms for Tiny ImageNet (64x64 images)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test transforms
    logger = get_logger("imagenet_dataset")
    
    train_transform, val_transform = get_imagenet_transforms()
    logger.info("ImageNet transforms created successfully")
    logger.info(f"Train transform: {train_transform}")
    logger.info(f"Val transform: {val_transform}")