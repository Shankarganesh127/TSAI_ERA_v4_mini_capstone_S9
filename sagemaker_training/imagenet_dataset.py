#!/usr/bin/env python3
"""
High-performance ImageNet-1K dataset loader using WebDataset (.tar I/O)
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from logger_setup import get_unified_logger

import webdataset as wds 
import io
import math


# Logger setup
logger = get_unified_logger("imagenet_dataset")

# Standard ImageNet Constants
IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000


def get_imagenet_transforms(input_size=224, lightweight=False):
    """
    Get ImageNet data transforms with optional lightweight mode
    
    Args:
        input_size: Input image size (default: 224 for ResNet)
        lightweight: If True, use faster but less aggressive augmentations
    
    Returns:
        train_transform, val_transform
    """
    
    if lightweight:
        # Lightweight version for maximum speed
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Full advanced augmentations
        train_transform = transforms.Compose([
            # Scale-aware random cropping (8%-100% of image, aspect ratio 3:4 to 4:3)
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            
            # Horizontal flip with 50% probability
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Advanced color augmentations for lighting/illumination robustness
            transforms.ColorJitter(
                brightness=0.4,  # ±40% brightness change
                contrast=0.4,    # ±40% contrast change  
                saturation=0.4,  # ±40% saturation change
                hue=0.1          # ±10% hue change
            ),
            
            # Geometric augmentations for spatial robustness
            transforms.RandomAffine(
                degrees=0,       # No rotation to preserve object orientation
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1),      # ±10% scaling
                shear=0.1,             # ±10% shearing
                fill=0
            ),
            
            # Gaussian blur for noise and focus robustness
            transforms.GaussianBlur(
                kernel_size=(3, 3), 
                sigma=(0.1, 2.0)     # Blur strength range
            ),
            
            transforms.ToTensor(),
            
            # Random Erasing (Cutout) for occlusion robustness - applied after ToTensor on [0,1] range
            transforms.RandomErasing(
                p=0.25,           # 25% probability
                scale=(0.02, 0.33),  # Erase 2-33% of image area
                ratio=(0.3, 3.3),    # Aspect ratio range
                value='random'       # Fill with random pixel values in [0,1] range
            ),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
            
            # Note: RandomErasing after normalization removed - causes issues with normalized tensors
            # transforms.RandomErasing(
            #     p=0.25,           # 25% probability
            #     scale=(0.02, 0.2),   # Smaller erasures after normalization
            #     ratio=(0.3, 3.3),
            #     value=0             # Erase to zero (black) after normalization
            # )
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


def get_test_time_augmentation_transforms(input_size=224, num_augmentations=10):
    """
    Get test-time augmentation transforms for improved validation accuracy
    
    NOTE: Currently unused in training pipeline. Available for future enhancement
    to boost validation accuracy by 1-2% at the cost of ~5x slower validation.
    
    Args:
        input_size: Input image size
        num_augmentations: Number of augmentations per image
    
    Returns:
        List of transforms for test-time augmentation
    """
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create multiple augmentation strategies
    augmentation_transforms = []
    
    for _ in range(num_augmentations):
        # Random crop positions and sizes
        crop_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        augmentation_transforms.append(crop_transforms)
    
    return augmentation_transforms


def get_imagenet_dataloaders(train, val, batch_size=32, num_workers=4, pin_memory=True, lightweight_augs=False):
    """
    Create ImageNet-1K data loaders
    
    Args:
        train: Path to training dataset directory
        val: Path to validation dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        lightweight_augs: Use lightweight augmentations for maximum speed
    
    Returns:
        train_loader, val_loader
    """
    
    logger = get_unified_logger("imagenet_dataset")
    logger.info(f"get_imagenet_dataloaders called with:")
    logger.info(f"  train={train}")
    logger.info(f"  val={val}")
    logger.info(f"  batch_size={batch_size}")
    logger.info(f"  num_workers={num_workers}")
    logger.info(f"  pin_memory={pin_memory}")
    logger.info(f"  lightweight_augs={lightweight_augs}")

    train_transform, val_transform = get_imagenet_transforms(lightweight=lightweight_augs)
    if lightweight_augs:
        logger.info("Using lightweight augmentations for maximum training speed")
    else:
        logger.info("Using advanced augmentations for better accuracy")
    logger.info("Transforms created")

    # Use the provided paths directly
    train_dir = train
    val_dir = val
    
    # 2. Define URL Glob Patterns (WebDataset input)
    # SageMaker mounts S3 data at data_dir. Assuming tar files are in a dedicated folder.
    train_urls = os.path.join(train, '*.tar')
    val_urls = os.path.join(val, '*.tar')
    
    #train_urls = wds.shardlists.split_by_node(train_urls)
    #val_urls = wds.shardlists.split_by_node(val_urls)

    logger.info(f"Checking paths - train_dir={train_dir}, val_dir={val_dir}")
    
    logger.info(f"🚀 Using WebDataset for training from: {train_urls}")
    logger.info(f"🚀 Using WebDataset for validation from: {val_urls}")
    
    # Calculate effective epoch size for the scheduler. This is critical.
    train_batches_per_epoch = math.ceil(IMAGENET_TRAIN_SIZE / batch_size)

    if not os.path.exists(train_dir):
        logger.error(f"Training directory does not exist: {train_dir}")
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        logger.error(f"Validation directory does not exist: {val_dir}")
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")

    logger.info("Both directories exist, creating datasets")
    
    # --- Training DataLoader (WebDataset) ---

    train_dataset = (
        wds.WebDataset(train_urls)
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.ignore_and_continue)
        .rename(image="jpg", label="cls")
        .map_dict(image=train_transform, handler=wds.handlers.ignore_and_continue)
        .to_tuple("image", "label")
        .with_epoch(train_batches_per_epoch)
        .batched(batch_size, partial=False)
    )

    # WebLoader is necessary to wrap the WebDataset object and correctly handle multi-threading/DDP
    train_loader = wds.WebLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # --- Validation DataLoader (WebDataset) ---

    val_batches_per_epoch = math.ceil(IMAGENET_VAL_SIZE / batch_size)

    # Note: Validation doesn't need shuffling, but still needs DDP partitioning
    val_dataset = (
        wds.WebDataset(val_urls)
        .decode("pil", handler=wds.handlers.ignore_and_continue)
        .rename(image="jpg", label="cls")
        .map_dict(image=val_transform, handler=wds.handlers.ignore_and_continue)
        .to_tuple("image", "label")
        .with_epoch(val_batches_per_epoch)
        .batched(batch_size, partial=True)
    )

    val_loader = wds.WebLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader

''''
    # Create datasets
    try:
        logger.info(f"Creating training dataset from {train_dir}")
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        logger.info(f"Training dataset created successfully with {len(train_dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise

    try:
        logger.info(f"Creating validation dataset from {val_dir}")
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
        logger.info(f"Validation dataset created successfully with {len(val_dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating validation dataset: {e}")
        raise

    # Log dataset information if called directly (not from other modules)
    try:
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Number of training classes: {len(train_dataset.classes)}")
        logger.info(f"Number of validation classes: {len(val_dataset.classes)}")
    except Exception:
        # Fallback for when logger is not available
        pass

    # Create data loaders
    logger.info("Creating data loaders")
    try:
        logger.info(f"Creating training data loader with batch_size={batch_size}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info("Training data loader created successfully")
    except Exception as e:
        logger.error(f"Error creating training data loader: {e}")
        raise

    try:
        logger.info(f"Creating validation data loader with batch_size={batch_size}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info("Validation data loader created successfully")
    except Exception as e:
        logger.error(f"Error creating validation data loader: {e}")
        raise

    logger.info("Both data loaders created, returning")
    return train_loader, val_loader
'''

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
    logger = get_unified_logger("imagenet_dataset")
    
    train_transform, val_transform = get_imagenet_transforms()
    logger.info("ImageNet transforms created successfully")
    logger.info(f"Train transform: {train_transform}")
    logger.info(f"Val transform: {val_transform}")