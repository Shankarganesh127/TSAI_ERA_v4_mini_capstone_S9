#!/usr/bin/env python3
"""
ILSVRC Dataset Loader
Handles the ImageNet Object Localization Challenge dataset format where
validation images are in a flat directory with labels in separate files.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
from logger_setup import get_unified_logger


class ILSVRCValidationDataset(Dataset):
    """
    Custom dataset for ILSVRC validation data where images are in a flat directory
    and labels are provided in a separate file.
    """
    
    def __init__(self, val_dir, val_labels_file, synset_mapping_file=None, val_solution_file=None, transform=None):
        """
        Args:
            val_dir: Directory containing validation images
            val_labels_file: Path to val.txt file with image_name class_index pairs (optional for compatibility)
            synset_mapping_file: Path to LOC_synset_mapping.txt
            val_solution_file: Path to LOC_val_solution.csv with actual class labels
            transform: Optional transform to be applied on images
        """
        self.val_dir = val_dir
        self.transform = transform
        self.logger = get_unified_logger("ilsvrc_dataset")
        
        # Load synset to index mapping first
        synset_to_idx = {}
        if synset_mapping_file and os.path.exists(synset_mapping_file):
            with open(synset_mapping_file, 'r') as f:
                for idx, line in enumerate(f):
                    synset = line.strip().split()[0]
                    synset_to_idx[synset] = idx
        
        # Read validation labels from solution file if available
        self.samples = []
        if val_solution_file and os.path.exists(val_solution_file):
            # Use LOC_val_solution.csv for proper class labels
            with open(val_solution_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        prediction_string = parts[1]
                        # Extract synset from prediction string (first element)
                        synset = prediction_string.split()[0]
                        
                        if synset in synset_to_idx:
                            class_idx = synset_to_idx[synset]
                            image_path = os.path.join(val_dir, f"{image_name}.JPEG")
                            if os.path.exists(image_path):
                                self.samples.append((image_path, class_idx))
        
        # Fallback to basic val.txt if solution file not available
        elif val_labels_file and os.path.exists(val_labels_file):
            self.logger.warning("Using val.txt labels - these may be sequential indices, not class labels")
            with open(val_labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        class_idx = int(parts[1]) - 1  # Convert to 0-based indexing
                        # Clamp to valid range for safety
                        class_idx = max(0, min(class_idx, 999))
                        image_path = os.path.join(val_dir, f"{image_name}.JPEG")
                        if os.path.exists(image_path):
                            self.samples.append((image_path, class_idx))
        
        # Check label range for debugging
        if self.samples:
            labels = [label for _, label in self.samples]
            min_label, max_label = min(labels), max(labels)
            self.logger.info(f"Loaded {len(self.samples)} validation samples")
            self.logger.info(f"Label range: {min_label} to {max_label} (should be 0-999 for ImageNet)")
            # Check for any invalid labels
            invalid_labels = [label for label in labels if label < 0 or label >= 1000]
            if invalid_labels:
                self.logger.warning(f"Found {len(invalid_labels)} invalid labels: {set(invalid_labels)}")
        else:
            self.logger.warning("No validation samples loaded!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_ilsvrc_dataloaders(data_root, batch_size=32, num_workers=4, pin_memory=True):
    """
    Create ILSVRC data loaders that handle the specific ILSVRC format.
    
    Args:
        data_root: Path to ILSVRC root directory OR CLS-LOC directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    
    # Determine if data_root is ILSVRC root or CLS-LOC directory
    if data_root.endswith("Data/CLS-LOC") or data_root.endswith("Data\\CLS-LOC"):
        # data_root points to CLS-LOC directory
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        # Go up two levels to find ILSVRC root for ImageSets
        ilsvrc_root = os.path.dirname(os.path.dirname(data_root))
        val_labels_file = os.path.join(ilsvrc_root, "ImageSets", "CLS-LOC", "val.txt")
        # Look for solution files in Downloads (common location)
        downloads_dir = "/home/ubuntu/Downloads"
        synset_mapping_file = os.path.join(downloads_dir, "LOC_synset_mapping.txt")
        val_solution_file = os.path.join(downloads_dir, "LOC_val_solution.csv")
    else:
        # Assume data_root is ILSVRC root directory
        train_dir = os.path.join(data_root, "Data", "CLS-LOC", "train")
        val_dir = os.path.join(data_root, "Data", "CLS-LOC", "val")
        val_labels_file = os.path.join(data_root, "ImageSets", "CLS-LOC", "val.txt")
        # Look for solution files in Downloads (common location)
        downloads_dir = "/home/ubuntu/Downloads"
        synset_mapping_file = os.path.join(downloads_dir, "LOC_synset_mapping.txt")
        val_solution_file = os.path.join(downloads_dir, "LOC_val_solution.csv")
    
    logger = get_unified_logger("ilsvrc_dataset")
    logger.info(f"Train directory: {train_dir}")
    logger.info(f"Val directory: {val_dir}")
    logger.info(f"Val labels file: {val_labels_file}")
    logger.info(f"Synset mapping file: {synset_mapping_file}")
    logger.info(f"Val solution file: {val_solution_file}")
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        logger.error(f"Validation directory not found: {val_dir}")
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not os.path.exists(val_labels_file):
        logger.error(f"Validation labels file not found: {val_labels_file}")
        raise FileNotFoundError(f"Validation labels file not found: {val_labels_file}")
    
    # Check for solution files
    if not os.path.exists(synset_mapping_file):
        logger.warning(f"Synset mapping file not found: {synset_mapping_file}")
        synset_mapping_file = None
    if not os.path.exists(val_solution_file):
        logger.warning(f"Validation solution file not found: {val_solution_file}")
        val_solution_file = None
    
    # Define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create datasets
    # Training: Use standard ImageFolder (works because train data is in class folders)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Validation: Use custom dataset (handles flat directory structure)
    val_dataset = ILSVRCValidationDataset(
        val_dir=val_dir, 
        val_labels_file=val_labels_file,
        synset_mapping_file=synset_mapping_file,
        val_solution_file=val_solution_file,
        transform=val_transform
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
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


def get_ilsvrc_transforms(input_size=224):
    """
    Get ILSVRC-specific data transforms
    
    Args:
        input_size: Input image size (default: 224)
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


if __name__ == "__main__":
    # Test the dataset loader
    logger = get_unified_logger("ilsvrc_dataset")
    data_root = "/home/ubuntu/Downloads/ILSVRC"
    try:
        train_loader, val_loader = get_ilsvrc_dataloaders(data_root, batch_size=8)
        logger.info("ILSVRC Dataset Loader Test")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        # Test loading a batch
        for batch_idx, (images, labels) in enumerate(val_loader):
            logger.info(f"Batch {batch_idx}: Images {images.shape}, Labels {labels.shape}")
            if batch_idx >= 2:  # Just test a few batches
                break
        logger.info("Dataset loading successful!")
    except Exception as e:
        logger.error(f"Error: {e}")