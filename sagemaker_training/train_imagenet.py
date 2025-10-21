#!/usr/bin/env python3
"""
Simple ImageNet-1K training script with ResNet50
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm

from imagenet_models import resnet50_imagenet
from imagenet_dataset import get_imagenet_dataloaders
from logger_setup import setup_logger, log_system_info, log_training_config, log_model_info, log_dataset_info


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    # Initialize logger
    logger = setup_logger("train_imagenet")
    
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet-1K')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to ImageNet dataset directory')
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of training epochs (default: 90)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    logger.info("ImageNet-1K Training with ResNet50")
    log_training_config(logger, args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log system information
    log_system_info(logger)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
        args.batch_size = min(args.batch_size, 32)  # Smaller batch for CPU
        logger.info(f"Reduced batch size to {args.batch_size} for CPU training")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading ImageNet dataset...")
    try:
        train_loader, val_loader = get_imagenet_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        log_dataset_info(logger, train_loader, val_loader)
    except FileNotFoundError as e:
        logger.error(f"Dataset loading failed: {e}")
        logger.error("Please ensure ImageNet dataset is properly organized:")
        logger.error("  data_dir/train/class_name/image.jpg")
        logger.error("  data_dir/val/class_name/image.jpg")
        return
    
    # Create model
    logger.info("Creating ResNet50 model...")
    model = resnet50_imagenet(num_classes=1000, pretrained=args.pretrained)
    model = model.to(device)
    
    # Log model information
    log_model_info(logger, model, "ResNet50")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            logger.info(f"New best model saved! Accuracy: {best_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    logger.info("Training completed!")
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()