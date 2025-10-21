"""
Simple utilities for ImageNet training
"""

import torch
import os
import json
from datetime import datetime
from logger_setup import get_logger


def save_training_config(args, save_dir):
    """Save training configuration to JSON file"""
    config = {
        'model': 'ResNet50',
        'dataset': 'ImageNet-1K',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'pretrained': args.pretrained,
        'num_workers': args.num_workers,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_acc']


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """Calculate top-k accuracy"""
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr_schedule_func(schedule_type='step'):
    """Get learning rate schedule function"""
    if schedule_type == 'step':
        def lr_schedule(epoch):
            if epoch < 30:
                return 1.0
            elif epoch < 60:
                return 0.1
            else:
                return 0.01
        return lr_schedule
    elif schedule_type == 'cosine':
        def lr_schedule(epoch, total_epochs=90):
            import math
            return 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        return lr_schedule
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary using logger"""
    logger = get_logger("utils")
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        logger.warning("torchsummary not available. Install with: pip install torchsummary")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Test utilities
    logger = get_logger("utils_test")
    logger.info("ImageNet training utilities loaded successfully")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(5):
        meter.update(i * 0.1)
    logger.info(f"Average meter test: {meter.avg:.3f}")
    
    # Test LR schedule
    lr_func = get_lr_schedule_func('step')
    for epoch in [0, 29, 30, 59, 60, 89]:
        logger.info(f"Epoch {epoch}: LR multiplier = {lr_func(epoch):.3f}")