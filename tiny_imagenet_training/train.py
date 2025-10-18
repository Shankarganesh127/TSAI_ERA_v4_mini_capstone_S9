#!/usr/bin/env python3
"""
Simplified Tiny-ImageNet Training
Single file containing all training logic, configuration, and CLI.
"""
from __future__ import annotations
import os, json, uuid, datetime as dt, argparse, logging, importlib.util, sys, warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from tqdm import tqdm

# Suppress the specific scheduler warning that occurs with GradScaler + OneCycleLR
# This is a known PyTorch issue that doesn't affect training correctness
warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step.*before.*optimizer.step", category=UserWarning)

# Suppress CUDA precision warnings that don't affect training
warnings.filterwarnings("ignore", message=".*setFloat32Precision.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*you have set wrong precision.*", category=UserWarning)


# === CONFIGURATION ===

DEFAULT_RESULTS_SUMMARY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lr_optimization', 'results', 'onecycle_summary.json'))

@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    run_id: str
    num_classes: int = 200
    input_size: int = 64
    epochs: int = 20
    batch_size: int = 64
    grad_accum_steps: int = 2
    amp: bool = True
    lr_min: float = 0.001
    lr_max: float = 0.1
    weight_decay: float = 1e-4
    pct_start: float = 0.3
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    final_div_factor: float = 1e3
    num_workers: int = 4
    seed: int = 42
    log_every: int = 1000

    @classmethod
    def from_auto(cls, data_dir: str, output_root: str, **overrides):
        run_id = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        cfg = cls(data_dir=data_dir, output_dir=output_dir, run_id=run_id, **overrides)
        return cfg
    
    def save(self) -> str:
        path = os.path.join(self.output_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return path


# === DATASET ===

def load_torchvision_datasets():
    try:
        from torchvision import datasets, transforms
        return datasets, transforms
    except ImportError:
        raise ImportError("torchvision not available")

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load class names
        wnids_file = os.path.join(root, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load samples
        self.samples = []
        if split == 'train':
            train_dir = os.path.join(root, 'train')
            for class_name in self.class_names:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.JPEG', '.jpg', '.png')):
                            self.samples.append((
                                os.path.join(class_dir, img_name),
                                self.class_to_idx[class_name]
                            ))
        elif split == 'val':
            val_dir = os.path.join(root, 'val')
            # Load validation annotations
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    self.samples.append((
                        os.path.join(val_dir, 'images', img_name),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        from PIL import Image
        sample = Image.open(path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def create_data_loaders(cfg: TrainConfig):
    datasets, transforms = load_torchvision_datasets()
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TinyImageNetDataset(cfg.data_dir, 'train', train_transform)
    val_dataset = TinyImageNetDataset(cfg.data_dir, 'val', val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader


# === MODEL ===

def get_resnet50_tiny():
    """ResNet50 adapted for Tiny-ImageNet (64x64 input)"""
    try:
        from torchvision import models
    except ImportError:
        raise ImportError("torchvision not available")
    
    model = models.resnet50(weights=None)
    # Modify for 64x64 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Modify final layer for 200 classes
    model.fc = nn.Linear(model.fc.in_features, 200)
    return model


# === LOSS ===

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        pred_log_softmax = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred_log_softmax)
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * pred_log_softmax, dim=1))


# === LOGGING ===

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_system_info(logger):
    import platform
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

def log_model_info(logger, model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    logger.info(f"{model_name} Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {non_trainable_params:,}")
    logger.info(f"  Model size: {model_size_mb:.2f} MB")

def log_dataset_info(logger, train_loader, val_loader):
    logger.info("Dataset Information:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    logger.info(f"  Batch size: {train_loader.batch_size}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    
    # Log sample shapes
    for x, y in train_loader:
        logger.info(f"  Input shape: {x.shape}")
        logger.info(f"  Target shape: {y.shape}")
        break


# === TRAINING ===

def setup(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_resnet50_tiny()
    
    # Device setup with optimizations
    if torch.cuda.is_available():
        model = model.to(device, memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for performance (if available)
        try:
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
        except Exception:
            pass
    else:
        model = model.to(device)

    criterion = LabelSmoothingCE(0.1)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_min, momentum=cfg.base_momentum, weight_decay=cfg.weight_decay, nesterov=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(cfg)
    
    # Create scheduler
    total_steps = len(train_loader) * cfg.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.lr_max, total_steps=total_steps,
        pct_start=cfg.pct_start, base_momentum=cfg.base_momentum, max_momentum=cfg.max_momentum,
        final_div_factor=cfg.final_div_factor
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if cfg.amp and torch.cuda.is_available() else None
    
    return device, model, criterion, optimizer, scheduler, scaler, train_loader, val_loader

def calculate_accuracy(output, target):
    """Calculate top-1 accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.mul_(100.0 / batch_size).item()
        return accuracy, batch_size

def train_epoch(cfg: TrainConfig, device, model, criterion, optimizer, scheduler, scaler, train_loader, epoch, logger=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{cfg.epochs}', 
                ncols=120, leave=True, dynamic_ncols=False, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        if torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=cfg.amp and torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y) / cfg.grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step optimizer and scheduler together
            # Note: The scheduler warning with GradScaler+OneCycleLR is a known PyTorch issue
            # that doesn't affect training correctness
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate accuracy like train_imagenet.py
        batch_correct, batch_total = calculate_accuracy(out, y)
        correct += batch_correct
        total += batch_total
        running_loss += loss.item() * cfg.grad_accum_steps
        
        # Update progress bar every 10 batches
        if batch_idx % 10 == 0:
            current_acc = 100. * correct / total if total > 0 else 0.0
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Log to file only (not console) at intervals to avoid interfering with tqdm
        if (batch_idx + 1) % max(1, getattr(cfg, 'log_every', 500)) == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            lr_pct = (cur_lr / max(1e-12, cfg.lr_max)) * 100.0
            batch_acc = 100. * correct / total if total > 0 else 0.0
            current_loss = running_loss / (batch_idx + 1)
            # Log to file only to avoid console interference
            if logger:
                logger.debug(
                    f"[train] step {batch_idx+1}/{len(train_loader)}: "
                    f"loss {current_loss:.6f}, "
                    f"acc {batch_acc:.2f}%, "
                    f"lr {cur_lr:.6f} ({lr_pct:.2f}%)"
                )
    
    final_loss = running_loss / len(train_loader)
    final_acc = 100. * correct / total if total > 0 else 0.0
    
    return {
        'loss': final_loss,
        'top1': final_acc,
        'top5': 0.0  # Not implemented for simplicity
    }

def evaluate(cfg: TrainConfig, device, model, criterion, val_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc='Validating', 
                ncols=120, leave=True, dynamic_ncols=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            if torch.cuda.is_available():
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=cfg.amp and torch.cuda.is_available()):
                out = model(x)
                loss = criterion(out, y)
            
            batch_correct, batch_total = calculate_accuracy(out, y)
            correct += batch_correct
            total += batch_total
            running_loss += loss.item()
    
    final_loss = running_loss / len(val_loader)
    final_acc = 100. * correct / total if total > 0 else 0.0
    
    return {
        'loss': final_loss,
        'top1': final_acc,
        'top5': 0.0  # Not implemented for simplicity
    }

def run_training(cfg: TrainConfig):
    # Setup logging
    log_file = os.path.join(cfg.output_dir, 'train.log')
    logger = setup_logger('tiny_imagenet_training', log_file)
    logger.info(f"Logging to {log_file}")
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    log_system_info(logger)
    
    # Log config
    logger.info("Training config:")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    # Validate dataset
    if not os.path.exists(cfg.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {cfg.data_dir}")
    
    required_files = ['wnids.txt', 'train', 'val']
    for f in required_files:
        path = os.path.join(cfg.data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file/directory not found: {path}")
    
    # Count samples for verification
    train_dir = os.path.join(cfg.data_dir, 'train')
    val_dir = os.path.join(cfg.data_dir, 'val', 'images')
    train_count = sum(len(os.listdir(os.path.join(train_dir, d, 'images'))) 
                     for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d, 'images')))
    val_count = len([f for f in os.listdir(val_dir) if f.endswith(('.JPEG', '.jpg', '.png'))])
    
    steps_per_epoch = train_count // cfg.batch_size
    logger.info(f"Dataset OK | train samples: {train_count}, val samples: {val_count}, classes: {cfg.num_classes}, steps/epoch: {steps_per_epoch}")

    device, model, criterion, optimizer, scheduler, scaler, train_loader, val_loader = setup(cfg)
    try:
        log_model_info(logger, model, model_name="ResNet50 (Tiny-ImageNet)")
        log_dataset_info(logger, train_loader, val_loader)
    except Exception:
        pass

    history = {
        'train_loss': [], 'train_top1': [], 'train_top5': [],
        'val_loss': [], 'val_top1': [], 'val_top5': [],
        'lr': [],
        # Percent views
        'train_loss_pct': [], 'val_loss_pct': [], 'lr_pct': [],
        'train_acc_pct': [], 'val_acc_pct': []
    }
    best_val_top1 = -1.0
    baseline_train_loss = None
    total_steps_taken = 0  # Track total optimizer steps across all epochs

    logger.info("Starting training...")
    print()  # Add blank line before progress bars start

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        tr = train_epoch(cfg, device, model, criterion, optimizer, scheduler, scaler, train_loader, epoch, logger)
        
        # Validate  
        va = evaluate(cfg, device, model, criterion, val_loader)
        
        history['train_loss'].append(tr['loss'])
        history['train_top1'].append(tr['top1'])
        history['train_top5'].append(tr['top5'])
        history['val_loss'].append(va['loss'])
        history['val_top1'].append(va['top1'])
        history['val_top5'].append(va['top5'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Percent fields
        if baseline_train_loss is None:
            baseline_train_loss = max(1e-12, tr['loss'])
        train_loss_pct = (tr['loss'] / baseline_train_loss) * 100.0
        val_loss_pct = (va['loss'] / baseline_train_loss) * 100.0
        lr_pct = (optimizer.param_groups[0]['lr'] / max(1e-12, cfg.lr_max)) * 100.0
        history['train_loss_pct'].append(train_loss_pct)
        history['val_loss_pct'].append(val_loss_pct)
        history['lr_pct'].append(lr_pct)
        history['train_acc_pct'].append(tr['top1'])
        history['val_acc_pct'].append(va['top1'])

        # Log epoch results like train_imagenet.py
        logger.info(f"Train Loss: {tr['loss']:.4f}, Train Acc: {tr['top1']:.2f}%")
        logger.info(f"Val Loss: {va['loss']:.4f}, Val Acc: {va['top1']:.2f}%")

        if va['top1'] > best_val_top1:
            best_val_top1 = va['top1']
            ckpt = {
                'model': model.state_dict(),
                'config': {**cfg.__dict__},
                'epoch': epoch,
                'val_top1': best_val_top1,
                'train_loss': tr['loss'],
                'val_loss': va['loss'],
                'train_acc': tr['top1'],
                'val_acc': va['top1'],
            }
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.output_dir, 'best.pth'))
            logger.info(f"New best model saved! Accuracy: {best_val_top1:.2f}%")

        with open(os.path.join(cfg.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # Save final model and metrics
    torch.save({'model': model.state_dict()}, os.path.join(cfg.output_dir, 'last.pth'))
    final = {
        'epoch': cfg.epochs,
        'train_acc': history['train_top1'][-1] if history['train_top1'] else 0.0,
        'val_acc': history['val_top1'][-1] if history['val_top1'] else 0.0,
        'train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
        'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
        'best_val_acc': best_val_top1,
        'final_lr': history['lr'][-1] if history['lr'] else 0.0,
    }
    with open(os.path.join(cfg.output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    return final


# === CLI ===

def main():
    ap = argparse.ArgumentParser(description='Simplified Tiny-ImageNet Training')
    ap.add_argument('--data', type=str, required=True, help='Path to tiny-imagenet-200 dataset')
    ap.add_argument('--out', type=str, default='./runs', help='Output directory for training runs')
    ap.add_argument('--epochs', type=int, help='Number of epochs')
    ap.add_argument('--batch-size', type=int, help='Batch size')
    ap.add_argument('--lr-max', type=float, help='Maximum learning rate')
    ap.add_argument('--lr-min', type=float, help='Minimum learning rate')
    ap.add_argument('--wd', type=float, help='Weight decay')
    
    args = ap.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset path '{args.data}' does not exist")
        return 1
    
    # Build config with overrides
    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lr_max is not None:
        overrides['lr_max'] = args.lr_max
    if args.lr_min is not None:
        overrides['lr_min'] = args.lr_min
    if args.wd is not None:
        overrides['weight_decay'] = args.wd
    
    cfg = TrainConfig.from_auto(args.data, args.out, **overrides)
    config_path = cfg.save()
    print(f"Config saved to {config_path}")
    
    # Setup main logger
    log_file = os.path.join('logs', f'tiny_imagenet_training_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger('tiny_imagenet_training', log_file)
    log_system_info(logger)
    
    final_metrics = run_training(cfg)
    print("Final metrics:", json.dumps(final_metrics, indent=2))
    return 0


if __name__ == '__main__':
    exit(main())