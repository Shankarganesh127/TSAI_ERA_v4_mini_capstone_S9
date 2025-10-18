#!/usr/bin/env python3
from __future__ import annotations
import os, json, logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from config import TrainConfig
from models import create_model
from datasets import build_dataloaders

# Try to import repo-level logger_setup dynamically; provide fallbacks if unavailable
import importlib.util, sys
def _load_repo_module(module_filename: str):
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(6):
        candidate = os.path.join(cur, module_filename)
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location(module_filename.replace('.py',''), candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            repo_root = os.path.dirname(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            spec.loader.exec_module(mod)
            return mod
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(f"Could not locate {module_filename} from {here}")

try:
    _log_mod = _load_repo_module('logger_setup.py')
    get_logger = _log_mod.get_logger
    log_model_info = _log_mod.log_model_info
    log_dataset_info = _log_mod.log_dataset_info
    log_system_info = _log_mod.log_system_info
except Exception:
    # Fallback lightweight logger utilities
    def get_logger(name: str):
        lg = logging.getLogger(name)
        if not lg.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
        return lg
    def log_model_info(logger, model, model_name="Model"):
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"{model_name}: total params {total:,}, trainable {trainable:,}")
        except Exception:
            pass
    def log_dataset_info(logger, train_loader, val_loader=None):
        try:
            logger.info(f"Dataset: train batches {len(train_loader)}, samples {len(train_loader.dataset)}")
            if val_loader:
                logger.info(f"Dataset: val batches {len(val_loader)}, samples {len(val_loader.dataset)}")
        except Exception:
            pass
    def log_system_info(logger):
        try:
            import torch as _t
            logger.info(f"PyTorch {_t.__version__}, CUDA {_t.cuda.is_available()}")
        except Exception:
            pass


# Module-level logger (will add per-run file handler in run_training)
logger = get_logger("tiny_imagenet_training")


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        n_class = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.eps / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() / target.size(0)) * 100.0)
        return res


def setup(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = build_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)
    # Print basic dataset stats to confirm path/structure
    try:
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_classes = len(getattr(train_loader.dataset, 'classes', [])) or cfg.num_classes
        steps_per_epoch = len(train_loader)
        logger.info(f"Dataset OK | train samples: {n_train}, val samples: {n_val}, classes: {n_classes}, steps/epoch: {steps_per_epoch}")
    except Exception as e:
        logger.warning(f"Dataset stats unavailable: {e}")
    # Always build from scratch (no pretrained weights)
    model = create_model(num_classes=cfg.num_classes, pretrained=False)
    if torch.cuda.is_available():
        model = model.to(device, memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        # Prefer new TF32 API when available; fallback to set_float32_matmul_precision
        try:
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'high'
        except Exception:
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
    else:
        model = model.to(device)

    criterion = LabelSmoothingCE(0.1)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_min, momentum=cfg.base_momentum, weight_decay=cfg.weight_decay, nesterov=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    div_factor = (cfg.lr_max / cfg.lr_min) if cfg.lr_min > 0 else 25.0
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr_max,
        total_steps=total_steps,
        pct_start=cfg.pct_start,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=cfg.base_momentum,
        max_momentum=cfg.max_momentum,
        div_factor=div_factor,
        final_div_factor=cfg.final_div_factor,
    )

    # Use new torch.amp API (avoids deprecation warnings)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp and device_type == 'cuda')

    return device, model, criterion, optimizer, scheduler, scaler, train_loader, val_loader


def train_epoch(cfg: TrainConfig, device, model, criterion, optimizer, scheduler, scaler, train_loader, progress_info: dict | None = None) -> Dict[str, float]:
    model.train()
    total_loss, total_top1, total_top5, total_count = 0.0, 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)


    for bidx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        if torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=cfg.amp and torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y) / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        if (bidx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        top1, top5 = accuracy(out, y, topk=(1,5))
        bs = y.size(0)
        total_loss += loss.item() * cfg.grad_accum_steps * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        total_count += bs
        if (bidx + 1) % max(1, getattr(cfg, 'log_every', 50)) == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            lr_pct = (cur_lr / max(1e-12, cfg.lr_max)) * 100.0
            logger.info(
                f"[train] step {bidx+1}/{len(train_loader)}: "
                f"loss {loss.item()*cfg.grad_accum_steps:.6f}, "
                f"top1 {top1:.2f}%, "
                f"lr {cur_lr:.6f} ({lr_pct:.2f}%)"
            )
        # Update progress bar, if provided
        if progress_info and 'obj' in progress_info and 'task' in progress_info:
            try:
                progress_info['obj'].update(progress_info['task'], advance=1)
            except Exception:
                pass
    # Completed one epoch
    return { 'loss': total_loss/total_count, 'top1': total_top1/total_count, 'top5': total_top5/total_count }


@torch.no_grad()
def evaluate(cfg: TrainConfig, device, model, criterion, val_loader, progress_info: dict | None = None) -> Dict[str, float]:
    model.eval()
    total_loss, total_top1, total_top5, total_count = 0.0, 0.0, 0.0, 0
    for bidx, (x, y) in enumerate(val_loader):
        x = x.to(device, non_blocking=True)
        if torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=cfg.amp and torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y)
        top1, top5 = accuracy(out, y, topk=(1,5))
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        total_count += bs
        if progress_info and 'obj' in progress_info and 'task' in progress_info:
            try:
                progress_info['obj'].update(progress_info['task'], advance=1)
            except Exception:
                pass
    return { 'loss': total_loss/total_count, 'top1': total_top1/total_count, 'top5': total_top5/total_count }


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    # Ensure per-run file logging into output_dir
    try:
        os.makedirs(cfg.output_dir, exist_ok=True)
        log_path = os.path.join(cfg.output_dir, 'train.log')
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path) for h in logger.handlers):
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Failed to attach per-run file handler: {e}")

    # Log system info and config
    try:
        log_system_info(logger)
    except Exception:
        pass
    try:
        logger.info("Training config:")
        logger.info(json.dumps({**cfg.__dict__}, indent=2))
    except Exception:
        pass

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

    # We only show per-epoch train and val bars (no global epochs bar)

    for epoch in range(cfg.epochs):
        # Train progress bar for this epoch
        train_bar = None
        train_task = None
        try:
            from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
            train_bar = Progress(
                TextColumn(f"[bold blue]Train (epoch {epoch+1}/{cfg.epochs})"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("• {task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=5,
            )
            train_task = train_bar.add_task("train_epoch", total=len(train_loader))
            train_bar.start()
        except Exception:
            pass

        tr = train_epoch(
            cfg, device, model, criterion, optimizer, scheduler, scaler, train_loader,
            progress_info={'obj': train_bar, 'task': train_task} if (train_bar and train_task is not None) else None
        )
        if train_bar is not None:
            try:
                train_bar.stop()
            except Exception:
                pass

        # Validation progress bar for this epoch
        val_bar = None
        val_task = None
        try:
            from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
            val_bar = Progress(
                TextColumn(f"[bold green]Val (epoch {epoch+1}/{cfg.epochs})"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("• {task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=5,
            )
            val_task = val_bar.add_task("val_epoch", total=len(val_loader))
            val_bar.start()
        except Exception:
            pass
        va = evaluate(
            cfg, device, model, criterion, val_loader,
            progress_info={'obj': val_bar, 'task': val_task} if (val_bar and val_task is not None) else None
        )
        if val_bar is not None:
            try:
                val_bar.stop()
            except Exception:
                pass
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

        # Compute percentage views
        if baseline_train_loss is None:
            baseline_train_loss = max(1e-12, tr['loss'])
        train_loss_pct = (tr['loss'] / baseline_train_loss) * 100.0
        val_loss_pct = (va['loss'] / baseline_train_loss) * 100.0
        cur_lr = optimizer.param_groups[0]['lr']
        lr_pct = (cur_lr / max(1e-12, cfg.lr_max)) * 100.0

        logger.info(
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"Train: acc {tr['top1']:.2f}%, loss {tr['loss']:.6f} ({train_loss_pct:.2f}%) | "
            f"Val: acc {va['top1']:.2f}%, loss {va['loss']:.6f} ({val_loss_pct:.2f}%) | "
            f"LR {cur_lr:.6f} ({lr_pct:.2f}%)"
        )


        if va['top1'] > best_val_top1:
            best_val_top1 = va['top1']
            ckpt = {
                'model': model.state_dict(),
                'config': {**cfg.__dict__},
                'epoch': epoch,
                'val_top1': best_val_top1
            }
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.output_dir, 'best.pth'))

        with open(os.path.join(cfg.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # Done training

    torch.save({'model': model.state_dict()}, os.path.join(cfg.output_dir, 'last.pth'))
    # Compose final metrics with percent views
    final = {
        'train_acc_pct': history['train_top1'][-1] if history['train_top1'] else None,
        'val_acc_pct': history['val_top1'][-1] if history['val_top1'] else None,
        'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'train_loss_pct': history['train_loss_pct'][-1] if history['train_loss_pct'] else None,
        'val_loss_pct': history['val_loss_pct'][-1] if history['val_loss_pct'] else None,
        'lr': history['lr'][-1] if history['lr'] else None,
        'lr_pct': history['lr_pct'][-1] if history['lr_pct'] else None,
    }
    with open(os.path.join(cfg.output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    return final
