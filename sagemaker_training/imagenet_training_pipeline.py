#!/usr/bin/env python3
"""
ImageNet Training Pipeline — SageMaker-ready (single node multi-GPU)

Features
- DDP-safe (init process group → run optimizers on rank 0 → broadcast → DDP wrap)
- AMP (autocast + GradScaler), gradient clipping
- Correct order: optimizer.step() → scheduler.step()
- Dynamic CPU threads per stage (preprocess / training / validation)
- Per-process DataLoader workers (global optimal divided by WORLD_SIZE)
- Research reporting:
  • CSV + JSON logs per epoch + final summary
  • CSV/JSON + plot for BatchSizeFinder, LRFinder, WeightDecay search, num_workers
  • TensorBoard (rank 0)
  • SageMaker metrics printing (CloudWatch)
- Works with your existing modules:
  imagenet_dataset.get_imagenet_dataloaders,
  imagenet_models.resnet50_imagenet,
  training_performance_optimizer (if present),
  utils.is_main_process,
  logger_setup.get_unified_logger
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

import multiprocessing as mp
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from torch.utils.tensorboard import SummaryWriter

from imagenet_models import resnet50_imagenet, resnet50_imagenet_no_ddp
from imagenet_dataset import get_imagenet_dataloaders
from logger_setup import get_unified_logger
from utils import is_main_process as _is_main_process  # existing util


# Optional: helps when combined with multi-process DataLoader
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# --- SAFETY: Prevent PyTorch thread reconfiguration errors ---
# (set BEFORE any DataLoader or model init)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "600"  # seconds


# Optional imports — fall back gracefully if not present
_BSF = _LRF = _WDS = _OPTW = None
try:
    from training_performance_optimizer import (
        BatchSizeFinder as _BSF,
        LRFinder as _LRF,
        HyperparameterOptimizer as _WDS,   # we'll use .weight_decay_search(...)
        optimize_num_workers as _OPTW       # if available in your project
    )
except Exception:
    pass

# ---------------------------------------------------------------------
# Environment hygiene
# ---------------------------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
TQDM_DISABLE = os.environ.get("TQDM_DISABLE", "0") == "1"

log = get_unified_logger("imagenet_pipeline")

def safe_broadcast_scalar(local_value, dtype, device, src=0, tag=""):
    """
    Broadcast scalar safely across all ranks. 
    Ensures every rank reaches the barrier even if one rank fails early.
    """
    if not dist.is_initialized():
        return local_value
    rank = dist.get_rank()
    world = dist.get_world_size()
    try:
        # Force all ranks to reach the same barrier
        dist.barrier()
        tensor = torch.tensor(local_value if rank == src else 0, dtype=dtype, device=device)
        dist.broadcast(tensor, src=src)
        return tensor.item()
    except Exception as e:
        print(f"[safe_broadcast_scalar:{tag}] Rank {rank}/{world} failed: {e}", flush=True)
        return local_value

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def is_main_process():
    return _is_main_process()

def set_stage_threads(stage: str, prefer_threads: int | None = None) -> int:
    """Dynamically set PyTorch CPU threads by stage, safely (won’t crash if too late)."""
    import psutil, torch, multiprocessing as mp

    cpu_count = psutil.cpu_count(logical=True) or mp.cpu_count()
    if prefer_threads is not None:
        threads = max(1, int(prefer_threads))
    else:
        if stage == "preprocess":
            threads = max(2, min(cpu_count, cpu_count // 2 or 1))
        elif stage == "training":
            threads = 1
        elif stage == "validation":
            threads = max(1, min(4, (cpu_count // 4) or 1))
        else:
            threads = max(1, cpu_count // 2 or 1)

    # --- 🔒 SAFE SET (won’t crash if late)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(min(4, max(1, threads)))
    except RuntimeError as e:
        # too late -> just log a warning and continue
        log.warning(f"[THREADS] Could not change thread count for stage={stage}: {e}")

    log.info(f"[THREADS] Stage={stage} num_threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    return threads



def init_dist_if_needed(args) -> tuple[bool, int, int, torch.device]:
    """Initialize process group if DDP requested/available, but DO NOT wrap the model yet."""
    # Determine world size + local rank
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SM_NUM_GPUS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    ddp_choice = str(args.ddp).lower()

    if ddp_choice == "auto":
        use_ddp = world_size > 1 and torch.cuda.is_available()
    elif ddp_choice in ("true", "1", "yes"):
        use_ddp = torch.cuda.is_available() and world_size > 1
    else:
        use_ddp = False

    if not use_ddp:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        log.info(f"[DIST] DDP disabled. Single process on {device}")
        return False, 1, 0, device

    if args.debug_ddp:
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        log.info("[DIST] DDP debug env enabled")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log.info(f"[DIST] Initialized: world_size={world_size} local_rank={local_rank} device={device}")
    return True, world_size, local_rank, device


def ddp_wrap(model: nn.Module, local_rank: int) -> nn.Module:
    """Make params contiguous → move to device → DDP wrap."""
    # Ensure contiguous params BEFORE wrapping
    model = model.to("cpu")
    fixed = 0
    for n, p in model.named_parameters():
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
            fixed += 1
    if fixed:
        log.warning(f"[DDP] Made {fixed} parameters contiguous")
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )
    log.info("[DDP] Model wrapped with DistributedDataParallel")
    return model


def broadcast_scalar(value, dtype, device, src=0):
    """Broadcast a scalar to all ranks. Returns Python scalar."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=dtype, device=device)
    dist.broadcast(t, src=src)
    return t.item()


def ensure_reports_dir(output_dir: str) -> Path:
    reports = Path(output_dir) / "reports"
    (reports / "epoch_reports").mkdir(parents=True, exist_ok=True)
    return reports


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_csv_row(path: Path, fieldnames: list[str], row: dict):
    newfile = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if newfile:
            w.writeheader()
        w.writerow(row)


def plot_xy(x, y, xlabel, ylabel, title, outpath: Path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()


# ---------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler: GradScaler, accumulation_steps: int, writer: SummaryWriter | None, epoch: int):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    step_idx = 0
    log.info(f"[AMP] GradScaler created with enabled={scaler.is_enabled()}")

    for step_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast(enabled=scaler.is_enabled()):
            if step_idx == 1 and is_main_process():
                log.info(f"[AMP] autocast active: {torch.is_autocast_enabled()}")
                
            if step_idx == 1 and is_main_process():
                print(f"[DEBUG] inputs.device={inputs.device}, model.device={next(model.parameters()).device}")

            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)

        if not torch.isfinite(loss):
            log.error(f"[TRAIN] Non-finite loss detected: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scale_loss = loss / accumulation_steps
        if scaler.is_enabled():
            scaler.scale(scale_loss).backward()
        else:
            scale_loss.backward()

        if step_idx % accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            total_correct += (pred == targets).sum().item()
            total += targets.size(0)

        # SageMaker metrics to stdout (CloudWatch)
        if is_main_process() and step_idx % 50 == 0:
            print(f"train_loss={loss.item():.6f}")  # CloudWatch metric
            sys.stdout.flush()

    avg_loss = total_loss / max(1, total)
    top1 = 100.0 * total_correct / max(1, total)

    if writer and is_main_process():
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch_top1", top1, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    return avg_loss, top1


@torch.no_grad()
def validate(model, loader, device, writer: SummaryWriter | None, epoch: int):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss_sum += loss.item() * inputs.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    val_loss = loss_sum / max(1, total)
    val_top1 = 100.0 * correct / max(1, total)

    if writer and is_main_process():
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)

        # also print to CloudWatch
        print(f"val_top1={val_top1:.6f}")
        print(f"val_loss={val_loss:.6f}")
        sys.stdout.flush()

    return val_loss, val_top1


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # IO
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    p.add_argument("--val", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    p.add_argument("--output", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    # Core
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.4)  # will scale by global batch / 256
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--enable-amp", type=str, default="true")
    # Dist / device
    p.add_argument("--ddp", type=str, default="auto", help="auto|true|false")
    p.add_argument("--debug_ddp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--local_rank", type=int, default=0)
    # Data loader workers (global suggestion; divided per process)
    p.add_argument("--num-workers", type=int, default=8)
    # QoL
    p.add_argument("--quick-mode", action="store_true")
    # Auto-optimizers toggles
    p.add_argument("--use-batchsize-finder", type=str, default="true")
    p.add_argument("--use-lr-finder", type=str, default="true")
    p.add_argument("--use-wd-search", type=str, default="true")
    p.add_argument("--use-workers-auto", type=str, default="true")
    return p.parse_args()


# --- Disable DDP temporarily so rank0 can test batch sizes safely ---
def disable_ddp_model(model: nn.Module) -> nn.Module:
    was_ddp = dist.is_initialized()
    if was_ddp:
        if hasattr(model, "module"):  # DDP-wrapped model
            model_to_use = model.module
        else:
            model_to_use = model
        log.info(f"[DDP] Rank {dist.get_rank()} ready on device {torch.cuda.current_device()}")
        log.info(f"[DDP] World size = {dist.get_world_size()}")
        dist.barrier()  # sync before disabling
        log.info(f"[DDP] Rank {dist.get_rank()} passed barrier.")
    else:
        model_to_use = model
    return was_ddp, model_to_use

def enable_ddp_model(model: nn.Module, was_ddp: bool, local_rank: int) -> nn.Module:
    if was_ddp:
        dist.barrier()  # sync before re-enabling
        model = ddp_wrap(model, local_rank)
    return model

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    args.enable_amp = (str(args.enable_amp).lower() == "true")
    use_bsf = (str(args.use_batchsize_finder).lower() == "true") and (_BSF is not None)
    use_lrf = (str(args.use_lr_finder).lower() == "true") and (_LRF is not None)
    use_wds = (str(args.use_wd_search).lower() == "true") and (_WDS is not None)
    use_workers_auto = (str(args.use_workers_auto).lower() == "true") and (_OPTW is not None)

    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    Path(args.output).mkdir(parents=True, exist_ok=True)
    reports_dir = ensure_reports_dir(args.output)
    (Path(args.output) / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    # Init (maybe) distributed for coordination — but don't wrap model yet
    dist_on, world_size, local_rank, device = init_dist_if_needed(args)

    # ----------------- Stage: Preprocessing / Auto-optimizers (rank 0) -----------------
    set_stage_threads("preprocess")

    # (A) num_workers optimizer
    if use_workers_auto:
        if is_main_process():
            log.info("[AUTO] Optimizing num_workers...")
            try:
                optimal_workers = _OPTW(dataset_path=args.train, max_workers=args.num_workers)
            except TypeError:
                # Some versions may have different signature; fallback to simple heuristic
                cpu_logical = psutil.cpu_count(logical=True) or mp.cpu_count()
                optimal_workers = min(args.num_workers, max(1, cpu_logical // 2))
            # Report & persist
            save_json(reports_dir / "num_workers.json", {"optimal_workers_global": int(optimal_workers)})
        else:
            # Non-main ranks: just wait for rank-0 to finish
            optimal_workers = args.num_workers
            log.info(f"[AUTO] non-main num_workers set to {optimal_workers} (global)")
            
        # Broadcast to all ranks so everyone uses the same worker count
        if dist.is_initialized():
            optimal_workers = safe_broadcast_scalar(optimal_workers, torch.int32, device, tag="num_workers")
            args.num_workers = int(optimal_workers)
            log.info(f"[AUTO] num_workers set to {args.num_workers} (global)")

    else:
        log.info(f"[AUTO] Skipping num_workers auto-tuning (using {args.num_workers})")

    # (B) BatchSize Finder
    if use_bsf:
        
        if is_main_process():
            log.info("[AUTO] Running BatchSizeFinder...")
            
            tmp_model = resnet50_imagenet_no_ddp(num_classes=1000, pretrained=False).to("cuda" if torch.cuda.is_available() else "cpu")
            
            log.info("[AUTO] Running BatchSizeFinder model initialized.")
            
            # Safety check for weights
            for name, p in tmp_model.named_parameters():
                if not torch.isfinite(p).all():
                    raise RuntimeError(
                        f"[BSF][INIT] Non-finite weights in {name}: "
                        f"min={p.data.min().item()}, max={p.data.max().item()}"
                    )
            
            log.info("[AUTO] Running BatchSizeFinder model initialized and weights checked.")
            
            #was_ddp, noddp_model = disable_ddp_model(model)
            reports_dir.mkdir(parents=True, exist_ok=True)

            log.info("[AUTO] Running BatchSizeFinder model disabled DDP temporarily.")

            try:
                bsf = _BSF(
                    model=tmp_model,
                    optimizer_cls=optim.SGD,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    train_dir=args.train,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_workers=args.num_workers,
                )

                log.info("[AUTO] Running BatchSizeFinder instance created, starting search...")

                bsf_result = bsf.find_max_batch(
                    start_bs=max(32, args.batch_size // 2), max_bs=2048
                )
                best_bs = int(bsf_result.get("best_batch_size", args.batch_size))

                log.info("[AUTO] Running BatchSizeFinder batch size search complete.")

                # Save report + plot
                save_json(reports_dir / "batchsize_finder.json", bsf_result)
                
                # clean up
                del tmp_model, bsf
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if "curve" in bsf_result:
                    xs = [p["bs"] for p in bsf_result["curve"]]
                    ys = [p["alloc_gb"] for p in bsf_result["curve"]]
                    plot_xy(
                        xs,
                        ys,
                        "batch_size",
                        "allocated_GB",
                        "Batch Size vs Allocated VRAM",
                        reports_dir / "batchsize_finder.png",
                    )

            except Exception as e:
                log.warning(f"[AUTO] BatchSizeFinder failed, keeping batch_size={args.batch_size}: {e}")
                best_bs = args.batch_size
        else:
            best_bs = args.batch_size
            log.info(f"[AUTO] non-main best batch_size set to {best_bs} (global)")

        # --- ✅ Reliable broadcast replaces broadcast_scalar ---
        if dist.is_initialized():
            best_bs = safe_broadcast_scalar(best_bs, torch.int32, device, tag="batchsize")
        args.batch_size = int(best_bs)
        log.info(f"[AUTO] batch_size set to {args.batch_size}")

    # (C) LR Finder (rank 0 only)
    if use_lrf:
        
        if is_main_process():
            log.info("[AUTO] Running LRFinder...")
            
            tmp_model = resnet50_imagenet_no_ddp(num_classes=1000, pretrained=False).to("cuda" if torch.cuda.is_available() else "cpu")

            log.info("[AUTO] Running LRFinder model initialized.")

            # Safety check for weights
            for name, p in tmp_model.named_parameters():
                if not torch.isfinite(p).all():
                    raise RuntimeError(
                        f"[BSF][INIT] Non-finite weights in {name}: "
                        f"min={p.data.min().item()}, max={p.data.max().item()}"
                    )

            log.info("[AUTO] Running LRFinder model initialized and weights checked.")

            reports_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                temp_device = "cuda" if torch.cuda.is_available() else "cpu"
                lrf = _LRF(
                    model=tmp_model,
                    train_dir=args.train,
                    batch_size=args.batch_size,
                    device=temp_device,
                    num_workers=args.num_workers,
                )

                log.info("[AUTO] Running LRFinder instance created, starting search...")

                lr_report = lrf.find(start_lr=1e-6, end_lr=1, iters=200)
                best_lr = float(lr_report.get("suggested_max_lr", args.lr))


                log.info("[AUTO] Running LRFinder search complete.")

                # Save report + plot (rank 0 only)
                save_json(reports_dir / "lr_finder.json", lr_report)
                
                del tmp_model, lrf
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if "curve" in lr_report:
                    xs = [p["lr"] for p in lr_report["curve"]]
                    ys = [p["loss"] for p in lr_report["curve"]]
                    plot_xy(xs, ys, "learning_rate", "loss",
                            "LR Range Test", reports_dir / "lr_finder.png")
                    
            except Exception as e:
                log.warning(f"[AUTO] LRFinder failed, keeping lr={args.lr}: {e}")
                best_lr = args.lr
        else:
            best_lr = args.lr
            log.info(f"[AUTO] non-main best learning_rate set to {best_lr} (global)")

        if dist.is_initialized():
            best_lr = safe_broadcast_scalar(best_lr, torch.float32, device, tag="lr")
        args.lr = float(best_lr)
        log.info(f"[AUTO] lr set to {args.lr}")
        
    # (D) Weight-decay search (rank 0 only)
    if use_wds:
            
        log.info("[AUTO] Running WeightDecay Search...")
            
        tmp_model = resnet50_imagenet_no_ddp(num_classes=1000, pretrained=False).to("cuda" if torch.cuda.is_available() else "cpu")
            
        log.info("[AUTO] Running Weight-Decay Search model initialized.")
            
        # Safety check for weights
        for name, p in tmp_model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"[BSF][INIT] Non-finite weights in {name}: "
                    f"min={p.data.min().item()}, max={p.data.max().item()}"
                )
            
        log.info("[AUTO] Running Weight-Decay Search model initialized and weights checked.")
                
        reports_dir.mkdir(parents=True, exist_ok=True)
        log.info("[AUTO] Running Weight-Decay Search...")
        try:
            temp_device = "cuda" if torch.cuda.is_available() else "cpu"
            # This HPO runs per-rank independent trials internally when DDP is active
            wds = _WDS(
                model_fn=lambda: tmp_model,
                train_dir=args.train,
                val_dir=args.val,
                device=temp_device,
                num_workers=args.num_workers,
            )

            log.info("[AUTO] Running Weight-Decay Search...")
            
            lr_config = {"min_lr": max(3e-6, args.lr / 10.0), "max_lr": args.lr }
            results, best_wd = wds.weight_decay_search(lr_config, batch_size=args.batch_size)
            
            log.info("[AUTO] Running Weight-Decay Search complete.")
    
            save_json(reports_dir / "weight_decay_search.json",
                      {"results": results, "best_weight_decay": float(best_wd)})
            
            del tmp_model, wds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            log.warning(f"[AUTO] WD search failed, keeping weight_decay={args.weight_decay}: {e}")
            best_wd = args.weight_decay
    
        if dist.is_initialized():
            best_wd = safe_broadcast_scalar(best_wd, torch.float32, device, tag="wd")
        args.weight_decay = float(best_wd)
        log.info(f"[AUTO] weight_decay set to {args.weight_decay}")

    # ----------------- Build final model & wrap for DDP -----------------
    # Always create a fresh model here – never reuse noddp_model or anything from finders.
    model = resnet50_imagenet(num_classes=1000, pretrained=args.pretrained)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()        # clear leftover allocations

    # Make sure model and optimizer are on the correct local device
    if dist_on:
        torch.cuda.set_device(local_rank)
        model = model.to(f"cuda:{local_rank}")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False
        )
        log.info(f"[imagenet_training_pipeline] ✅ Model wrapped with DDP on device cuda:{local_rank} (backend=nccl)")
    else:
        model = model.to(device)

    # ----------------- Scale LR by global batch -----------------
    global_batch = args.batch_size * (dist.get_world_size() if dist.is_initialized() else 1)
    scaled_lr = args.lr * (global_batch / 256.0)
    if is_main_process():
        log.info(f"[LR] base_lr={args.lr} → scaled_lr={scaled_lr:.6f} (global_batch={global_batch})")

    # ----------------- DataLoaders (per-process workers) -----------------
    per_proc_workers = max(1, int(args.num_workers // (dist.get_world_size() if dist.is_initialized() else 1)))
    if is_main_process():
        log.info(f"[DATALOADER] global_workers={args.num_workers} world_size={dist.get_world_size() if dist.is_initialized() else 1} → per_proc={per_proc_workers}")

    train_loader, val_loader, train_batches_per_epoch, _ = get_imagenet_dataloaders(
        args.train,
        args.val,
        batch_size=args.batch_size,
        num_workers=per_proc_workers,
        pin_memory=True,
        # IMPORTANT: for full DDP training we want per-rank splitting
        disable_distributed_splitting=False,
    )

    # ----------------- Optimizer / Scheduler / AMP -----------------
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    steps_per_epoch = max(1, train_batches_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = OneCycleLR(optimizer, max_lr=scaled_lr, total_steps=total_steps)

    scaler = GradScaler(enabled=(args.enable_amp and torch.cuda.is_available()))

    target_effective_batch = 256
    accumulation_steps = max(1, target_effective_batch // max(1, args.batch_size))

    # ----------------- TensorBoard (rank 0) -----------------
    writer = SummaryWriter(log_dir=str(reports_dir / "tensorboard")) if is_main_process() else None

    # ----------------- Training loop -----------------
    set_stage_threads("training")
    best_val = -1.0
    csv_path = reports_dir / "training_log.csv"
    csv_fields = ["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "lr", "epoch_time_sec"]

    for epoch in range(1, args.epochs + 1):
        if is_main_process():
            log.info(f"===== EPOCH {epoch}/{args.epochs} =====")

        t0 = time.time()
        train_loss, train_top1 = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, accumulation_steps, writer, epoch)
        set_stage_threads("validation")
        val_loss, val_top1 = validate(model, val_loader, device, writer, epoch)
        set_stage_threads("training")
        epoch_time = time.time() - t0

        # Metric prints for CloudWatch
        if is_main_process():
            print(f"epoch={epoch}")
            print(f"epoch_time_sec={epoch_time:.3f}")
            print(f"epoch_train_loss={train_loss:.6f}")
            print(f"epoch_val_loss={val_loss:.6f}")
            print(f"epoch_val_top1={val_top1:.6f}")
            sys.stdout.flush()

        if is_main_process():
            append_csv_row(csv_path, csv_fields, {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_top1": f"{train_top1:.4f}",
                "val_loss": f"{val_loss:.6f}",
                "val_top1": f"{val_top1:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                "epoch_time_sec": f"{epoch_time:.2f}",
            })
            # per-epoch JSON (optional)
            save_json(reports_dir / "epoch_reports" / f"epoch_{epoch:03d}.json", {
                "epoch": epoch,
                "train_loss": train_loss, "train_top1": train_top1,
                "val_loss": val_loss, "val_top1": val_top1,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time
            })

            if val_top1 > best_val:
                best_val = val_top1
                torch.save({"model": (model.module if hasattr(model, "module") else model).state_dict(),
                            "epoch": epoch, "val_top1": val_top1},
                           Path(args.output) / "best.pth")
                log.info(f"[CHECKPOINT] New best top1={best_val:.2f}% → saved best.pth")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.quick_mode and epoch >= 1:
            break

    # Final save + summary
    if is_main_process():
        torch.save({"model": (model.module if hasattr(model, "module") else model).state_dict(),
                    "epoch": epoch, "best_val_top1": best_val},
                   Path(args.output) / "final.pth")
        save_json(reports_dir / "training_summary.json", {
            "best_val_top1": best_val,
            "epochs": epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "scaled_lr": scaled_lr,
            "weight_decay": args.weight_decay,
            "num_workers_global": args.num_workers,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "timestamp": datetime.utcnow().isoformat()
        })
        log.info("[DONE] Training complete — artifacts written to model dir and reports/")

    if writer:
        writer.flush()
        writer.close()

    if dist.is_initialized():
        log.info(f"[DDP] Rank {dist.get_rank()} ready on device {torch.cuda.current_device()}")
        log.info(f"[DDP] World size = {dist.get_world_size()}")
        dist.barrier()  # ensure rank-0 wrote the file before broadcast
        log.info(f"[DDP] Rank {dist.get_rank()} passed barrier.")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
