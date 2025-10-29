#!/usr/bin/env python3
"""
training_performance_optimizer.py

Research-grade utilities:
- BatchSizeFinder: Find largest batch size that fits in GPU memory (quick OOM-safe search)
- LRFinder: LR range test (records curve + suggests a max LR)
- HyperparameterOptimizer: weight_decay_search(...) with DDP-aware parallel partitioning
- optimize_num_workers: quick dataloader benchmark to suggest a global num_workers

Design goals:
- Safe in SageMaker DDP: rank 0 coordinates, all ranks use the same tuned results
- Minimal side-effects: temporary models/loaders for probes; your main pipeline builds the final ones
- Produces JSON-ready result dicts (curves, suggestions) for your reporting layer

Assumptions:
- imagenet_dataset.get_imagenet_dataloaders(...) exists in your project
- imagenet_models.resnet50_imagenet(...) or a model_fn supplied by caller
"""

from __future__ import annotations
import os
import time
import math
import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

# Import your dataset loader util
from imagenet_dataset import get_imagenet_dataloaders
from logger_setup import get_unified_logger

log = get_unified_logger("tpo")

# ----------------------------
# Helpers
# ----------------------------
def _is_ddp() -> bool:
    return dist.is_initialized()

def _rank() -> int:
    return dist.get_rank() if _is_ddp() else 0

def _world() -> int:
    return dist.get_world_size() if _is_ddp() else 1

def _bcast_scalar(value, dtype, device, src: int = 0):
    """Broadcast a scalar from src to all ranks; return Python scalar."""
    if not _is_ddp():
        return value
    t = torch.tensor(value, dtype=dtype, device=device)
    dist.broadcast(t, src=src)
    return t.item()

def _gather_dicts_local_to_rank0(local: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gather arbitrary Python dicts from all ranks to rank 0 (using all_gather_object)."""
    objs = [None for _ in range(_world())]
    if _is_ddp():
        dist.all_gather_object(objs, local)
    else:
        objs = [local]
    return objs

def _suggest_max_lr_from_curve(curve: List[Dict[str, float]]) -> float:
    """
    Basic heuristic:
    - Smooth minimal loss L_min
    - Choose an LR where loss starts increasing sharply: 
      pick LR at 0.8 * argmin(loss), or 1/10 of LR at loss min, whichever is smaller but > start.
    """
    if not curve:
        return 1e-3
    losses = [p["loss"] for p in curve]
    lrs = [p["lr"] for p in curve]
    i_min = max(0, int(losses.index(min(losses))))
    # conservative: pick 0.1 * LR at minimal loss, but not smaller than first LR
    return max(lrs[0], lrs[i_min] * 0.1)


# ----------------------------
# Batch Size Finder
# ----------------------------
class BatchSizeFinder:
    """
    Finds the largest per-process batch size that fits in memory with a quick forward/backward loop.

    API:
        bsf = BatchSizeFinder(model, optimizer_cls, lr, momentum, weight_decay, train_dir, device="cuda")
        result = bsf.find_max_batch(start_bs=64, max_bs=2048, steps=20)
        result = {
            "best_batch_size": 448,
            "curve": [{"bs": 64, "ok": True, "alloc_gb": 3.1}, ...],
            "device": "cuda:0",
            "rank": 0
        }
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: Callable[..., optim.Optimizer],
        lr: float,
        momentum: float,
        weight_decay: float,
        train_dir: str,
        device: str = "cuda",
        num_workers: int = 2,
    ):
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_dir = train_dir
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.num_workers = num_workers

    def _probe(self, bs: int, steps: int = 20) -> Dict[str, Any]:
        # small loader for a few iterations
        train_loader, _, train_batches, _ = get_imagenet_dataloaders(
            self.train_dir, self.train_dir,
            batch_size=bs,
            num_workers=self.num_workers,
            pin_memory=True,
            # BS finder runs standalone; do not split across ranks
            disable_distributed_splitting=True,
        )

        model = self.model.to(self.device)
        model.train()
        opt = self.optimizer_cls(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        torch.cuda.reset_peak_memory_stats(self.device) if torch.cuda.is_available() else None

        ok = True
        iters = min(steps, train_batches)
        try:
            it_count = 0
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with autocast(enabled=scaler.is_enabled()):
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                opt.zero_grad(set_to_none=True)

                it_count += 1
                if it_count >= iters:
                    break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                ok = False
            else:
                raise
        finally:
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        alloc_gb = 0.0
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        return {"bs": bs, "ok": ok, "alloc_gb": float(alloc_gb), "device": str(self.device), "rank": _rank()}

    def find_max_batch(self, start_bs: int = 64, max_bs: int = 2048, steps: int = 20) -> Dict[str, Any]:
        """
        1) Geometric growth until OOM (or max)
        2) Binary search between last OK and first OOM
        """
        curve = []
        bs = max(1, int(start_bs))
        last_ok = 0
        first_oom = None

        # ramp-up
        while bs <= max_bs:
            r = self._probe(bs, steps=steps)
            curve.append(r)
            if r["ok"]:
                last_ok = bs
                bs *= 2
            else:
                first_oom = bs
                break

        if last_ok == 0:
            # even start_bs failed; return conservative choice
            return {"best_batch_size": start_bs, "curve": curve, "device": str(self.device), "rank": _rank()}

        # binary search
        lo, hi = last_ok, (first_oom if first_oom is not None else last_ok)
        if first_oom is None:
            # never OOM'ed up to max_bs; accept last_ok
            return {"best_batch_size": last_ok, "curve": curve, "device": str(self.device), "rank": _rank()}

        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = self._probe(mid, steps=steps)
            curve.append(r)
            if r["ok"]:
                lo = mid
            else:
                hi = mid

        return {"best_batch_size": lo, "curve": curve, "device": str(self.device), "rank": _rank()}


# ----------------------------
# LR Finder
# ----------------------------
class LRFinder:
    """
    Classic LR range test (Leslie Smith).
    - linearly/exponentially increase LR across a short run and record loss curve
    - suggest a conservative max LR based on min loss region

    API:
        lrf = LRFinder(model, train_dir, batch_size, device)
        report = lrf.find(start_lr=1e-6, end_lr=1.0, iters=200, mode="exp")
        report = {"curve":[{"lr":..., "loss":...}, ...], "suggested_max_lr":..., "rank":0}
    """

    def __init__(self, model: nn.Module, train_dir: str, batch_size: int, device: str = "cuda", num_workers: int = 2):
        self.model = model
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.num_workers = num_workers

    def find(self, start_lr: float = 1e-6, end_lr: float = 1.0, iters: int = 200, mode: str = "exp") -> Dict[str, Any]:
        train_loader, _, train_batches, _ = get_imagenet_dataloaders(
            self.train_dir, self.train_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # LR finder runs standalone; do not split across ranks
            disable_distributed_splitting=True,
        )

        model = self.model.to(self.device)
        model.train()

        opt = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        curve = []
        lr = start_lr
        gamma = (end_lr / start_lr) ** (1.0 / max(1, iters)) if mode == "exp" else (end_lr - start_lr) / max(1, iters)

        it_count = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # set LR for this step
            for pg in opt.param_groups:
                pg["lr"] = lr

            with autocast(enabled=scaler.is_enabled()):
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)

            curve.append({"lr": float(lr), "loss": float(loss.item())})

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            opt.zero_grad(set_to_none=True)

            it_count += 1
            if it_count >= iters:
                break

            # update LR
            if mode == "exp":
                lr *= gamma
            else:
                lr += gamma

        # suggest LR
        suggested = _suggest_max_lr_from_curve(curve)
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"curve": curve, "suggested_max_lr": float(suggested), "rank": _rank()}


# ----------------------------
# Weight Decay Search (DDP-parallel)
# ----------------------------
class HyperparameterOptimizer:
    """
    DDP-aware weight decay search.
    - Split candidate WD list across ranks (disjoint subsets)
    - Each rank trains a small number of steps per candidate, evaluates on a small val split
    - Gather results to rank 0; choose best WD; return (results, best_wd)

    API:
        hpo = HyperparameterOptimizer(model_fn, train_dir, val_dir, device="cuda")
        results, best_wd = hpo.weight_decay_search(lr_config, batch_size, candidates=None, steps=200)

        results = [{"weight_decay": 1e-4, "val_top1": 67.1, "val_loss": 1.49, "rank": 1}, ...]
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        train_dir: str | None = None,
        val_dir: str | None = None,
        device: str = "cuda",
        num_workers: int = 2,
    ):
        self.model_fn = model_fn
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.num_workers = num_workers

    def _build_loaders(self, batch_size: int):
        assert self.train_dir is not None and self.val_dir is not None, \
            "train_dir and val_dir must be provided for weight_decay_search"
        train_loader, val_loader, train_batches, val_batches = get_imagenet_dataloaders(
            self.train_dir, self.val_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # WD search runs independently per rank (no DDP sampling)
            disable_distributed_splitting=True,
        )
        
        # ✅ Sanity check: make sure we received iterable WebDataset loaders
        assert hasattr(train_loader, "__iter__"), "Expected an iterable train_loader"
        assert hasattr(val_loader, "__iter__"), "Expected an iterable val_loader"

        return train_loader, val_loader, train_batches, val_batches

    @torch.no_grad()
    def _validate(self, model: nn.Module, loader) -> Dict[str, float]:
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return {
            "val_loss": loss_sum / max(1, total),
            "val_top1": 100.0 * correct / max(1, total),
        }

    def weight_decay_search(
        self,
        lr_config: Dict[str, float],
        batch_size: int,
        candidates: Optional[List[float]] = None,
        steps: int = 200,
    ):
        """
        Distributed strategy:
        - Build small train/val loaders per process
        - Split candidates among ranks: each rank trains/evaluates only its subset
        - all_gather_object to rank 0; choose best_wd; broadcast if caller needs

        lr_config: {"min_lr": ..., "max_lr": ...} -> we'll just use max_lr as constant LR here
        """
        if candidates is None:
            # sensible defaults spanning common WD orders
            candidates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

        world = _world()
        rank = _rank()
        # partition
        my_cands = [c for i, c in enumerate(candidates) if i % world == rank]

        train_loader, val_loader, train_batches, _ = self._build_loaders(batch_size=batch_size)
        max_iter = min(steps, train_batches)

        results_local = []
        for wd in my_cands:
            model = self.model_fn().to(self.device)
            model.train()
            opt = optim.SGD(model.parameters(), lr=float(lr_config.get("max_lr", 0.1)), momentum=0.9, weight_decay=wd, nesterov=True)
            scaler = GradScaler(enabled=torch.cuda.is_available())

            it = 0
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with autocast(enabled=scaler.is_enabled()):
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                opt.zero_grad(set_to_none=True)
                it += 1
                if it >= max_iter:
                    break

            # quick val
            metrics = self._validate(model, val_loader)
            res = {"weight_decay": float(wd), "rank": rank}
            res.update(metrics)
            results_local.append(res)

            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Gather to rank 0
        results_all = _gather_dicts_local_to_rank0({"results": results_local})
        if rank == 0:
            merged = []
            for pack in results_all:
                merged.extend(pack.get("results", []))
            # pick best by val_top1 (desc)
            if merged:
                merged_sorted = sorted(merged, key=lambda r: (r.get("val_top1", 0.0), -r.get("val_loss", 1e9)), reverse=True)
                best_wd = float(merged_sorted[0]["weight_decay"])
            else:
                best_wd = candidates[0]
            return merged, best_wd
        else:
            # return local — caller on non-rank0 can ignore or expect broadcast from pipeline
            return results_local, candidates[0]


# ----------------------------
# num_workers optimizer
# ----------------------------
def optimize_num_workers(dataset_path: str, max_workers: int = 8, probe_batches: int = 64, batch_size: int = 64) -> int:
    """
    Very fast, deterministic benchmark to suggest a global num_workers:
    - iterate workers in [1..max_workers]
    - time to load 'probe_batches' batches from a tiny loader (no model)
    - choose the smallest worker count within 5% of the best throughput (to avoid oversubscription)
    """
    from time import perf_counter

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = []

    for nw in range(1, max(2, max_workers) + 1):
        loader, _, steps, _ = get_imagenet_dataloaders(
            dataset_path, dataset_path, batch_size=batch_size, num_workers=nw, pin_memory=True
        )

        t0 = perf_counter()
        n = 0
        for x, y in loader:
            # emulate GPU pin-transfer cost
            x = x.to(device, non_blocking=True) if torch.cuda.is_available() else x
            n += 1
            if n >= probe_batches:
                break
        dt = max(1e-6, perf_counter() - t0)
        throughput = n / dt
        stats.append({"num_workers": nw, "throughput_bps": throughput})

    # pick best within 5% of top
    best = max(stats, key=lambda s: s["throughput_bps"])
    threshold = best["throughput_bps"] * 0.95
    candidates = [s["num_workers"] for s in stats if s["throughput_bps"] >= threshold]
    suggestion = int(min(candidates))
    log.info(f"[OPT_WORKERS] stats={stats} | best={best} | suggestion={suggestion}")
    return suggestion
