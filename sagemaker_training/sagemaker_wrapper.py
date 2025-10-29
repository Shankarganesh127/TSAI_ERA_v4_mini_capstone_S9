#!/usr/bin/env python3
"""
SageMaker wrapper:
- Parses hyperparameters with SageMaker-friendly defaults
- Calls imagenet_training_pipeline.main() directly (one process per rank)
- Leaves DDP orchestration to SageMaker (WORLD_SIZE/LOCAL_RANK env)
- Passes flags so rank-0 can generate research reports (CSV/JSON/plots + TensorBoard)
"""

import os
import sys
import argparse
from pathlib import Path

from logger_setup import setup_unified_logger, get_unified_logger
from imagenet_training_pipeline import main as training_main

def parse_hyperparams():
    p = argparse.ArgumentParser()
    # SageMaker channels
    p.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    p.add_argument("--val_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # Core
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "100")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "0.4")))
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true")

    # Dist / device
    p.add_argument("--ddp", type=str, default=os.environ.get("DDP", "auto"))  # auto|true|false
    p.add_argument("--debug_ddp", action="store_true")
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))

    # QoL
    p.add_argument("--quick-mode", action="store_true")

    # Auto-optimizers toggles (enabled by default for research)
    p.add_argument("--use-batchsize-finder", type=str, default=os.environ.get("USE_BSF", "true"))
    p.add_argument("--use-lr-finder", type=str, default=os.environ.get("USE_LRF", "true"))
    p.add_argument("--use-wd-search", type=str, default=os.environ.get("USE_WDS", "true"))
    p.add_argument("--use-workers-auto", type=str, default=os.environ.get("USE_WORKERS_AUTO", "true"))
    return p.parse_args()

def main():
    setup_unified_logger()
    log = get_unified_logger("sagemaker_wrapper")

    args = parse_hyperparams()

    # Validate channels exist (SageMaker mounts them)
    if not Path(args.data_dir).exists():
        log.error(f"Training channel not found: {args.data_dir}")
        sys.exit(1)
    if not Path(args.val_dir).exists():
        log.error(f"Validation channel not found: {args.val_dir}")
        sys.exit(1)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build argv for the pipeline
    argv = [
        "--train", args.data_dir,
        "--val", args.val_dir,
        "--output", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--ddp", args.ddp,
        "--device", args.device,
        "--use-batchsize-finder", args.use_batchsize_finder,
        "--use-lr-finder", args.use_lr_finder,
        "--use-wd-search", args.use_wd_search,
        "--use-workers-auto", args.use_workers_auto,
    ]
    if args.pretrained:
        argv.append("--pretrained")
    if args.quick_mode:
        argv.append("--quick-mode")
    if args.debug_ddp:
        argv.append("--debug-ddp")

    log.info(f"[LAUNCH] imagenet_training_pipeline.py {' '.join(argv)}")
    sys.argv = ["imagenet_training_pipeline.py"] + argv
    training_main()
    log.info("[DONE] Training pipeline finished successfully.")

if __name__ == "__main__":
    main()
