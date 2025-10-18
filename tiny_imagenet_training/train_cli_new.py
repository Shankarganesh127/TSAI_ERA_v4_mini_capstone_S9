#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
from config import TrainConfig
from trainer import run_training


def main():
    ap = argparse.ArgumentParser(description='Tiny-ImageNet Training CLI')
    ap.add_argument('--data', type=str, required=True, help='Path to tiny-imagenet-200 root')
    ap.add_argument('--out', type=str, default='./runs_tiny', help='Output root for runs')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr-max', type=float, default=None)
    ap.add_argument('--lr-min', type=float, default=None)
    ap.add_argument('--wd', type=float, default=None)
    args = ap.parse_args()

    cfg = TrainConfig.from_auto(args.data, args.out)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.lr_max is not None:
        cfg.lr_max = float(args.lr_max)
    if args.lr_min is not None:
        cfg.lr_min = float(args.lr_min)
    if args.wd is not None:
        cfg.weight_decay = float(args.wd)

    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save()
    print('Config saved to', os.path.join(cfg.output_dir, 'config.json'))

    metrics = run_training(cfg)
    print('Final metrics:', json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
