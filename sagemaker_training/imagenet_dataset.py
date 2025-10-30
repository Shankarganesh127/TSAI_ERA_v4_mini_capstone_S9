import os
import math
import logging
from typing import Tuple, Optional, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import webdataset as wds
from torchvision import transforms

# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------
LOG = logging.getLogger("imagenet_dataset")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# small cache to avoid rebuilding the exact same loader several times in one run
_DATASET_CACHE = {}


def is_dist() -> bool:
    return dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# --------------------------------------------------------------------------------------
# core transforms (you can tweak to match your original file)
# --------------------------------------------------------------------------------------
def make_train_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def make_val_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# --------------------------------------------------------------------------------------
# defensive sample → (image, label) conversion
# this is the important part for your LR Finder NaN
# --------------------------------------------------------------------------------------
def _to_image_and_label(sample, transform, dataset_name="train"):
    """
    sample: dict coming from WebDataset
    we try jpg/jpeg/png
    we try cls / cls.txt / json["label"]
    everything becomes (C,H,W) float tensor and int64 label in [0,999]
    """
    # 1. image
    img = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if img is None:
        raise ValueError(f"[{dataset_name}] Sample has no image key: {list(sample.keys())}")

    if transform is not None:
        img = transform(img)

    # 2. label
    label = sample.get("cls", None)

    if label is None and "cls.txt" in sample:
        # often bytes -> decode -> int
        label = int(sample["cls.txt"].decode("utf-8"))
    elif label is None and "json" in sample:
        meta = sample["json"]
        # try common keys
        if "label" in meta:
            label = int(meta["label"])
        elif "class" in meta:
            label = int(meta["class"])
        else:
            raise ValueError(f"[{dataset_name}] json has no 'label'/'class': {meta}")

    if label is None:
        raise ValueError(f"[{dataset_name}] Sample has no label (cls/cls.txt/json)")

    # to tensor
    if not torch.is_tensor(label):
        label = torch.tensor(label, dtype=torch.long)
    else:
        label = label.long()

    # sometimes comes as shape (1,)
    if label.dim() != 0:
        label = label.view(-1)[0]

    # clamp to imagenet range
    label = torch.clamp(label, 0, 999)

    return img, label


# --------------------------------------------------------------------------------------
# helper to turn s3://bucket/prefix/... into list of urls
# but in SageMaker you often get local mounted paths already, so we just accept a
# directory path and expand *.tar
# --------------------------------------------------------------------------------------
def _expand_shards(root: str) -> List[str]:
    # root may be 's3://...' in some cases, but in training container it will usually be
    # something like /opt/ml/input/data/training
    if root.startswith("s3://"):
        # webdataset can read s3 urls directly if s3fs/https is available
        # just return pattern
        return [os.path.join(root, "*.tar")]
    # local dir → list all .tar files
    if os.path.isdir(root):
        files = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.endswith(".tar")
            ]
        )
        if not files:
            LOG.warning(f"[imagenet_dataset] No .tar files found under: {root}")
        return files
    # single file
    return [root]


# --------------------------------------------------------------------------------------
# main factory
# --------------------------------------------------------------------------------------
def get_imagenet_dataloaders(
    train_dir: str,
    val_dir: Optional[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    disable_distributed_splitting: bool = False,
    img_size: int = 224,
    epoch_size_train: Optional[int] = None,
    epoch_size_val: Optional[int] = None,
    resampled: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], int, int]:
    """
    Returns:
        train_loader, val_loader, train_batches_per_epoch, val_batches_per_epoch
    """

    # ---- cache key ----
    cache_key = (
        train_dir,
        val_dir,
        batch_size,
        num_workers,
        pin_memory,
        disable_distributed_splitting,
        img_size,
        epoch_size_train,
        epoch_size_val,
        resampled,
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    world_size = get_world_size()
    rank = get_rank()

    train_urls = _expand_shards(train_dir)
    val_urls = _expand_shards(val_dir) if val_dir else None

    train_transform = make_train_transform(img_size)
    val_transform = make_val_transform(img_size)

    # ----------------------------------------------------------------------------------
    # TRAIN DATASET
    # ----------------------------------------------------------------------------------
    # resampled=True is important for DDP so that each worker can loop independently
    train_data = (
        wds.WebDataset(
            train_urls,
            resampled=resampled,
            shardshuffle=True,
            nodesplitter=wds.split_by_node if not disable_distributed_splitting else None,
        )
        .shuffle(10000)
        .decode("pil")
        .map(lambda s: _to_image_and_label(s, train_transform, dataset_name="train"))
    )

    # we do batching inside WebDataset
    train_data = train_data.batched(batch_size, partial=False)

    # WebLoader is the recommended loader for WebDataset
    train_loader = wds.WebLoader(
        train_data,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # with_epoch controls how many batches we see as "one epoch"
    # if user didn't specify, estimate from #shards
    if epoch_size_train is None:
        # simple heuristic: 128k images / batch_size
        # but we can't count exactly because of resampled=True
        approx_images = 128000
        epoch_size_train = math.ceil(approx_images / batch_size)

    train_loader = train_loader.with_epoch(epoch_size_train)

    # ----------------------------------------------------------------------------------
    # VAL DATASET
    # ----------------------------------------------------------------------------------
    if val_urls:
        val_data = (
            wds.WebDataset(
                val_urls,
                resampled=False,  # val is finite
                shardshuffle=False,
                nodesplitter=wds.split_by_node if not disable_distributed_splitting else None,
            )
            .decode("pil")
            .map(lambda s: _to_image_and_label(s, val_transform, dataset_name="val"))
            .batched(batch_size, partial=False)
        )
        val_loader = wds.WebLoader(
            val_data,
            batch_size=None,
            num_workers=max(1, num_workers // 2),
            pin_memory=pin_memory,
        )

        if epoch_size_val is None:
            approx_val_images = 50000  # imagenet val
            epoch_size_val = math.ceil(approx_val_images / batch_size)

        val_loader = val_loader.with_epoch(epoch_size_val)
    else:
        val_loader = None
        epoch_size_val = 0

    # ----------------------------------------------------------------------------------
    # ONE-TIME SANITY LOG (rank 0 only)
    # ----------------------------------------------------------------------------------
    if rank == 0:
        try:
            it = iter(train_loader)
            x, y = next(it)
            LOG.info(
                f"[imagenet_dataset] sanity batch: x={tuple(x.shape)} {x.dtype}, "
                f"y={tuple(y.shape)} {y.dtype}, y_min={int(y.min())}, y_max={int(y.max())}"
            )
        except Exception as e:
            LOG.error(f"[imagenet_dataset] failed to read first batch: {e}")

    result = (train_loader, val_loader, epoch_size_train, epoch_size_val)
    _DATASET_CACHE[cache_key] = result
    return result
