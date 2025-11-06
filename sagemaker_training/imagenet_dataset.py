import os
import math
import logging
from typing import Tuple, Optional, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import psutil

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

def _build_train_dataset(urls, transform):
    # good shuffle mix; resampled keeps infinite stream for distributed
    dataset = (
        wds.WebDataset(urls, resampled=True)
          .shuffle(10000)              # large buffer for class mixing
          .decode("pil")
          .to_tuple("jpg", "cls")      # DO NOT .batched here
          .map(lambda x, y: (transform(x), int(y)))
    )
    return dataset

def _build_val_dataset(urls, transform):
    dataset = (
        wds.WebDataset(urls)
          .decode("pil")
          .to_tuple("jpg", "cls")
          .map(lambda x, y: (transform(x), int(y)))
    )
    return dataset

def make_train_transform(img_size=224, normalize=True):
    t = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # -> [0,1]
    ]
    if normalize:
        t.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225)))
    return transforms.Compose(t)

def make_val_transform(img_size=224, normalize=True):
    t = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize:
        t.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225)))
    return transforms.Compose(t)


# --------------------------------------------------------------------------------------
# core transforms (you can tweak to match your original file)
# --------------------------------------------------------------------------------------
#def make_train_transform(img_size: int = 224, normalize: bool = True):
#    # Full advanced augmentations
#    train_transform_list = [
#        # Scale-aware random cropping (8%-100% of image, aspect ratio 3:4 to 4:3)
#        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
#        
#        # Horizontal flip with 50% probability
#        transforms.RandomHorizontalFlip(p=0.5),
#        
#        ## Advanced color augmentations for lighting/illumination robustness
#        #transforms.ColorJitter(
#        #    brightness=0.4,  # ±40% brightness change
#        #    contrast=0.4,    # ±40% contrast change  
#        #    saturation=0.4,  # ±40% saturation change
#        #    hue=0.1          # ±10% hue change
#        #),
#        
#        ## Geometric augmentations for spatial robustness
#        #transforms.RandomAffine(
#        #    degrees=0,       # No rotation to preserve object orientation
#        #    translate=(0.1, 0.1),  # ±10% translation
#        #    scale=(0.9, 1.1),      # ±10% scaling
#        #    shear=0.1,             # ±10% shearing
#        #    fill=0
#        #),
#        
#        ## Gaussian blur for noise and focus robustness
#        #transforms.GaussianBlur(
#        #    kernel_size=(3, 3), 
#        #    sigma=(0.1, 2.0)     # Blur strength range
#        #),
#        
#        transforms.ToTensor(),
#        
#        ## Random Erasing (Cutout) for occlusion robustness - applied after ToTensor on [0,1] range
#        #transforms.RandomErasing(
#        #    p=0.25,           # 25% probability
#        #    scale=(0.02, 0.33),  # Erase 2-33% of image area
#        #    ratio=(0.3, 3.3),    # Aspect ratio range
#        #    value='random'       # Fill with random pixel values in [0,1] range
#        #),
#        ]
#    
#    if normalize:
#        train_transform_list.append(
#            transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225))
#        )
#
#    return transforms.Compose(train_transform_list)

#def make_train_transform(img_size: int = 224, normalize: bool = True):
#    transform_list = [
#        transforms.RandomResizedCrop(img_size),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#    ]
#    if normalize:
#        transform_list.append(
#            transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225))
#        )
#    return transforms.Compose(transform_list)


#def make_val_transform(img_size: int = 224, normalize: bool = True):
#    transform_list = [
#        transforms.Resize(256),
#        transforms.CenterCrop(img_size),
#        transforms.ToTensor(),
#    ]
#    if normalize:
#        transform_list.append(
#            transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225))
#        )
#    return transforms.Compose(transform_list)



# --------------------------------------------------------------------------------------
# defensive sample → (image, label) conversion
# this is the important part for your LR Finder NaN
# --------------------------------------------------------------------------------------
def _to_image_and_label(sample, transform, dataset_name="train"):
    """
    Converts a WebDataset sample dict to (image_tensor, label_tensor).
    Ensures image ∈ [0,1] before normalization and label ∈ [0,999].
    """
    import io
    from PIL import Image
    import torch

    # --- 1. Decode image ---
    img_bytes = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if img_bytes is None:
        raise ValueError(f"[{dataset_name}] Missing image key: {list(sample.keys())}")

    if isinstance(img_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        img = img_bytes  # already a PIL image from WebDataset decode("pil")

    # --- 2. Apply transform ---
    if transform is not None:
        img = transform(img)

    # --- 3. Decode label ---
    label = sample.get("cls")
    if label is None and "cls.txt" in sample:
        label = int(sample["cls.txt"].decode("utf-8").strip())
    elif label is None and "json" in sample:
        meta = sample["json"]
        if isinstance(meta, (bytes, bytearray)):
            import json
            meta = json.loads(meta.decode("utf-8"))
        label = int(meta.get("label", meta.get("class", -1)))

    if label is None or not (0 <= int(label) < 1000):
        raise ValueError(f"[{dataset_name}] Invalid label {label}")

    label = torch.tensor(int(label), dtype=torch.long)

    # Optional: occasional sanity debug
    import random
    if isinstance(img, torch.Tensor) and random.random() < 1e-4:
        print(f"[DEBUG-after-transform][rank={get_rank()}] "
              f"min={img.min():.4f} max={img.max():.4f}")

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
        return os.path.join(root, "*.tar")
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
    LOG.info(f"[imagenet_dataset] Using single shard file: {root}")
    return root


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
    normalize: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    batched: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], int, int]:
    """
    Returns:
        train_loader, val_loader, train_batches_per_epoch, val_batches_per_epoch
    """

    # --- NEW: Optimize PyTorch thread usage ---
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if torch.get_num_threads() <= 1:
        total_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 8
        world = dist.get_world_size() if dist.is_initialized() else 1
        threads = max(2, total_cpus // world)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(min(4, max(1, threads // 2)))

    print(f"[THREADS] Stage=training using preconfigured threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")

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
        normalize,
        batched,
        prefetch_factor, 
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    world_size = get_world_size()
    rank = get_rank()

    train_urls = _expand_shards(train_dir)
    val_urls = _expand_shards(val_dir) if val_dir else None

    train_transform = make_train_transform(img_size, normalize=normalize)
    val_transform = make_val_transform(img_size, normalize=normalize)
    
    train_ds = _build_train_dataset(train_urls, train_transform)
    val_ds   = _build_val_dataset(val_urls,   val_transform)

    # ----------------------------------------------------------------------------------
    # TRAIN DATASET
    # ----------------------------------------------------------------------------------
    # pick a better splitter for class-grouped shards
    splitter = None
    if not disable_distributed_splitting:
        # worker-level split gives better mixing than node-level
        splitter = wds.split_by_worker

    LOG.info(f"[imagenet_dataset] world_size={world_size} rank={rank} " f"disable_distributed_splitting={disable_distributed_splitting} splitter={splitter}")
    LOG.info(f"[imagenet_dataset] train_urls={train_urls}")
    LOG.info(f"[imagenet_dataset] val_urls={val_urls}")
    
    ## IMPORTANT: for class-heavy tars, resampled=True will keep giving 1-class bursts,
    ## so force resampled=False here.
    #train_data = (
    #    wds.WebDataset(
    #        train_urls,
    #        resampled=False,          # ← ← key change
    #        shardshuffle=True,
    #        nodesplitter=splitter,
    #    )
    #    # big shuffle BEFORE decode/map
    #    .shuffle(10000)
    #    .decode("pil")
    #    .map(lambda s: _to_image_and_label(s, train_transform, dataset_name="train"))
    #    #.batched(batch_size, partial=False)
    #)

    #if batched:
    #    # we do batching inside WebDataset
    #    train_data = train_data.batched(batch_size, partial=False)
    
    # Respect pipeline-provided num_workers if passed
    if num_workers is not None:
        train_workers = max(1, int(num_workers))
        val_workers = max(1, int(num_workers // 2))
    else:
        cpu_per_rank = (psutil.cpu_count(logical=True) or 8) // max(1, dist.get_world_size())
        train_workers = min(4, max(2, cpu_per_rank // 2))
        val_workers = max(2, train_workers // 2)

    ## WebLoader is the recommended loader for WebDataset
    #train_loader = wds.WebLoader(
    #    train_data,
    #    batch_size=None,
    #    num_workers=train_workers,
    #    pin_memory=pin_memory,
    #    persistent_workers=persistent_workers,
    #    prefetch_factor=prefetch_factor,
    #)

    # with_epoch controls how many batches we see as "one epoch"
    # if user didn't specify, estimate from #shards
    #if epoch_size_train is None:
    #    # simple heuristic: 128k images / batch_size
    #    # but we can't count exactly because of resampled=True
    #    approx_images = 128000
    #    epoch_size_train = math.ceil(approx_images / batch_size)
    #    
    #    # 👇 NEW: if we're in DDP and we're actually splitting by node,
    #    # then EACH rank should do only its share.
    #    if dist.is_initialized() and not disable_distributed_splitting:
    #        world = dist.get_world_size()
    #        epoch_size_train = max(1, epoch_size_train // world)

    #train_loader = train_loader.with_epoch(epoch_size_train)

    # ----------------------------------------------------------------------------------
    # VAL DATASET
    # ----------------------------------------------------------------------------------
    #if val_urls:
    #    val_data = (
    #        wds.WebDataset(
    #            val_urls,
    #            resampled=False,  # val is finite
    #            shardshuffle=False,
    #            nodesplitter=wds.split_by_node if not disable_distributed_splitting else None,
    #            empty_check=False,
    #        )
    #        .decode("pil")
    #        .map(lambda s: _to_image_and_label(s, val_transform, dataset_name="val"))
    #        #.batched(batch_size, partial=False)
    #    )
    #    
    #    if batched:
    #        # we do batching inside WebDataset
    #        val_data = val_data.batched(batch_size, partial=False)
    #    
    #    val_loader = wds.WebLoader(
    #        val_data,
    #        batch_size=None,
    #        num_workers=val_workers,
    #        pin_memory=pin_memory,
    #        persistent_workers=persistent_workers,
    #        prefetch_factor=prefetch_factor,
    #    )

    #    if epoch_size_val is None:
    #        approx_val_images = 50000  # imagenet val
    #        epoch_size_val = math.ceil(approx_val_images / batch_size)
    #        if dist.is_initialized() and not disable_distributed_splitting:
    #            world = dist.get_world_size()
    #            epoch_size_val = max(1, epoch_size_val // world)

    #    val_loader = val_loader.with_epoch(epoch_size_val)
    #else:
    #    val_loader = None
    #    epoch_size_val = 0
        
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=train_workers,
                              pin_memory=pin_memory,
                              persistent_workers=True,
                              drop_last=True)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=val_workers,
                            pin_memory=pin_memory,
                            persistent_workers=True)

    # ----------------------------------------------------------------------------------
    # ONE-TIME SANITY LOG (rank 0 only)
    # ----------------------------------------------------------------------------------
    if rank == 0:
        try:
            # Pick what to sanity check based on batched mode
            if batched:
                it = iter(train_loader)
                for i in range(3):
                    x, y = next(it)
                    LOG.info(
                        f"[imagenet_dataset] sanity batch {i}: "
                        f"x={tuple(x.shape)} {x.dtype}, "
                        f"y={tuple(y.shape)} {y.dtype}, "
                        f"y_min={int(y.min())}, y_max={int(y.max())}"
                    )
            else:
                it = iter(train_ds)
                for i in range(3):
                    x, y = next(it)
                    LOG.info(
                        f"[imagenet_dataset] sanity sample {i}: "
                        f"x={tuple(x.shape)} {x.dtype}, "
                        f"y={y.item() if torch.is_tensor(y) else y}"
                    )
        except Exception as e:
            LOG.error(f"[imagenet_dataset] failed to read first samples: {e}")

    # ----------------------------------------------------------------------------------
    # RETURN dual mode
    # ----------------------------------------------------------------------------------
    if not batched:
        # For LR finder / BS finder / WD search → return raw datasets
        return train_ds, val_ds, 0, 0
    else:
        result = (train_loader, val_loader, epoch_size_train, epoch_size_val)
        _DATASET_CACHE[cache_key] = result
        return result
