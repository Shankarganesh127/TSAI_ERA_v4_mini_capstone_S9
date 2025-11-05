import boto3
import json
import webdataset as wds
from collections import Counter

def list_s3_tars(bucket, prefix):
    """List all .tar files in a given S3 prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    tars = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar"):
                tars.append(f"s3://{bucket}/{obj['Key']}")
    return sorted(tars)

def get_label_ids_from_wds(tar_urls, n_samples=5000):
    """Read sample of labels from a list of tar shards on S3."""
    dataset = (
        wds.WebDataset(tar_urls, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
    )
    label_counter = Counter()
    for i, (_, y) in enumerate(dataset):
        label_counter[int(y)] += 1
        if i >= n_samples:
            break
    return set(label_counter.keys()), label_counter

def check_mapping(train_s3_prefix, val_s3_prefix, n_samples=5000):
    """Check consistency between train and val WebDataset label mappings."""
    # Parse S3 bucket and prefix
    def parse_s3_uri(uri):
        assert uri.startswith("s3://"), f"Not an S3 URI: {uri}"
        path = uri[5:]
        bucket, prefix = path.split("/", 1)
        return bucket, prefix

    train_bucket, train_prefix = parse_s3_uri(train_s3_prefix)
    val_bucket, val_prefix = parse_s3_uri(val_s3_prefix)

    train_tars = list_s3_tars(train_bucket, train_prefix)
    val_tars = list_s3_tars(val_bucket, val_prefix)

    print(f"🔍 Found {len(train_tars)} train shards, {len(val_tars)} val shards")

    # Sample from shards
    train_labels, train_counts = get_label_ids_from_wds(train_tars, n_samples)
    val_labels, val_counts = get_label_ids_from_wds(val_tars, n_samples)

    print(f"\n✅ Train labels: {len(train_labels)} unique")
    print(f"✅ Val labels:   {len(val_labels)} unique")

    overlap = train_labels & val_labels
    missing_in_val = train_labels - val_labels
    missing_in_train = val_labels - train_labels

    print(f"\n🧩 Overlap: {len(overlap)} classes")
    print(f"🚫 Missing in val: {len(missing_in_val)}")
    print(f"🚫 Missing in train: {len(missing_in_train)}")

    if len(overlap) < 900:
        print("\n❌ WARNING: Likely class-index mismatch between train and val!\n")
    else:
        print("\n✅ Class mappings look consistent.\n")

    # Optional dump
    with open("label_stats_train.json", "w") as f:
        json.dump(train_counts, f, indent=2)
    with open("label_stats_val.json", "w") as f:
        json.dump(val_counts, f, indent=2)
    print("📄 Saved label frequency stats locally.")

if __name__ == "__main__":
    # 👇 replace with your S3 prefixes
    train_prefix = "s3://tsai-era-v4-mini-capstone/webdataset_shards/train_tars/"
    val_prefix   = "s3://tsai-era-v4-mini-capstone/webdataset_shards/val_tars/"

    check_mapping(train_prefix, val_prefix, n_samples=10000)
