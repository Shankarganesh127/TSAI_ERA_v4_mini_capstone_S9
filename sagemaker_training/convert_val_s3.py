"""
convert_val_s3.py
Reorganize ImageNet validation dataset from flat folder to class-based folder structure.
Can run as a SageMaker Processing step.


s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/val
s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/LOC_val_solution.csv
s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val

Usage example (inside SageMaker):
python convert_val_s3.py \
    --input_s3 s3://my-bucket/imagenet/val_flat \
    --labels_s3 s3://my-bucket/imagenet/val_labels.csv \
    --output_s3 s3://my-bucket/imagenet/val_reorganized
"""

import os
import boto3
import csv
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def download_s3_folder(bucket, prefix, local_dir):
    """Download all objects under a prefix from S3 to a local directory."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

def upload_s3_folder(local_dir, bucket, prefix):
    """Upload all files in a local directory to S3 under a prefix."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, rel_path)
            s3.upload_file(local_path, bucket, s3_key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_s3", required=True, help="S3 path to flat val images (e.g. s3://bucket/val)")
    parser.add_argument("--labels_s3", required=True, help="S3 path to CSV mapping file (image, synset)")
    parser.add_argument("--output_s3", required=True, help="S3 output path to upload reorganized val folder")
    parser.add_argument("--workdir", default="/opt/ml/processing", help="Local working directory")
    args = parser.parse_args()

    s3 = boto3.resource("s3")

    # Parse S3 URIs
    def split_s3_uri(s3_uri):
        assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    input_bucket, input_prefix = split_s3_uri(args.input_s3)
    labels_bucket, labels_key = split_s3_uri(args.labels_s3)
    output_bucket, output_prefix = split_s3_uri(args.output_s3)

    local_input = Path(args.workdir) / "val_flat"
    local_labels = Path(args.workdir) / "val_labels.csv"
    local_output = Path(args.workdir) / "val_reorg"
    os.makedirs(local_input, exist_ok=True)
    os.makedirs(local_output, exist_ok=True)

    print(f"Downloading images from s3://{input_bucket}/{input_prefix}...")
    download_s3_folder(input_bucket, input_prefix, str(local_input))

    print(f"Downloading label file from s3://{labels_bucket}/{labels_key}...")
    s3.Bucket(labels_bucket).download_file(labels_key, str(local_labels))

    print("Reorganizing validation dataset by class...")
    with open(local_labels, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in tqdm(reader):
            if len(row) < 2:
                continue
            img_name, synset = row[0].strip(), row[1].strip().split()[0]
            src_path = local_input / f"{img_name}.JPEG"
            if not src_path.exists():
                continue
            dst_dir = local_output / synset
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_dir / f"{img_name}.JPEG"))

    print("Uploading reorganized validation dataset to S3...")
    upload_s3_folder(str(local_output), output_bucket, output_prefix)

    print(f"✅ Done! Uploaded reorganized validation set to s3://{output_bucket}/{output_prefix}")

if __name__ == "__main__":
    main()
