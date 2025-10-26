#!/usr/bin/env python3
"""
SageMaker-Optimized WebDataset Converter: 
Converts ImageFolder structure from S3 input channels into sharded .tar files 
and writes them to the S3 output channel.
"""
import os
import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import io
import sys

# --- Configuration ---
SAMPLES_PER_SHARD = 5000 
# Note: Ensure the core functions (create_synset_to_index_mapping, convert_to_tar_shards) 
# are copied directly from the script provided in the previous turn.
# The main change is in the 'main' function to handle SageMaker paths.

# --- [PASTE Core Functions Here: create_synset_to_index_mapping, convert_to_tar_shards] ---
# ... (Assuming you paste the helper functions from the previous turn here) ...
# You must ensure the functions `create_synset_to_index_mapping` and 
# `convert_to_tar_shards` are present in this file.

def create_synset_to_index_mapping(base_dir: Path) -> Dict[str, int]:
    """
    Creates a mapping from ImageNet Synset ID (folder name) to integer class index (0-999).
    This is determined by the sorted order of the Synset ID folders, which is standard.
    """
    print(f"🔍 Creating Synset to Index Mapping from directories under: {base_dir}")
    
    # List all subdirectories (Synset IDs) and sort them alphabetically
    synset_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    
    if len(synset_dirs) != 1000:
        print(f"⚠️ Warning: Found {len(synset_dirs)} directories, expected 1000.")
        
    mapping = {synset_id: idx for idx, synset_id in enumerate(synset_dirs)}
    
    print(f"✅ Created mapping for {len(mapping)} classes (e.g., '{synset_dirs[0]}' -> 0)")
    
    return mapping

def convert_to_tar_shards(
    input_dir: Path, 
    output_dir: Path, 
    synset_to_index_map: Dict[str, int], 
    prefix: str
):
    """
    Converts the ImageFolder structure in input_dir into sharded .tar files in output_dir.
    """
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Starting Conversion for {prefix.upper()} Data ---")
    print(f"Source: {input_dir}")
    print(f"Target: {output_dir}")

    # Find all JPEG files and store them along with their Synset ID
    all_files = []
    for class_folder in input_dir.iterdir():
        if class_folder.is_dir():
            synset_id = class_folder.name
            if synset_id in synset_to_index_map:
                for img_path in class_folder.glob('*.JPEG'):
                    all_files.append((img_path, synset_id))

    if not all_files:
        print(f"❌ Error: No .JPEG files found in the structure: {input_dir}/SynsetID/*.JPEG. Check path.")
        return

    print(f"📦 Found {len(all_files):,} total samples to process.")

    # The conversion loop
    current_shard_index = 0
    current_shard_path = output_dir / f"{prefix}-shard-{current_shard_index:04d}.tar"
    tar_writer = tarfile.open(current_shard_path, "w")
    current_sample_count = 0

    try:
        for i, (img_path, synset_id) in enumerate(tqdm(all_files, desc=f"Converting {prefix}")):
            
            # --- 1. Prepare Data ---
            class_index = synset_to_index_map[synset_id]
            
            # Use the image file name (without extension) as the sample ID
            sample_id = img_path.stem
            
            # --- 2. Create Sample Files in Memory ---
            
            # Image Data (.jpg)
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            # Label Data (.cls) - Store the integer index as a byte string
            label_data = str(class_index).encode('utf-8')

            # --- 3. Write to Tar File ---
            
            # Write Image
            tar_info_img = tarfile.TarInfo(name=f"{sample_id}.jpg")
            tar_info_img.size = len(img_data)
            tar_writer.addfile(tar_info_img, io.BytesIO(img_data))
            
            # Write Label
            tar_info_cls = tarfile.TarInfo(name=f"{sample_id}.cls")
            tar_info_cls.size = len(label_data)
            tar_writer.addfile(tar_info_cls, io.BytesIO(label_data))
            
            current_sample_count += 1

            # --- 4. Sharding Logic ---
            if current_sample_count >= SAMPLES_PER_SHARD and (i < len(all_files) - 1):
                tar_writer.close()
                current_shard_index += 1
                current_sample_count = 0
                current_shard_path = output_dir / f"{prefix}-shard-{current_shard_index:04d}.tar"
                tar_writer = tarfile.open(current_shard_path, "w")
        
        # Close the last tar file
        tar_writer.close()
        print(f"🎉 Conversion complete! Created {current_shard_index + 1} shards in {output_dir}")

    except Exception as e:
        print(f"A critical error occurred: {e}")
        # Ensure the tar file is closed even on error
        if tar_writer and not tar_writer.closed:
            tar_writer.close()
        raise



# --- Main Function for SageMaker Processing Job ---
def main():
    parser = argparse.ArgumentParser(description="ImageNet WebDataset Converter for SageMaker Processing")
    
    # SageMaker Processing automatically mounts input channels to these paths
    parser.add_argument("--train-dir", type=str, default="/opt/ml/processing/input/train",
                        help="Local path to the mounted S3 train data folder.")
    parser.add_argument("--val-dir", type=str, default="/opt/ml/processing/input/val",
                        help="Local path to the mounted S3 validation data folder.")
    # SageMaker Processing automatically mounts the output channel to this path
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output/tars",
                        help="Local path where sharded .tar files will be written before uploading to S3.")
    
    args = parser.parse_args()
    
    train_input_dir = Path(args.train_dir)
    val_input_dir = Path(args.val_dir)
    output_root = Path(args.output_dir)
    
    # --- 1. Create Mapping ---
    # We rely on the 'train' folder structure to establish the 0-999 index order.
    # The synset folders are directly inside the 'train' input directory.
    synset_to_index_map = create_synset_to_index_mapping(train_input_dir)
    
    # Save the mapping for future reference (written to the output channel)
    mapping_path = output_root / 'synset_to_index_map.json'
    output_root.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(synset_to_index_map, f, indent=4)
    print(f"💾 Saved class mapping to {mapping_path}")

    # --- 2. Convert Training Data ---
    convert_to_tar_shards(
        input_dir=train_input_dir, 
        output_dir=output_root / 'train_tars', 
        synset_to_index_map=synset_to_index_map, 
        prefix='train'
    )

    # --- 3. Convert Validation Data ---
    convert_to_tar_shards(
        input_dir=val_input_dir, 
        output_dir=output_root / 'val_tars', 
        synset_to_index_map=synset_to_index_map, 
        prefix='val'
    )
    
    print("✅ WebDataset Conversion Pipeline Complete!")

if __name__ == "__main__":
    import io
    # Ensure the core functions (from the previous turn) are defined here before calling main()
    # For safety, you might need to manually ensure they are present.
    # ... (core functions) ...
    main()