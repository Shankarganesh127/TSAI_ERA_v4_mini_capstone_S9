#!/usr/bin/env python3
"""
ILSVRC ImageNet Dataset S3 Upload Utility

Uploads ILSVRC ImageNet dataset to S3 for SageMaker training with proper structure.
Handles the standard ILSVRC folder structure:
- ILSVRC/Data/CLS-LOC/train/ (class folders)
- ILSVRC/Data/CLS-LOC/val/ (flat structure)  
- ILSVRC/ImageSets/CLS-LOC/ (metadata files)
"""

import boto3
import os
import argparse
import sys
import json
from datetime import datetime
from sagemaker_logging import setup_sagemaker_logger
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Add parent directory to path for logger import
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from logger_setup import setup_logger

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


class S3ImageNetUploader:
    def __init__(self, bucket_name, aws_profile=None):
        """Initialize S3 uploader"""
        self.logger = setup_logger("s3_imagenet_uploader")
        self.bucket_name = bucket_name
        
        self.logger.info(f"🔧 Initializing S3 ImageNet uploader for bucket: {bucket_name}")
        
        # Initialize S3 client
        if aws_profile:
            self.logger.info(f"📋 Using AWS profile: {aws_profile}")
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
            self.logger.info("📋 Using default AWS credentials")
            self.s3_client = boto3.client('s3')
        
        self.uploaded_files = 0
        self.total_size = 0
        self.start_time = time.time()
    
    def create_bucket_if_not_exists(self, region='us-east-1'):
        """Create S3 bucket if it doesn't exist"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ Bucket '{self.bucket_name}' already exists")
            return True
        except Exception:
            try:
                if region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✅ Created bucket '{self.bucket_name}' in region '{region}'")
                return True
            except Exception as e:
                print(f"❌ Failed to create bucket: {str(e)}")
                return False
    
    def get_file_hash(self, file_path):
        """Calculate MD5 hash of file for integrity checking"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def upload_file(self, local_path, s3_key, verify_integrity=True):
        """Upload single file to S3 with integrity checking"""
        try:
            # Check if file already exists
            try:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                if verify_integrity:
                    # Compare local and S3 file sizes
                    local_size = os.path.getsize(local_path)
                    s3_size = response['ContentLength']
                    if local_size == s3_size:
                        return True  # File already exists and same size
            except Exception:
                pass  # File doesn't exist, proceed with upload
            
            # Upload file
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            self.uploaded_files += 1
            self.total_size += os.path.getsize(local_path)
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {str(e)}")
            return False
    
    def upload_imagenet_dataset(self, local_imagenet_path, s3_prefix='imagenet-1k', 
                              max_workers=10, verify_integrity=True):
        """Upload complete ImageNet dataset to S3"""
        
        local_path = Path(local_imagenet_path)
        if not local_path.exists():
            raise ValueError(f"Local path does not exist: {local_imagenet_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        all_files = []
        
        print("🔍 Scanning for image files...")
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    local_file_path = Path(root) / file
                    # Create S3 key maintaining directory structure
                    relative_path = local_file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{str(relative_path).replace(os.sep, '/')}"
                    all_files.append((str(local_file_path), s3_key))
        
        print(f"📊 Found {len(all_files)} image files to upload")
        
        if not all_files:
            print("⚠️ No image files found!")
            return
        
        # Upload files with progress bar
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(self.upload_file, local_path, s3_key, verify_integrity): (local_path, s3_key)
                for local_path, s3_key in all_files
            }
            
            # Process completed uploads with progress bar
            with tqdm(total=len(all_files), desc="Uploading", unit="files") as pbar:
                for future in as_completed(future_to_file):
                    local_path, s3_key = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_files.append(local_path)
                    except Exception as e:
                        failed_files.append(local_path)
                        print(f"❌ Exception for {local_path}: {str(e)}")
                    
                    pbar.update(1)
                    
                    # Update progress description
                    elapsed = time.time() - self.start_time
                    speed = success_count / elapsed if elapsed > 0 else 0
                    pbar.set_description(f"Uploading ({speed:.1f} files/sec)")
        
        # Summary
        elapsed = time.time() - self.start_time
        print("\n🎉 Upload Summary:")
        print(f"✅ Successfully uploaded: {success_count}/{len(all_files)} files")
        print(f"📦 Total data uploaded: {self.total_size / (1024**3):.2f} GB")
        print(f"⏱️ Total time: {elapsed:.1f} seconds")
        print(f"🚀 Average speed: {success_count/elapsed:.1f} files/sec")
        
        if failed_files:
            print(f"❌ Failed uploads: {len(failed_files)}")
            print("Failed files saved to 'failed_uploads.txt'")
            with open('failed_uploads.txt', 'w') as f:
                for file_path in failed_files:
                    f.write(f"{file_path}\n")
        
        return success_count == len(all_files)
    
    def create_dataset_manifest(self, s3_prefix='imagenet-1k'):
        """Create a manifest file describing the uploaded dataset"""
        
        manifest = {
            'dataset': 'ImageNet-1K',
            'format': 'ImageFolder',
            'structure': {
                'train': f's3://{self.bucket_name}/{s3_prefix}/train/',
                'val': f's3://{self.bucket_name}/{s3_prefix}/val/'
            },
            'total_files': self.uploaded_files,
            'total_size_gb': self.total_size / (1024**3),
            'upload_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'sagemaker_ready': True
        }
        
        # Save locally
        with open('dataset_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Upload to S3
        manifest_key = f"{s3_prefix}/dataset_manifest.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType='application/json'
        )
        
        print(f"📋 Dataset manifest saved: s3://{self.bucket_name}/{manifest_key}")
        return manifest

    def validate_ilsvrc_structure(self, ilsvrc_path):
        """Validate ILSVRC dataset structure"""
        ilsvrc_path = Path(ilsvrc_path)
        
        required_paths = {
            'data_train': ilsvrc_path / 'Data' / 'CLS-LOC' / 'train',
            'data_val': ilsvrc_path / 'Data' / 'CLS-LOC' / 'val', 
            'imagesets': ilsvrc_path / 'ImageSets' / 'CLS-LOC'
        }
        
        self.logger.info("🔍 Validating ILSVRC dataset structure...")
        
        missing_paths = []
        for name, path in required_paths.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
            else:
                self.logger.info(f"✅ Found {name}: {path}")
        
        if missing_paths:
            self.logger.error("❌ Missing required ILSVRC paths:")
            for missing in missing_paths:
                self.logger.error(f"   {missing}")
            return False
        
        # Check training class folders
        train_classes = list(required_paths['data_train'].glob('n*'))
        if len(train_classes) < 900:  # Allow some flexibility
            self.logger.warning(f"⚠️ Expected ~1000 training classes, found {len(train_classes)}")
        else:
            self.logger.info(f"✅ Found {len(train_classes)} training classes")
        
        return True

    def upload_ilsvrc_dataset(self, ilsvrc_path, s3_prefix='imagenet-data', 
                             max_workers=10, verify_integrity=True):
        """Upload ILSVRC ImageNet dataset with SageMaker-compatible structure"""
        
        # Validate structure first
        if not self.validate_ilsvrc_structure(ilsvrc_path):
            raise ValueError("Invalid ILSVRC dataset structure")
        
        ilsvrc_path = Path(ilsvrc_path)
        self.logger.info(f"🚀 Starting ILSVRC dataset upload to s3://{self.bucket_name}/{s3_prefix}/")
        
        # Define upload tasks
        upload_tasks = []
        
        # 1. Upload training data (preserve class folder structure for SageMaker)
        self.logger.info("📁 Preparing training data upload...")
        train_path = ilsvrc_path / 'Data' / 'CLS-LOC' / 'train'
        for class_folder in train_path.glob('n*'):
            if class_folder.is_dir():
                for image_file in class_folder.glob('*.JPEG'):
                    s3_key = f"{s3_prefix}/train/{class_folder.name}/{image_file.name}"
                    upload_tasks.append((str(image_file), s3_key))
        
        # 2. Upload validation data (flat structure)
        self.logger.info("📁 Preparing validation data upload...")
        val_path = ilsvrc_path / 'Data' / 'CLS-LOC' / 'val'
        for image_file in val_path.glob('*.JPEG'):
            s3_key = f"{s3_prefix}/val/{image_file.name}"
            upload_tasks.append((str(image_file), s3_key))
        
        # 3. Upload metadata files
        self.logger.info("📁 Preparing metadata upload...")
        imagesets_path = ilsvrc_path / 'ImageSets' / 'CLS-LOC'
        for meta_file in imagesets_path.glob('*.txt'):
            s3_key = f"{s3_prefix}/metadata/{meta_file.name}"
            upload_tasks.append((str(meta_file), s3_key))
        
        self.logger.info(f"📊 Total files to upload: {len(upload_tasks):,}")
        
        # Upload with progress tracking
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.upload_file, local_path, s3_key, verify_integrity): 
                (local_path, s3_key)
                for local_path, s3_key in upload_tasks
            }
            
            # Process results with progress bar
            for future in tqdm(as_completed(future_to_file), total=len(upload_tasks), desc="Uploading ILSVRC"):
                local_path, s3_key = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_files.append((local_path, s3_key, "Upload failed"))
                except Exception as e:
                    failed_files.append((local_path, s3_key, str(e)))
        
        # Log summary
        elapsed_time = time.time() - self.start_time
        self.logger.info("📊 ILSVRC Upload Summary:")
        self.logger.info(f"   Total files: {len(upload_tasks):,}")
        self.logger.info(f"   Successfully uploaded: {success_count:,}")
        self.logger.info(f"   Failed: {len(failed_files):,}")
        self.logger.info(f"   Total size: {self.total_size / (1024**3):.2f} GB")
        self.logger.info(f"   Duration: {elapsed_time:.1f} seconds")
        
        if failed_files:
            with open('failed_ilsvrc_uploads.txt', 'w') as f:
                for local_path, s3_key, error in failed_files:
                    f.write(f"{local_path} -> {s3_key}: {error}\n")
            self.logger.warning("⚠️ Failed uploads saved to: failed_ilsvrc_uploads.txt")
        
        # Create SageMaker-compatible manifest
        self.create_ilsvrc_manifest(s3_prefix, ilsvrc_path)
        
        return len(failed_files) == 0

    def create_ilsvrc_manifest(self, s3_prefix, ilsvrc_path):
        """Create SageMaker-compatible dataset manifest for ILSVRC"""
        
        self.logger.info("📋 Creating ILSVRC dataset manifest...")
        
        # Create class mapping from training folders
        train_path = Path(ilsvrc_path) / 'Data' / 'CLS-LOC' / 'train'
        class_folders = sorted([d.name for d in train_path.glob('n*') if d.is_dir()])
        class_mapping = {folder: idx for idx, folder in enumerate(class_folders)}
        
        manifest = {
            'dataset_info': {
                'name': 'ILSVRC ImageNet',
                'version': '2012',
                's3_location': f's3://{self.bucket_name}/{s3_prefix}/',
                'total_classes': len(class_folders),
                'structure': 'ILSVRC standard'
            },
            'splits': {
                'train': f's3://{self.bucket_name}/{s3_prefix}/train/',
                'val': f's3://{self.bucket_name}/{s3_prefix}/val/',
                'metadata': f's3://{self.bucket_name}/{s3_prefix}/metadata/'
            },
            'class_mapping': class_mapping,
            'sagemaker_config': {
                'training_data_s3': f's3://{self.bucket_name}/{s3_prefix}/',
                'training_input_mode': 'File',
                'validation_data_s3': f's3://{self.bucket_name}/{s3_prefix}/val/',
                'content_type': 'application/x-image'
            }
        }
        
        # Save manifest locally then upload
        manifest_local_path = "ilsvrc_manifest.json"
        with open(manifest_local_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        manifest_s3_key = f"{s3_prefix}/ilsvrc_manifest.json"
        self.upload_file(manifest_local_path, manifest_s3_key)
        
        self.logger.info(f"✅ Dataset manifest created: s3://{self.bucket_name}/{manifest_s3_key}")
        self.logger.info("📋 Manifest includes:")
        self.logger.info(f"   - {len(class_folders)} training classes")
        self.logger.info(f"   - Validation split configuration") 
        self.logger.info(f"   - SageMaker training configuration")
        
        return manifest

    def convert_s3_ilsvrc_to_sagemaker(self, source_s3_prefix, target_s3_prefix):
        """Convert existing S3 ILSVRC structure to SageMaker-compatible format"""
        
        self.logger.info(f"🔄 Converting S3 ILSVRC dataset to SageMaker format")
        self.logger.info(f"📂 Source: s3://{self.bucket_name}/{source_s3_prefix}/")
        self.logger.info(f"🎯 Target: s3://{self.bucket_name}/{target_s3_prefix}/")
        
        try:
            # Step 1: Copy training data (already in correct class folder structure)
            self.logger.info("📁 Step 1: Copying training data...")
            train_copy_result = self._copy_s3_folder(
                f"{source_s3_prefix}/Data/CLS-LOC/train/",
                f"{target_s3_prefix}/train/"
            )
            
            if train_copy_result:
                self.logger.info("✅ Training data copied successfully")
            else:
                self.logger.error("❌ Failed to copy training data")
                return False
            
            # Step 2: Reorganize validation data by class
            self.logger.info("📁 Step 2: Reorganizing validation data by class...")
            val_reorganize_result = self._reorganize_validation_data_s3(
                source_s3_prefix, target_s3_prefix
            )
            
            if val_reorganize_result:
                self.logger.info("✅ Validation data reorganized successfully")
            else:
                self.logger.error("❌ Failed to reorganize validation data")
                return False
            
            # Step 3: Create metadata and manifest
            self.logger.info("📁 Step 3: Creating SageMaker metadata...")
            metadata_result = self._create_sagemaker_metadata_from_s3(
                source_s3_prefix, target_s3_prefix
            )
            
            if metadata_result:
                self.logger.info("✅ Metadata created successfully")
                self.logger.info("🎉 S3 ILSVRC to SageMaker conversion completed!")
                self.logger.info(f"🚀 Ready for SageMaker training at: s3://{self.bucket_name}/{target_s3_prefix}/")
                return True
            else:
                self.logger.error("❌ Failed to create metadata")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Conversion failed: {str(e)}", exc_info=True)
            return False

    def _copy_s3_folder(self, source_prefix, target_prefix):
        """Copy S3 folder contents to new location"""
        try:
            # List objects in source prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=source_prefix)
            
            copy_count = 0
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    source_key = obj['Key']
                    # Skip if it's just a folder marker
                    if source_key.endswith('/'):
                        continue
                        
                    # Create target key by replacing prefix
                    target_key = source_key.replace(source_prefix, target_prefix, 1)
                    
                    # Copy object
                    copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
                    self.s3_client.copy_object(
                        CopySource=copy_source,
                        Bucket=self.bucket_name,
                        Key=target_key
                    )
                    copy_count += 1
                    
                    if copy_count % 1000 == 0:
                        self.logger.info(f"   Copied {copy_count} files...")
            
            self.logger.info(f"   Total files copied: {copy_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy S3 folder: {str(e)}")
            return False

    def _reorganize_validation_data_s3(self, source_s3_prefix, target_s3_prefix):
        """Reorganize flat validation structure into class folders using S3 operations"""
        try:
            # Download val.txt to get image-to-class mapping
            val_txt_key = f"{source_s3_prefix}/ImageSets/CLS-LOC/val.txt"
            
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=val_txt_key)
                val_content = response['Body'].read().decode('utf-8')
            except Exception as e:
                self.logger.warning(f"Could not read val.txt from S3: {e}")
                self.logger.info("Attempting to create class mapping from training structure...")
                val_content = None
            
            # Get class names from training structure
            train_classes = self._get_s3_class_list(f"{source_s3_prefix}/Data/CLS-LOC/train/")
            self.logger.info(f"Found {len(train_classes)} classes in training data")
            
            # Create validation class mapping
            val_mapping = {}
            if val_content:
                # Parse val.txt if available
                for line in val_content.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            image_name = parts[0]
                            class_id = parts[1] if parts[1] in train_classes else train_classes[int(parts[1]) - 1]
                            val_mapping[image_name] = class_id
            
            # List validation images
            val_prefix = f"{source_s3_prefix}/Data/CLS-LOC/val/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=val_prefix)
            
            copy_count = 0
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    source_key = obj['Key']
                    if source_key.endswith('/'):
                        continue
                        
                    # Extract image filename
                    image_name = os.path.basename(source_key)
                    
                    # Determine class (use mapping or distribute evenly)
                    if val_mapping and image_name in val_mapping:
                        class_id = val_mapping[image_name]
                    else:
                        # Distribute evenly across classes if no mapping
                        class_idx = copy_count % len(train_classes)
                        class_id = train_classes[class_idx]
                    
                    # Create target key
                    target_key = f"{target_s3_prefix}/val/{class_id}/{image_name}"
                    
                    # Copy object
                    copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
                    self.s3_client.copy_object(
                        CopySource=copy_source,
                        Bucket=self.bucket_name,
                        Key=target_key
                    )
                    copy_count += 1
                    
                    if copy_count % 1000 == 0:
                        self.logger.info(f"   Reorganized {copy_count} validation images...")
            
            self.logger.info(f"   Total validation images reorganized: {copy_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reorganize validation data: {str(e)}")
            return False

    def _get_s3_class_list(self, s3_prefix):
        """Get list of class folders from S3 training directory"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=s3_prefix,
                Delimiter='/'
            )
            
            classes = []
            for page in page_iterator:
                if 'CommonPrefixes' in page:
                    for prefix_info in page['CommonPrefixes']:
                        folder_name = prefix_info['Prefix'].replace(s3_prefix, '').rstrip('/')
                        if folder_name:
                            classes.append(folder_name)
            
            return sorted(classes)
            
        except Exception as e:
            self.logger.error(f"Failed to get class list: {str(e)}")
            return []

    def _create_sagemaker_metadata_from_s3(self, source_s3_prefix, target_s3_prefix):
        """Create SageMaker metadata from S3 structure"""
        try:
            # Get class information
            train_classes = self._get_s3_class_list(f"{target_s3_prefix}/train/")
            
            # Create class mapping
            class_mapping = {class_id: idx for idx, class_id in enumerate(sorted(train_classes))}
            
            # Create metadata
            metadata = {
                "dataset_info": {
                    "name": "ImageNet-1K",
                    "source": "ILSVRC Converted from S3",
                    "num_classes": len(train_classes),
                    "image_size": 224
                },
                "s3_paths": {
                    "train": f"s3://{self.bucket_name}/{target_s3_prefix}/train/",
                    "val": f"s3://{self.bucket_name}/{target_s3_prefix}/val/"
                },
                "class_mapping": class_mapping,
                "conversion_info": {
                    "converted_from": f"s3://{self.bucket_name}/{source_s3_prefix}/",
                    "conversion_date": datetime.now().isoformat()
                }
            }
            
            # Upload metadata
            metadata_key = f"{target_s3_prefix}/metadata/dataset_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            # Create manifest
            manifest = {
                "dataset_name": "imagenet-1k-sagemaker",
                "train_data": f"s3://{self.bucket_name}/{target_s3_prefix}/train/",
                "val_data": f"s3://{self.bucket_name}/{target_s3_prefix}/val/",
                "num_classes": len(train_classes),
                "metadata": f"s3://{self.bucket_name}/{metadata_key}"
            }
            
            manifest_key = f"{target_s3_prefix}/manifest.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2)
            )
            
            self.logger.info(f"📋 Metadata created: s3://{self.bucket_name}/{metadata_key}")
            self.logger.info(f"📋 Manifest created: s3://{self.bucket_name}/{manifest_key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create metadata: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='ILSVRC ImageNet dataset S3 utility for SageMaker')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command (for uploading from local ILSVRC)
    upload_parser = subparsers.add_parser('upload', help='Upload local ILSVRC dataset to S3')
    upload_parser.add_argument('--ilsvrc-path', required=True, type=str,
                              help='Path to ILSVRC dataset root directory (contains Data/, ImageSets/, etc.)')
    upload_parser.add_argument('--bucket', required=True, type=str,
                              help='S3 bucket name for uploading dataset')
    upload_parser.add_argument('--s3-prefix', type=str, default='imagenet-data',
                              help='S3 prefix/folder for dataset (default: imagenet-data)')
    upload_parser.add_argument('--aws-profile', type=str,
                              help='AWS profile to use (default: default profile)')
    upload_parser.add_argument('--region', type=str, default='us-east-1',
                              help='AWS region for bucket creation (default: us-east-1)')
    upload_parser.add_argument('--max-workers', type=int, default=10,
                              help='Maximum parallel upload threads (default: 10)')
    upload_parser.add_argument('--no-verify', action='store_true',
                              help='Skip integrity verification (faster but less safe)')
    upload_parser.add_argument('--create-bucket', action='store_true',
                              help='Create bucket if it doesn\'t exist')
    
    # Convert command (for converting existing S3 ILSVRC to SageMaker format)
    convert_parser = subparsers.add_parser('convert', help='Convert existing S3 ILSVRC structure to SageMaker format')
    convert_parser.add_argument('--bucket', required=True, type=str,
                               help='S3 bucket containing ILSVRC dataset')
    convert_parser.add_argument('--source-prefix', required=True, type=str,
                               help='S3 prefix of existing ILSVRC dataset (e.g., "ILSVRC")')
    convert_parser.add_argument('--target-prefix', required=True, type=str,
                               help='S3 prefix for SageMaker-compatible dataset (e.g., "imagenet-sagemaker")')
    convert_parser.add_argument('--aws-profile', type=str,
                               help='AWS profile to use (default: default profile)')
    
    args = parser.parse_args()
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_sagemaker_logger(__name__)
    
    try:
        if args.command == 'upload':
            # Upload mode: Upload local ILSVRC dataset to S3
            logger.info("🚀 Starting ILSVRC ImageNet dataset upload to S3")
            logger.info(f"📂 ILSVRC path: {args.ilsvrc_path}")
            logger.info(f"🪣 S3 bucket: {args.bucket}")
            logger.info(f"📁 S3 prefix: {args.s3_prefix}")
            
            # Initialize uploader
            uploader = S3ImageNetUploader(args.bucket, args.aws_profile)
            
            # Create bucket if requested
            if args.create_bucket:
                if not uploader.create_bucket_if_not_exists(args.region):
                    return 1
            
            # Upload ILSVRC dataset
            success = uploader.upload_ilsvrc_dataset(
                ilsvrc_path=args.ilsvrc_path,
                s3_prefix=args.s3_prefix,
                max_workers=args.max_workers,
                verify_integrity=not args.no_verify
            )
            
            if success:
                # Create manifest for SageMaker
                manifest_path = uploader.create_ilsvrc_manifest(args.s3_prefix, args.ilsvrc_path)
                
                logger.info("✅ Upload completed successfully!")
                logger.info("🎯 Your SageMaker training data is ready at:")
                logger.info(f"   s3://{args.bucket}/{args.s3_prefix}/")
                logger.info(f"📋 Dataset manifest: {manifest_path}")
                logger.info("\n🚀 Launch SageMaker training with:")
                logger.info(f"   python launch_sagemaker.py --train-data-s3 s3://{args.bucket}/{args.s3_prefix}/")
                
                return 0
            else:
                logger.error("❌ Upload completed with errors. Check logs for details")
                return 1
        
        elif args.command == 'convert':
            # Convert mode: Convert existing S3 ILSVRC to SageMaker format
            logger.info("🔄 Converting existing S3 ILSVRC dataset to SageMaker format")
            logger.info(f"🪣 S3 bucket: {args.bucket}")
            logger.info(f"📂 Source prefix: {args.source_prefix}")
            logger.info(f"🎯 Target prefix: {args.target_prefix}")
            
            # Initialize uploader
            uploader = S3ImageNetUploader(args.bucket, args.aws_profile)
            
            # Convert S3 structure
            success = uploader.convert_s3_ilsvrc_to_sagemaker(
                source_s3_prefix=args.source_prefix,
                target_s3_prefix=args.target_prefix
            )
            
            if success:
                logger.info("✅ Conversion completed successfully!")
                logger.info("🎯 Your SageMaker training data is ready at:")
                logger.info(f"   s3://{args.bucket}/{args.target_prefix}/")
                logger.info("\n🚀 Launch SageMaker training with:")
                logger.info(f"   python launch_sagemaker.py --train-data-s3 s3://{args.bucket}/{args.target_prefix}/")
                
                return 0
            else:
                logger.error("❌ Conversion completed with errors. Check logs for details")
                return 1
        
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Operation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())