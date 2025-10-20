#!/usr/bin/env python3
"""
ImageNet Dataset S3 Upload Utility
Uploads ImageNet dataset to S3 for SageMaker training with proper structure and progress tracking.
"""

import boto3
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class S3ImageNetUploader:
    def __init__(self, bucket_name, aws_profile=None):
        """Initialize S3 uploader"""
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
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


def main():
    parser = argparse.ArgumentParser(description='Upload ImageNet dataset to S3 for SageMaker')
    
    # Required arguments
    parser.add_argument('--local-data', required=True, type=str,
                       help='Path to local ImageNet dataset directory')
    parser.add_argument('--bucket', required=True, type=str,
                       help='S3 bucket name for uploading dataset')
    
    # Optional arguments
    parser.add_argument('--s3-prefix', type=str, default='imagenet-1k',
                       help='S3 prefix/folder for dataset (default: imagenet-1k)')
    parser.add_argument('--aws-profile', type=str,
                       help='AWS profile to use (default: default profile)')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region for bucket creation (default: us-east-1)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum parallel upload threads (default: 10)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip integrity verification (faster but less safe)')
    parser.add_argument('--create-bucket', action='store_true',
                       help='Create bucket if it doesn\'t exist')
    
    args = parser.parse_args()
    
    print("🚀 Starting ImageNet S3 Upload")
    print(f"📂 Local data: {args.local_data}")
    print(f"🪣 S3 bucket: {args.bucket}")
    print(f"📁 S3 prefix: {args.s3_prefix}")
    
    try:
        # Initialize uploader
        uploader = S3ImageNetUploader(args.bucket, args.aws_profile)
        
        # Create bucket if requested
        if args.create_bucket:
            if not uploader.create_bucket_if_not_exists(args.region):
                return 1
        
        # Upload dataset
        success = uploader.upload_imagenet_dataset(
            local_imagenet_path=args.local_data,
            s3_prefix=args.s3_prefix,
            max_workers=args.max_workers,
            verify_integrity=not args.no_verify
        )
        
        if success:
            # Create manifest
            uploader.create_dataset_manifest(args.s3_prefix)
            
            print("\n✅ Upload completed successfully!")
            print("🎯 Your SageMaker training data is ready at:")
            print(f"   s3://{args.bucket}/{args.s3_prefix}/")
            print("\n🚀 Launch SageMaker training with:")
            print(f"   python launch_sagemaker_job.py --train-data-s3 s3://{args.bucket}/{args.s3_prefix}/")
            
            return 0
        else:
            print("❌ Upload completed with errors. Check failed_uploads.txt")
            return 1
            
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())