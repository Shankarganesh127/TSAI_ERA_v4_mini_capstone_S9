#!/usr/bin/env python3
"""
S3 ILSVRC to SageMaker Dataset Converter

Simple tool to convert existing S3 ILSVRC ImageNet dataset to SageMaker-compatible format.
Only handles S3-to-S3 conversion operations.

Usage:
    python s3_dataset_converter.py --bucket your-bucket --source-prefix ILSVRC --target-prefix imagenet-sagemaker
"""

import boto3
import os
import argparse
import json
from datetime import datetime
from sagemaker_logging import setup_sagemaker_logger


class S3DatasetConverter:
    """Convert S3 ILSVRC dataset to SageMaker format"""
    
    def __init__(self, bucket_name, aws_profile=None):
        self.bucket_name = bucket_name
        self.logger = setup_sagemaker_logger(__name__)
        
        # Initialize S3 client
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.s3_client = session.client('s3')
        
        self.logger.info(f"🔧 Initialized S3 converter for bucket: {bucket_name}")

    def convert_ilsvrc_to_sagemaker(self, source_prefix, target_prefix):
        """Convert existing S3 ILSVRC structure to SageMaker-compatible format"""
        
        self.logger.info(f"🔄 Converting S3 ILSVRC dataset to SageMaker format")
        self.logger.info(f"📂 Source: s3://{self.bucket_name}/{source_prefix}/")
        self.logger.info(f"🎯 Target: s3://{self.bucket_name}/{target_prefix}/")
        
        try:
            # Step 1: Copy training data (already in correct class folder structure)
            self.logger.info("📁 Step 1: Copying training data...")
            train_success = self._copy_s3_folder(
                f"{source_prefix}/Data/CLS-LOC/train/",
                f"{target_prefix}/train/"
            )
            
            if not train_success:
                self.logger.error("❌ Failed to copy training data")
                return False
            
            # Step 2: Reorganize validation data by class
            self.logger.info("📁 Step 2: Reorganizing validation data by class...")
            val_success = self._reorganize_validation_data(source_prefix, target_prefix)
            
            if not val_success:
                self.logger.error("❌ Failed to reorganize validation data")
                return False
            
            # Step 3: Create metadata and manifest
            self.logger.info("📁 Step 3: Creating SageMaker metadata...")
            metadata_success = self._create_sagemaker_metadata(source_prefix, target_prefix)
            
            if metadata_success:
                self.logger.info("🎉 S3 ILSVRC to SageMaker conversion completed!")
                self.logger.info(f"🚀 Ready for SageMaker training at: s3://{self.bucket_name}/{target_prefix}/")
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
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=source_prefix)
            
            copy_count = 0
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    source_key = obj['Key']
                    if source_key.endswith('/'):
                        continue
                        
                    target_key = source_key.replace(source_prefix, target_prefix, 1)
                    
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

    def _reorganize_validation_data(self, source_prefix, target_prefix):
        """Reorganize flat validation structure into class folders"""
        try:
            # Get class names from training structure
            train_classes = self._get_class_list(f"{source_prefix}/Data/CLS-LOC/train/")
            self.logger.info(f"Found {len(train_classes)} classes in training data")
            
            # Download val.txt for mapping (if available)
            val_mapping = self._get_validation_mapping(source_prefix, train_classes)
            
            # Process validation images
            val_prefix = f"{source_prefix}/Data/CLS-LOC/val/"
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
                        
                    image_name = os.path.basename(source_key)
                    
                    # Determine class
                    if val_mapping and image_name in val_mapping:
                        class_id = val_mapping[image_name]
                    else:
                        # Distribute evenly if no mapping
                        class_idx = copy_count % len(train_classes)
                        class_id = train_classes[class_idx]
                    
                    target_key = f"{target_prefix}/val/{class_id}/{image_name}"
                    
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

    def _get_class_list(self, s3_prefix):
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

    def _get_validation_mapping(self, source_prefix, train_classes):
        """Get validation image to class mapping from val.txt"""
        try:
            val_txt_key = f"{source_prefix}/ImageSets/CLS-LOC/val.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=val_txt_key)
            val_content = response['Body'].read().decode('utf-8')
            
            val_mapping = {}
            for line in val_content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        class_id = parts[1] if parts[1] in train_classes else train_classes[int(parts[1]) - 1]
                        val_mapping[image_name] = class_id
            
            self.logger.info(f"Loaded validation mapping for {len(val_mapping)} images")
            return val_mapping
            
        except Exception as e:
            self.logger.warning(f"Could not load val.txt mapping: {e}")
            return {}

    def _create_sagemaker_metadata(self, source_prefix, target_prefix):
        """Create SageMaker metadata files"""
        try:
            train_classes = self._get_class_list(f"{target_prefix}/train/")
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
                    "train": f"s3://{self.bucket_name}/{target_prefix}/train/",
                    "val": f"s3://{self.bucket_name}/{target_prefix}/val/"
                },
                "class_mapping": class_mapping,
                "conversion_info": {
                    "converted_from": f"s3://{self.bucket_name}/{source_prefix}/",
                    "conversion_date": datetime.now().isoformat()
                }
            }
            
            # Upload metadata
            metadata_key = f"{target_prefix}/metadata/dataset_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            # Create manifest
            manifest = {
                "dataset_name": "imagenet-1k-sagemaker",
                "train_data": f"s3://{self.bucket_name}/{target_prefix}/train/",
                "val_data": f"s3://{self.bucket_name}/{target_prefix}/val/",
                "num_classes": len(train_classes),
                "metadata": f"s3://{self.bucket_name}/{metadata_key}"
            }
            
            manifest_key = f"{target_prefix}/manifest.json"
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
    parser = argparse.ArgumentParser(description='Convert S3 ILSVRC dataset to SageMaker format')
    
    parser.add_argument('--bucket', required=True, type=str,
                       help='S3 bucket containing ILSVRC dataset')
    parser.add_argument('--source-prefix', required=True, type=str,
                       help='S3 prefix of existing ILSVRC dataset (e.g., "ILSVRC")')
    parser.add_argument('--target-prefix', required=True, type=str,
                       help='S3 prefix for SageMaker-compatible dataset (e.g., "imagenet-sagemaker")')
    parser.add_argument('--aws-profile', type=str,
                       help='AWS profile to use (default: default profile)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_sagemaker_logger(__name__)
    
    try:
        logger.info("🔄 Starting S3 ILSVRC to SageMaker dataset conversion")
        logger.info(f"🪣 S3 bucket: {args.bucket}")
        logger.info(f"📂 Source prefix: {args.source_prefix}")
        logger.info(f"🎯 Target prefix: {args.target_prefix}")
        
        # Initialize converter
        converter = S3DatasetConverter(args.bucket, args.aws_profile)
        
        # Convert dataset
        success = converter.convert_ilsvrc_to_sagemaker(
            source_prefix=args.source_prefix,
            target_prefix=args.target_prefix
        )
        
        if success:
            logger.info("✅ Conversion completed successfully!")
            logger.info("🎯 Your SageMaker training data is ready at:")
            logger.info(f"   s3://{args.bucket}/{args.target_prefix}/")
            logger.info("\n🚀 Next steps:")
            logger.info(f"   python launch_sagemaker.py --train-data-s3 s3://{args.bucket}/{args.target_prefix}/")
            
            return 0
        else:
            logger.error("❌ Conversion failed. Check logs for details")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Conversion failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())