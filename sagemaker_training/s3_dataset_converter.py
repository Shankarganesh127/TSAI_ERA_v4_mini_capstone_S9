#!/usr/bin/env python3
"""
S3 ILSVRC to SageMaker Dataset Converter

Optimized tool to convert S3 ILSVRC validation and test data to SageMaker-compatible format.
Skips training data copy (already organized) and only reorganizes val/test data by class.

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
            # Step 1: Skip training data copy (already in proper class structure)
            self.logger.info("📁 Step 1: Skipping training data copy (already in proper class structure)")
            self.logger.info("   Training data will be used directly from source location")
            
            # Step 2: Reorganize validation data by class
            self.logger.info("📁 Step 2: Reorganizing validation data by class...")
            val_success = self._reorganize_validation_data(source_prefix, target_prefix)
            
            if not val_success:
                self.logger.error("❌ Failed to reorganize validation data")
                return False
            
            # Step 3: Reorganize test data by class (if exists)
            self.logger.info("📁 Step 3: Reorganizing test data by class...")
            test_success = self._reorganize_test_data(source_prefix, target_prefix)
            
            if not test_success:
                self.logger.warning("⚠️ Test data processing failed or no test data found")
                # Don't fail the entire conversion if test data is missing
            
            # Step 4: Create metadata and manifest (pointing to original training data)
            self.logger.info("📁 Step 4: Creating SageMaker metadata...")
            metadata_success = self._create_sagemaker_metadata(source_prefix, target_prefix)
            
            if metadata_success:
                self.logger.info("🎉 S3 ILSVRC to SageMaker conversion completed!")
                self.logger.info(f"🚀 Ready for SageMaker training at: s3://{self.bucket_name}/{target_prefix}/")
                self.logger.info("📋 Training data will be used from original location (already organized)")
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

    def _reorganize_test_data(self, source_prefix, target_prefix):
        """Reorganize flat test structure into class folders"""
        try:
            # Check if test data exists
            test_prefix = f"{source_prefix}/Data/CLS-LOC/test/"
            
            # List test directory to see if it exists and has content
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=test_prefix, 
                PaginationConfig={'MaxItems': 1}
            )
            
            test_exists = False
            for page in page_iterator:
                if 'Contents' in page and len(page['Contents']) > 0:
                    test_exists = True
                    break
            
            if not test_exists:
                self.logger.info("   No test data found, skipping test reorganization")
                return True
            
            # Get class names from training structure
            train_classes = self._get_class_list(f"{source_prefix}/Data/CLS-LOC/train/")
            self.logger.info(f"Found {len(train_classes)} classes for test data organization")
            
            # Try to get test mapping from test.txt (if available)
            test_mapping = self._get_test_mapping(source_prefix, train_classes)
            
            # Process test images
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=test_prefix)
            
            copy_count = 0
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    source_key = obj['Key']
                    if source_key.endswith('/'):
                        continue
                        
                    image_name = os.path.basename(source_key)
                    
                    # Determine class for test image
                    if test_mapping and image_name in test_mapping:
                        class_id = test_mapping[image_name]
                    else:
                        # For test data without labels, distribute evenly or use special folder
                        # Option 1: Distribute evenly across classes
                        class_idx = copy_count % len(train_classes)
                        class_id = train_classes[class_idx]
                        
                        # Option 2: Use a special 'unknown' folder (uncomment if preferred)
                        # class_id = "unknown"
                    
                    target_key = f"{target_prefix}/test/{class_id}/{image_name}"
                    
                    copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
                    self.s3_client.copy_object(
                        CopySource=copy_source,
                        Bucket=self.bucket_name,
                        Key=target_key
                    )
                    copy_count += 1
                    
                    if copy_count % 1000 == 0:
                        self.logger.info(f"   Reorganized {copy_count} test images...")
            
            self.logger.info(f"   Total test images reorganized: {copy_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reorganize test data: {str(e)}")
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

    def _get_test_mapping(self, source_prefix, train_classes):
        """Get test image to class mapping from test.txt (if available)"""
        try:
            test_txt_key = f"{source_prefix}/ImageSets/CLS-LOC/test.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=test_txt_key)
            test_content = response['Body'].read().decode('utf-8')
            
            test_mapping = {}
            for line in test_content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        # Test data might not have class labels, so handle gracefully
                        if len(parts) > 1 and parts[1] in train_classes:
                            class_id = parts[1]
                        elif len(parts) > 1 and parts[1].isdigit():
                            class_idx = int(parts[1]) - 1
                            class_id = train_classes[class_idx] if class_idx < len(train_classes) else train_classes[0]
                        else:
                            # If no valid class info, skip this entry
                            continue
                        test_mapping[image_name] = class_id
            
            self.logger.info(f"Loaded test mapping for {len(test_mapping)} images")
            return test_mapping
            
        except Exception as e:
            self.logger.info(f"Could not load test.txt mapping (test data may not have labels): {e}")
            return {}

    def _create_sagemaker_metadata(self, source_prefix, target_prefix):
        """Create SageMaker metadata files"""
        try:
            # Get training classes from original location (not copied)
            train_classes = self._get_class_list(f"{source_prefix}/Data/CLS-LOC/train/")
            class_mapping = {class_id: idx for idx, class_id in enumerate(sorted(train_classes))}
            
            # Check if test data was converted
            test_exists = self._check_test_data_exists(target_prefix)
            
            # Create metadata with original training path and converted val/test paths
            s3_paths = {
                "train": f"s3://{self.bucket_name}/{source_prefix}/Data/CLS-LOC/train/",  # Original location
                "val": f"s3://{self.bucket_name}/{target_prefix}/val/"  # Converted location
            }
            
            if test_exists:
                s3_paths["test"] = f"s3://{self.bucket_name}/{target_prefix}/test/"  # Converted location
            
            metadata = {
                "dataset_info": {
                    "name": "ImageNet-1K",
                    "source": "ILSVRC Converted from S3",
                    "num_classes": len(train_classes),
                    "image_size": 224,
                    "has_test_data": test_exists,
                    "training_data_location": "original"  # Indicate training data not copied
                },
                "s3_paths": s3_paths,
                "class_mapping": class_mapping,
                "conversion_info": {
                    "converted_from": f"s3://{self.bucket_name}/{source_prefix}/",
                    "conversion_date": datetime.now().isoformat(),
                    "training_data_copied": False,  # Training data not copied
                    "validation_data_reorganized": True,
                    "test_data_reorganized": test_exists
                }
            }
            
            # Upload metadata
            metadata_key = f"{target_prefix}/metadata/dataset_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            # Create manifest with original training path
            manifest = {
                "dataset_name": "imagenet-1k-sagemaker",
                "train_data": f"s3://{self.bucket_name}/{source_prefix}/Data/CLS-LOC/train/",  # Original location
                "val_data": f"s3://{self.bucket_name}/{target_prefix}/val/",  # Converted location
                "num_classes": len(train_classes),
                "metadata": f"s3://{self.bucket_name}/{metadata_key}",
                "note": "Training data uses original ILSVRC location (already organized by class)"
            }
            
            # Add test data to manifest if available
            if test_exists:
                manifest["test_data"] = f"s3://{self.bucket_name}/{target_prefix}/test/"  # Converted location
            
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

    def _check_test_data_exists(self, target_prefix):
        """Check if test data was successfully converted"""
        try:
            test_prefix = f"{target_prefix}/test/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=test_prefix, 
                PaginationConfig={'MaxItems': 1}
            )
            
            for page in page_iterator:
                if 'Contents' in page and len(page['Contents']) > 0:
                    return True
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking test data: {e}")
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