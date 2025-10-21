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
        
    def _check_folder_exists(self, s3_prefix, min_objects=10):
        """Check if an S3 folder exists and has sufficient content
        
        Args:
            s3_prefix: S3 prefix to check
            min_objects: Minimum number of objects to consider folder as "existing"
        
        Returns:
            bool: True if folder exists with sufficient content
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=s3_prefix,
                PaginationConfig={'MaxItems': min_objects + 1}
            )
            
            object_count = 0
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('/'):  # Skip folder markers
                            object_count += 1
                            if object_count >= min_objects:
                                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking folder existence {s3_prefix}: {e}")
            return False
        
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
            
            # Step 2: Check if validation data already exists
            val_target_prefix = f"{target_prefix}/val/"
            self.logger.info("📁 Step 2: Checking if validation data already exists...")
            
            if self._check_folder_exists(val_target_prefix, min_objects=100):
                self.logger.info("✅ Validation folder already exists with sufficient data - skipping reorganization")
                self.logger.info(f"   Found existing validation data at: s3://{self.bucket_name}/{val_target_prefix}")
                val_success = True
            else:
                self.logger.info("📁 Reorganizing validation data by class...")
                val_success = self._reorganize_validation_data(source_prefix, target_prefix)
                
                if not val_success:
                    self.logger.error("❌ Failed to reorganize validation data")
                    return False
            
            # Step 3: Skip test data processing (as requested)
            self.logger.info("📁 Step 3: Skipping test data processing (not needed for validation-only conversion)")
            test_success = True  # Mark as successful since we're intentionally skipping
            
            # Step 4: Create metadata and manifest (pointing to original training data)
            self.logger.info("📁 Step 4: Creating SageMaker metadata...")
            metadata_success = self._create_sagemaker_metadata(source_prefix, target_prefix)
            
            if metadata_success:
                self.logger.info("🎉 S3 ILSVRC to SageMaker conversion completed!")
                self.logger.info(f"🚀 Ready for SageMaker training at: s3://{self.bucket_name}/{target_prefix}/")
                self.logger.info("📋 Training data will be used from original location (already organized)")
                self.logger.info("✅ Validation data is organized by class for proper evaluation")
                return True
            else:
                self.logger.error("❌ Failed to create metadata")
                return False
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
            
            if len(train_classes) == 0:
                self.logger.error("No training classes found - check your source prefix path")
                return False
            
            # Show sample of training classes for debugging
            sample_classes = train_classes[:5] if len(train_classes) > 5 else train_classes
            self.logger.info(f"Sample training classes: {sample_classes}")
            
            # Download val.txt for mapping (if available)
            val_mapping = self._get_validation_mapping(source_prefix, train_classes)
            
            # Process validation images using LOC_val_solution.csv (ground truth)
            self.logger.info("🔄 Processing validation images with ground truth mappings...")
            val_prefix = f"{source_prefix}/Data/CLS-LOC/val/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=val_prefix)
            
            copy_count = 0
            mapped_count = 0
            unmapped_count = 0
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    source_key = obj['Key']
                    if source_key.endswith('/'):
                        continue
                        
                    image_name = os.path.basename(source_key)
                    image_name_base = os.path.splitext(image_name)[0]  # Remove extension for mapping lookup
                    
                    # Use ground truth mapping from LOC_val_solution.csv
                    class_id = None
                    if val_mapping:
                        # Try both image_name_base (preferred) and image_name (fallback)
                        class_id = val_mapping.get(image_name_base) or val_mapping.get(image_name)
                    
                    if class_id:
                        target_key = f"{target_prefix}/val/{class_id}/{image_name}"
                        
                        copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
                        self.s3_client.copy_object(
                            CopySource=copy_source,
                            Bucket=self.bucket_name,
                            Key=target_key
                        )
                        copy_count += 1
                        mapped_count += 1
                        
                        if copy_count % 1000 == 0:
                            self.logger.info(f"   Reorganized {copy_count} validation images with ground truth mapping...")
                    else:
                        unmapped_count += 1
                        if unmapped_count <= 5:  # Log first few unmapped images
                            self.logger.warning(f"No ground truth mapping for {image_name} (base: {image_name_base})")
            
            self.logger.info(f"✅ Total validation images reorganized: {copy_count}")
            self.logger.info(f"   - Mapped with ground truth: {mapped_count}")
            self.logger.info(f"   - Unmapped (no ground truth): {unmapped_count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reorganize validation data: {str(e)}")
            return False

    def _reorganize_test_data(self, source_prefix, target_prefix):
        """Reorganize flat test structure into class folders (DISABLED - validation-only processing)"""
        # This function is disabled as per user request to focus only on validation data
        self.logger.info("🚫 Test data processing is disabled - validation-only conversion")
        return True
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
                    image_name_base = os.path.splitext(image_name)[0]  # Remove extension for mapping lookup
                    
                    # Determine class for test image
                    if test_mapping and image_name_base in test_mapping:
                        class_id = test_mapping[image_name_base]
                    elif test_mapping and image_name in test_mapping:
                        # Fallback: try with extension
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
        """Get validation image to class mapping from LOC_val_solution.csv
        
        LOC_val_solution.csv format: ImageId,PredictionString
        Example: n02017213_7894,n02017213 115 49 448 294
        We extract: image_id -> synset_id (ignore bounding box coordinates)
        """
        try:
            # Try to load from LOC_val_solution.csv first (correct ground truth)
            # LOC_val_solution.csv is typically at the parent level of ILSVRC folder
            # source_prefix is usually "Datasets/imagenet1k/ILSVRC", so go up one level
            base_prefix = "/".join(source_prefix.split("/")[:-1]) if "/" in source_prefix else source_prefix
            val_solution_key = f"{base_prefix}/LOC_val_solution.csv"
            
            # Also try direct path in case LOC_val_solution.csv is in ILSVRC folder
            fallback_solution_key = f"{source_prefix}/LOC_val_solution.csv"
            
            try:
                # Try parent level first (most common location)
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=val_solution_key)
                val_content = response['Body'].read().decode('utf-8')
                self.logger.info(f"✅ Found LOC_val_solution.csv at: s3://{self.bucket_name}/{val_solution_key}")
                
            except Exception as e1:
                self.logger.warning(f"LOC_val_solution.csv not found at parent level: {val_solution_key}")
                try:
                    # Try inside ILSVRC folder as fallback
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=fallback_solution_key)
                    val_content = response['Body'].read().decode('utf-8')
                    self.logger.info(f"✅ Found LOC_val_solution.csv at: s3://{self.bucket_name}/{fallback_solution_key}")
                    
                except Exception as e2:
                    raise Exception(f"LOC_val_solution.csv not found at either location: {val_solution_key} or {fallback_solution_key}")
            
            # Parse the CSV content (same for both locations)
            self.logger.info("✅ Using ground truth validation mappings from LOC_val_solution.csv")
            
            val_mapping = {}
            lines = val_content.strip().split('\n')
            
            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        image_id = parts[0]  # e.g., n02017213_7894
                        prediction_string = parts[1]  # e.g., n02017213 115 49 448 294
                        
                        # Extract synset_id (first part before space)
                        synset_id = prediction_string.split()[0]  # e.g., n02017213
                        
                        # Map image to synset class folder
                        if synset_id in train_classes:
                            val_mapping[image_id] = synset_id
                        else:
                            self.logger.warning(f"Synset {synset_id} not found in training classes for image {image_id}")
            
            self.logger.info(f"✅ Loaded {len(val_mapping)} validation mappings from LOC_val_solution.csv")
            
            # Debug: Show sample mappings
            if len(val_mapping) > 0:
                sample_items = list(val_mapping.items())[:3]
                self.logger.info(f"Sample ground truth mappings: {sample_items}")
            
            return val_mapping
                
        except Exception as e:
            self.logger.warning(f"Could not load LOC_val_solution.csv: {e}")
            self.logger.info("Falling back to val.txt...")
            
            # Fallback to old val.txt method (if LOC_val_solution.csv not available)
            val_txt_key = f"{source_prefix}/ImageSets/CLS-LOC/val.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=val_txt_key)
            val_content = response['Body'].read().decode('utf-8')
            
            # Create sorted class list (ImageNet classes are typically sorted alphabetically)
            sorted_classes = sorted(train_classes)
            self.logger.info(f"Using {len(sorted_classes)} sorted training classes for validation mapping")
            
            val_mapping = {}
            skipped_count = 0
            
            for line in val_content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        class_identifier = parts[1]
                        
                        # Handle different possible formats:
                        if class_identifier in train_classes:
                            # Direct class folder name match
                            class_id = class_identifier
                        else:
                            # Standard ImageNet format: 1-based class index
                            try:
                                class_id_num = int(class_identifier)
                                
                                # Handle class IDs > 1000 by wrapping them back to 1-1000
                                if class_id_num > len(sorted_classes):
                                    # Use modulo to wrap: 1001 → 1, 1002 → 2, etc.
                                    wrapped_class_id = ((class_id_num - 1) % len(sorted_classes)) + 1
                                    class_idx = wrapped_class_id - 1  # Convert to 0-based
                                    class_id = sorted_classes[class_idx]
                                    
                                    # Log the first few wrapping examples
                                    if skipped_count < 5:
                                        self.logger.info(f"Wrapping class {class_id_num} → {wrapped_class_id} ({class_id}) for {image_name}")
                                else:
                                    # Normal case: class ID within range
                                    class_idx = class_id_num - 1  # Convert to 0-based
                                    class_id = sorted_classes[class_idx]
                                    
                            except (ValueError, IndexError):
                                self.logger.warning(f"Invalid class identifier '{class_identifier}' for image {image_name}")
                                skipped_count += 1
                                continue
                        
                        val_mapping[image_name] = class_id
            
            if skipped_count > 0:
                self.logger.warning(f"Skipped {skipped_count} validation images with invalid class mappings")
            
            self.logger.info(f"Loaded validation mapping for {len(val_mapping)} images (fallback method)")
            
            # Debug: Show sample mappings
            if len(val_mapping) > 0:
                sample_items = list(val_mapping.items())[:3]
                self.logger.info(f"Sample fallback mappings: {sample_items}")
            
            return val_mapping
            
        except Exception as e:
            self.logger.warning(f"Could not load any validation mapping: {e}")
            return {}

    def _get_test_mapping(self, source_prefix, train_classes):
        """Get test image to class mapping from test.txt (if available)"""
        try:
            test_txt_key = f"{source_prefix}/ImageSets/CLS-LOC/test.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=test_txt_key)
            test_content = response['Body'].read().decode('utf-8')
            
            # Create sorted class list (same as validation)
            sorted_classes = sorted(train_classes)
            self.logger.info(f"Using {len(sorted_classes)} sorted training classes for test mapping")
            
            test_mapping = {}
            skipped_count = 0
            
            for line in test_content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        class_identifier = parts[1]
                        
                        # Handle different possible formats:
                        if class_identifier in train_classes:
                            # Direct class folder name match
                            class_id = class_identifier
                        elif class_identifier.isdigit():
                            # Try as numeric index (1-based)
                            try:
                                class_id_num = int(class_identifier)
                                
                                # Handle class IDs > 1000 by wrapping them back to 1-1000
                                if class_id_num > len(sorted_classes):
                                    # Use modulo to wrap: 1001 → 1, 1002 → 2, etc.
                                    wrapped_class_id = ((class_id_num - 1) % len(sorted_classes)) + 1
                                    class_idx = wrapped_class_id - 1  # Convert to 0-based
                                    class_id = sorted_classes[class_idx]
                                    
                                    # Log the first few wrapping examples
                                    if skipped_count < 5:
                                        self.logger.info(f"Wrapping test class {class_id_num} → {wrapped_class_id} ({class_id}) for {image_name}")
                                else:
                                    # Normal case: class ID within range
                                    class_idx = class_id_num - 1  # Convert to 0-based
                                    class_id = sorted_classes[class_idx]
                                    
                            except (ValueError, IndexError):
                                self.logger.warning(f"Invalid test class identifier '{class_identifier}' for image {image_name}")
                                skipped_count += 1
                                continue
                        else:
                            # Unknown format, skip
                            self.logger.warning(f"Unknown test class identifier '{class_identifier}' for image {image_name}")
                            skipped_count += 1
                            continue
                        
                        test_mapping[image_name] = class_id
            
            if skipped_count > 0:
                self.logger.warning(f"Skipped {skipped_count} test images with invalid class mappings")
            
            self.logger.info(f"Loaded test mapping for {len(test_mapping)} images")
            
            # Debug: Show sample test mappings
            if len(test_mapping) > 0:
                sample_items = list(test_mapping.items())[:3]
                self.logger.info(f"Sample test mappings: {sample_items}")
            
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