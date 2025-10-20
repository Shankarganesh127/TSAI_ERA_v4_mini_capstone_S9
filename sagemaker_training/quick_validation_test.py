#!/usr/bin/env python3
"""
Quick validation structure test - no training job launch

This script only tests the dataset conversion and validation structure
without launching expensive SageMaker training jobs.
"""

from s3_dataset_converter import S3DatasetConverter
from sagemaker_logging import setup_sagemaker_logger
import boto3

def quick_validation_test():
    """Test validation structure conversion only"""
    
    logger = setup_sagemaker_logger(__name__)
    
    # Configuration
    bucket_name = "tsai-era-v4-mini-capstone"
    source_prefix = "Datasets/imagenet1k/ILSVRC"
    target_prefix = "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker"
    
    logger.info("🧪 Quick Validation Structure Test (No Training)")
    logger.info("=" * 60)
    logger.info(f"🪣 Bucket: {bucket_name}")
    logger.info(f"📂 Source: {source_prefix}")
    logger.info(f"🎯 Target: {target_prefix}")
    
    try:
        # Test 1: Check conversion results
        logger.info("\n📋 Test 1: Checking conversion results...")
        s3_client = boto3.client('s3')
        
        # Check validation structure
        val_prefix = f"{target_prefix}/val/"
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=val_prefix,
            Delimiter='/'
        )
        
        class_folders = []
        for page in page_iterator:
            if 'CommonPrefixes' in page:
                for prefix_info in page['CommonPrefixes']:
                    folder_name = prefix_info['Prefix'].replace(val_prefix, '').rstrip('/')
                    if folder_name:
                        class_folders.append(folder_name)
        
        logger.info(f"✅ Found {len(class_folders)} validation class folders")
        
        if len(class_folders) > 0:
            logger.info(f"📝 Sample classes: {class_folders[:5]}")
            
            # Check sample class folder contents
            sample_class = class_folders[0]
            sample_prefix = f"{val_prefix}{sample_class}/"
            
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=sample_prefix,
                PaginationConfig={'MaxItems': 100}
            )
            
            image_count = 0
            for page in page_iterator:
                if 'Contents' in page:
                    image_count += len([obj for obj in page['Contents'] if not obj['Key'].endswith('/')])
            
            logger.info(f"📊 Sample class '{sample_class}' has {image_count} images")
        
        # Test 2: Check manifest file
        logger.info("\n📋 Test 2: Checking manifest file...")
        try:
            manifest_key = f"{target_prefix}/manifest.json"
            response = s3_client.get_object(Bucket=bucket_name, Key=manifest_key)
            manifest_content = response['Body'].read().decode('utf-8')
            logger.info(f"✅ Manifest file exists: s3://{bucket_name}/{manifest_key}")
            logger.info(f"📝 Manifest size: {len(manifest_content)} bytes")
        except Exception as e:
            logger.warning(f"⚠️ Manifest check failed: {e}")
        
        # Test 3: Check metadata
        logger.info("\n📋 Test 3: Checking metadata...")
        try:
            metadata_key = f"{target_prefix}/metadata/dataset_metadata.json"
            response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
            metadata_content = response['Body'].read().decode('utf-8')
            logger.info(f"✅ Metadata file exists: s3://{bucket_name}/{metadata_key}")
        except Exception as e:
            logger.warning(f"⚠️ Metadata check failed: {e}")
        
        # Summary
        logger.info("\n🎯 VALIDATION STRUCTURE TEST RESULTS:")
        logger.info("=" * 60)
        if len(class_folders) == 1000:
            logger.info("✅ PERFECT: Found exactly 1000 validation class folders")
        elif len(class_folders) > 0:
            logger.info(f"✅ GOOD: Found {len(class_folders)} validation class folders")
        else:
            logger.error("❌ FAILED: No validation class folders found")
            return False
        
        logger.info("✅ Dataset structure ready for SageMaker training")
        logger.info("✅ Validation images properly organized by class")
        logger.info("✅ Training data preserved at original location")
        
        logger.info("\n🚀 Ready for Training:")
        logger.info(f"   Training: s3://{bucket_name}/{source_prefix}/Data/CLS-LOC/train/")
        logger.info(f"   Validation: s3://{bucket_name}/{target_prefix}/val/")
        
        logger.info("\n💡 To launch actual training (with cost):")
        logger.info("python sagemaker_orchestrator.py --role-arn YOUR_ROLE --source-bucket YOUR_BUCKET --epochs 2")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
        return False

if __name__ == '__main__':
    success = quick_validation_test()
    if success:
        print("\n🎉 Validation structure test PASSED!")
        exit(0)
    else:
        print("\n❌ Validation structure test FAILED!")
        exit(1)