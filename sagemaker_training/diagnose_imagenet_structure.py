#!/usr/bin/env python3
"""
Diagnostic script to analyze ImageNet dataset structure

Usage:
    python diagnose_imagenet_structure.py
"""

import boto3
from sagemaker_logging import setup_sagemaker_logger

def diagnose_structure():
    """Analyze the ImageNet dataset structure"""
    
    logger = setup_sagemaker_logger(__name__)
    
    # Configuration
    bucket_name = "tsai-era-v4-mini-capstone"
    source_prefix = "Datasets/imagenet1k/ILSVRC"
    
    logger.info("🔍 Diagnosing ImageNet dataset structure")
    logger.info(f"🪣 Bucket: {bucket_name}")
    logger.info(f"📂 Source: {source_prefix}")
    
    try:
        s3_client = boto3.client('s3')
        
        # 1. Check training classes
        logger.info("\n📋 Analyzing training classes...")
        train_prefix = f"{source_prefix}/Data/CLS-LOC/train/"
        
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=train_prefix,
            Delimiter='/'
        )
        
        train_classes = []
        for page in page_iterator:
            if 'CommonPrefixes' in page:
                for prefix_info in page['CommonPrefixes']:
                    folder_name = prefix_info['Prefix'].replace(train_prefix, '').rstrip('/')
                    if folder_name:
                        train_classes.append(folder_name)
        
        train_classes = sorted(train_classes)
        logger.info(f"✅ Found {len(train_classes)} training classes")
        logger.info(f"📝 First 10 classes: {train_classes[:10]}")
        logger.info(f"📝 Last 10 classes: {train_classes[-10:]}")
        
        # 2. Check val.txt structure
        logger.info("\n📋 Analyzing val.txt structure...")
        try:
            val_txt_key = f"{source_prefix}/ImageSets/CLS-LOC/val.txt"
            response = s3_client.get_object(Bucket=bucket_name, Key=val_txt_key)
            val_content = response['Body'].read().decode('utf-8')
            
            lines = val_content.strip().split('\n')
            logger.info(f"✅ Found val.txt with {len(lines)} lines")
            
            # Analyze class IDs
            class_ids = []
            for line in lines[:1000]:  # Sample first 1000 lines
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        class_id = int(parts[1])
                        class_ids.append(class_id)
                    except ValueError:
                        pass
            
            if class_ids:
                logger.info(f"📊 Class ID range: {min(class_ids)} - {max(class_ids)}")
                logger.info(f"📊 Unique class IDs in sample: {len(set(class_ids))}")
                logger.info(f"📝 First 10 class IDs: {class_ids[:10]}")
                
                # Check if class IDs exceed training classes
                max_valid = len(train_classes)
                out_of_range = [cid for cid in class_ids if cid > max_valid]
                if out_of_range:
                    logger.warning(f"⚠️ Found {len(out_of_range)} class IDs > {max_valid} (training class count)")
                    logger.warning(f"⚠️ Max out-of-range ID: {max(out_of_range)}")
                else:
                    logger.info(f"✅ All sampled class IDs are within valid range (1-{max_valid})")
            
            # Show sample mappings
            logger.info(f"\n📝 Sample val.txt entries:")
            for line in lines[:5]:
                logger.info(f"   {line.strip()}")
                
        except Exception as e:
            logger.error(f"❌ Could not analyze val.txt: {e}")
        
        # 3. Check validation images count
        logger.info("\n📋 Analyzing validation images...")
        val_images_prefix = f"{source_prefix}/Data/CLS-LOC/val/"
        
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=val_images_prefix
        )
        
        val_image_count = 0
        for page in page_iterator:
            if 'Contents' in page:
                val_image_count += len([obj for obj in page['Contents'] if not obj['Key'].endswith('/')])
        
        logger.info(f"✅ Found {val_image_count} validation images")
        
        # Summary
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Training classes: {len(train_classes)}")
        logger.info(f"   Validation images: {val_image_count}")
        logger.info(f"   Expected validation structure: ~50 images per class")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False

if __name__ == '__main__':
    diagnose_structure()