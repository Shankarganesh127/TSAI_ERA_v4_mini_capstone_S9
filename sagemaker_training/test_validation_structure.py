#!/usr/bin/env python3
"""
Quick test script for validation file structure conversion

Usage:
    python test_validation_structure.py
"""

from s3_dataset_converter import S3DatasetConverter
from sagemaker_logging import setup_sagemaker_logger

def test_validation_structure():
    """Test only the validation data reorganization"""
    
    logger = setup_sagemaker_logger(__name__)
    
    # Configuration
    bucket_name = "tsai-era-v4-mini-capstone"
    source_prefix = "Datasets/imagenet1k/ILSVRC"
    target_prefix = "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker"
    
    logger.info("🧪 Testing validation file structure conversion")
    logger.info(f"🪣 Bucket: {bucket_name}")
    logger.info(f"📂 Source: {source_prefix}")
    logger.info(f"🎯 Target: {target_prefix}")
    
    try:
        # Initialize converter
        converter = S3DatasetConverter(bucket_name)
        
        # Test conversion
        success = converter.convert_ilsvrc_to_sagemaker(source_prefix, target_prefix)
        
        if success:
            logger.info("✅ Validation structure test completed successfully!")
            logger.info("\n📋 Results:")
            logger.info(f"✅ Training: s3://{bucket_name}/{source_prefix}/Data/CLS-LOC/train/")
            logger.info(f"✅ Validation: s3://{bucket_name}/{target_prefix}/val/")
            logger.info(f"✅ Test (optional): s3://{bucket_name}/{target_prefix}/test/")
        else:
            logger.error("❌ Validation structure test failed")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    return success

if __name__ == '__main__':
    success = test_validation_structure()
    exit(0 if success else 1)