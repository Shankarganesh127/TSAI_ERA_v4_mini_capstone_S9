#!/bin/bash
# Simple setup script for S3 Dataset Converter

echo "🔧 Setting up S3 Dataset Converter for SageMaker..."

# Install converter dependencies
echo "📦 Installing converter dependencies..."
pip install -r converter_requirements.txt

# Install SageMaker dependencies  
echo "📦 Installing SageMaker dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Configure AWS credentials: aws configure"
echo "2. Convert your dataset: python s3_dataset_converter.py --bucket your-bucket --source-prefix ILSVRC --target-prefix imagenet-sagemaker"
echo "3. Launch training: python launch_sagemaker.py --job-name training --role-arn your-role --train-data-s3 s3://bucket/imagenet-sagemaker/"