# Launch SageMaker with Separate Train/Val Datasets

# Replace <bucket> with your actual bucket name
python sagemaker_training/launch_sagemaker.py \
    --s3-bucket "s3://<bucket>" \
    --train-data-s3 "Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/" \
    --val-data-s3 "Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/" \
    --instance-type "ml.p3.8xlarge" \
    --epochs 30 \
    --spot-training