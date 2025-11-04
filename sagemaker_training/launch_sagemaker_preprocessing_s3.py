from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:2.0.1-cpu-py310",
    role="arn:aws:iam::872109682518:role/service-role/AmazonSageMaker-ExecutionRole-20251009T010774",
    instance_count=1,
    instance_type="ml.m5.4xlarge",
    command=["python3"],
)

'''
s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/val
s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/LOC_val_solution.csv
s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val
'''

processor.run(
    code="convert_val_s3.py",
    arguments=[
        "--input_s3", "s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/val",
        "--labels_s3", "s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/LOC_val_solution.csv",
        "--output_s3", "s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val"
    ]
)
