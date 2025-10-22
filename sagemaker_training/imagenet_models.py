#!/usr/bin/env python3
"""
ResNet models optimized for ImageNet-1K training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger_setup import get_logger
import smdistributed.dataparallel.torch.torch_smddp
import torch.distributed as dist
import os
import torch

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNetImageNet(nn.Module):
    """ResNet for ImageNet-1K (224x224 input, 1000 classes)"""
    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetImageNet, self).__init__()
        self.in_channels = 64

        # Initial convolution for ImageNet (larger kernel, stride=2, maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def model_device_setup_for_ddp(model):
    """
    Setup device and device_ids for DistributedDataParallel (DDP)
    based on LOCAL_RANK environment variable.
    
    Returns:
        current_device: torch.device for the current process
        device_ids: List of device IDs for DDP constructor
    """

    # --- 1. Get the local rank from the environment ---
    # The 'LOCAL_RANK' environment variable is typically set by the launcher
    # (e.g., torchrun, sagemaker's distributed setup) for each process.
    try:
        # Use the rank assigned to this specific process on the current node.
        local_rank = int(os.environ['LOCAL_RANK'])
    except KeyError:
        # Fallback for non-distributed or single-GPU testing
        local_rank = 0

    print(f"Process is running on local GPU rank: {local_rank}")

    # --- 2. Set the device for the current process ---
    # This ensures PyTorch operations and the model are correctly placed.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        current_device = torch.device('cuda', local_rank)
    else:
        current_device = torch.device('cpu')

    # --- 3. Determine device_ids for DDP constructor ---
    # For DistributedDataParallel, device_ids is a list containing the local rank.
    if current_device.type == 'cuda':
        device_ids = [local_rank]
    else:
        # device_ids is often ignored or set to None for CPU-only DDP (which is rare)
        device_ids = None

    print(f"DDP device_ids parameter: {device_ids}")
    dist.init_process_group(backend='smddp')
    model = torch.nn.parallel.DistributedDataParallel(model.to(current_device), device_ids=device_ids)

def resnet50_imagenet(num_classes=1000, pretrained=False):
    """
    ResNet-50 model for ImageNet-1K
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        pretrained: Whether to load pretrained weights
    """
    model = ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes)
    
    model = model_device_setup_for_ddp(model)
    
    if pretrained:
        # Load pretrained weights from torchvision
        import torchvision.models as models
        pretrained_model = models.resnet50(pretrained=True)
        
        # Copy weights (excluding final FC layer if num_classes != 1000)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out unnecessary keys and mismatched fc layer
        if num_classes != 1000:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and not k.startswith('fc.')}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test the model
    logger = get_logger("imagenet_models")
    
    model = resnet50_imagenet(num_classes=1000)
    total, trainable = count_parameters(model)
    logger.info("ResNet-50 for ImageNet:")
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")