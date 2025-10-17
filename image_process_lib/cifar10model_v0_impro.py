
# cifar10model_v0.py - CIFAR-10 CNN with Depthwise Separable and Dilated Convolutions
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_setup import DataSetup
import logging
from image_preprocess_pytorch_imp_channel4out import image_processing

# ================================
# cifar10model_v0.py - Model Architecture
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution = Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise: each input channel convolved separately
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class Net(nn.Module):
    """CIFAR-10 CNN with C1C2C3C4 architecture, Depthwise Sep Conv, Dilated Conv, and GAP."""
    def __init__(self):
        super(Net, self).__init__()
        self.img_pro = image_processing()

        # C1: Initial feature extraction (32x32 -> 32x32)
        self.c1 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),    # 32x32x8, RF=3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),   # 32x32x16, RF=5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # 32x32x32, RF=7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # C2: Feature extraction with Dilated Convolutions (32x32 -> 32x32)
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=0, dilation=1, bias=False),  # 32x32x32, RF=9
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False),  # 32x32x32, RF=13
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3, padding=0, dilation=1, bias=False),  # 32x32x32, RF=21
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # C3: Pattern recognition with Depthwise Separable Conv (32x32 -> 16x16)
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=2),  # 32x32x32 -> 16x16x32, RF=23
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),  # 16x16x64, RF=25
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # 16x16x64, RF=27
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),

        )

        # C4: Final convolution with stride=2 (16x16 -> 8x8), then GAP + FC
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),  # 8x8x128, RF=55
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1x128, RF=covers entire input
            nn.Flatten(),
            nn.Linear(128, 10)  # FC after GAP to target classes
        )

    def forward(self, x):
        x = self.img_pro.extract_image_features(x, kernel=10)  # Preprocess image to single channel
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return F.log_softmax(x, dim=1)

class set_config_v0:
    """Basic configuration for CIFAR-10 training."""
    def __init__(self):
        self.epochs = 35
        self.nll_loss = torch.nn.NLLLoss()
        self.criterion = self.nll_loss

    def setup(self, model, use_onecycle: bool = True):
        self.use_onecycle = use_onecycle
        base_lr = 0.01
        self.optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        self.device = next(model.parameters()).device
        self.dataloader_args = self.get_dataloader_args()
        self.data_setup_instance = DataSetup(**self.dataloader_args)
        if self.use_onecycle:
            steps_per_epoch = len(self.data_setup_instance.train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=base_lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
                div_factor=10,
                final_div_factor=100,
                anneal_strategy='cos'
            )
            self.scheduler.batch_step = True
            logging.getLogger().info(
                f"Model v0: OneCycleLR max_lr={base_lr} pct_start=0.2 div_factor=10 final_div_factor=100 epochs={self.epochs}"
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.1)
            self.scheduler.batch_step = False
            logging.getLogger().info(
                f"Model v0: StepLR lr={base_lr} step_size=6 gamma=0.1"
            )
        logging.getLogger().info(f"Dataloader arguments: {self.dataloader_args}")
        return self

    def get_dataloader_args(self):
        if hasattr(self, 'device') and self.device.type == "cuda":
            args = dict(batch_size_train=32, batch_size_test=1000, shuffle_train=True, shuffle_test=False,
                        num_workers=2, pin_memory=True)
        else:
            args = dict(batch_size_train=32, batch_size_test=1000, shuffle_train=True, shuffle_test=False)
        logging.info(f"Model v0 dataloader args: {args}")
        return args

# ================================
