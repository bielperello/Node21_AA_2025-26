import torch
import torch.nn as nn
from torchvision import models

class TinyCXRNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 512 → 256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 256 → 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 128 → 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Pooling 2 * [128, 4, 4]
        self.pool_avg = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_max = nn.AdaptiveMaxPool2d((4, 4))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * 128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)  # Sortida: 1 logits (classe 0 i 1)
        )

    def forward(self, x):
        x = self.features(x)
        xa = self.pool_avg(x)
        xm = self.pool_max(x)
        x = torch.cat([xa, xm], dim=1)  # concat canals: [B, 256, 4, 4]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_tiny_cxrnet(in_channels=1):
    return TinyCXRNet(in_channels=in_channels)


# -----------------------------------------
# Transfer Learning: ResNet18Binary (1 canal)
# -----------------------------------------
class ResNet18Binary(nn.Module):
    """
    ResNet-18 preentrenada per classificació binària CXR (nòdul / no nòdul)
    Sortida: 1 logit (sense Sigmoid)
    """
    def __init__(self):
        super().__init__()

        # Backbone preentrenat
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Substituïm el classifier final
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)


