import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection.ssd import SSDClassificationHead

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
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adaptar conv1 de 3 canals a 1 canal
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=(old_conv.kernel_size[0], old_conv.kernel_size[1]),
            stride=(old_conv.stride[0], old_conv.stride[1]),
            padding=(old_conv.padding[0], old_conv.padding[1]),
            bias=old_conv.bias is not None
        )

        # Inicialitzar pesos: mitjana dels 3 canals
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.conv1 = new_conv

        # Capçal binari
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# -----------------------------------------
# Detection: SSD300 + VGG16
# -----------------------------------------
class SSD300VGG16Detector(nn.Module):
    """
    SSD300 + VGG16 (torchvision) adaptat a 2 classes:
      - 0: background
      - 1: nodule

    NOTA: El model preentrenat espera imatges de 3 canals.
    Recomanat: convertir el teu tensor [1,H,W] a [3,H,W] al Dataset (repeat).
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = models.detection.SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model = models.detection.ssd300_vgg16(weights=weights)

        # classification_head és un SSDClassificationHead(SSDScoringHead) que té module_list de convs
        old_cls_head = model.head.classification_head
        in_channels = [m.in_channels for m in old_cls_head.module_list]

        # num_anchors per feature-map level
        num_anchors = model.anchor_generator.num_anchors_per_location()

        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        self.model = model

    def forward(self, images, targets=None):
        """
        images: list[Tensor] amb forma [3,H,W] (per pesos preentrenats)
        targets (train): list[dict] amb:
          - boxes: FloatTensor [N,4] (x1,y1,x2,y2)
          - labels: Int64Tensor [N] (1 per nòdul)
        """
        return self.model(images, targets)




