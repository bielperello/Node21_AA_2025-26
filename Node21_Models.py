import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
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
#  Model 2: Transfer Learning: ResNet18Binary (1 canal)
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
# Model 3: DenseNet121 (Transfer Learning)
# -----------------------------------------

class DenseNet121Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # 1. Adaptar la primera capa (conv0) de 3 canals a 1 canal
        old_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Inicialitzar pesos: mitjana dels 3 canals originals
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.features.conv0 = new_conv

        # 2. Adaptar el classificador final
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------------------
# Model 4: InceptionV3 (Transfer Learning)
# -----------------------------------------
class InceptionV3Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.inception_v3(weights='DEFAULT', aux_logits=True, transform_input=False)

        # InceptionV3 necessita desactivar aux_logits per no fallar en el forward simple
        self.backbone.transform_input = False
        self.backbone._transform_input = lambda x: x

        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None

        # 1. Adaptar la primera capa (Conv2d_1a_3x3.conv) de 3 canals a 1
        old_conv = self.backbone.Conv2d_1a_3x3.conv
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.Conv2d_1a_3x3.conv = new_conv

        # 2. Adaptar la sortida (fc)
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
# DETECCIÓ
# -----------------------------------------

# -----------------------------------------
# Model 1: RetinaNetDetector
# -----------------------------------------
class RetinaNetDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

        self.model = retinanet_resnet50_fpn_v2(weights=weights, min_size=512, max_size=512)

        base_sizes = [8, 16, 32, 64, 128]

        scales = [2**0, 2**(1/3), 2**(2/3)]

        anchor_sizes = tuple(tuple(int(s * sc) for sc in scales) for s in base_sizes)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        new_anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.model.anchor_generator = new_anchor_gen

        in_channels = self.model.head.classification_head.conv[0][0].in_channels
        num_anchors = new_anchor_gen.num_anchors_per_location()[0]

        self.model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.model.head.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def forward(self, images, targets=None):
        return self.model(images, targets)

# -----------------------------------------
# Model 2: SSD300VGG16Detector
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
