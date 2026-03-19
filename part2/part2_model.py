from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetDetector(nn.Module):
    def __init__(self, hidden_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(960, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, images):
        features = self.features(images)
        pooled = self.avgpool(features)
        flattened = pooled.flatten(1)
        return self.regressor(flattened)

    def backbone_parameters(self):
        return self.features.parameters()

    def head_parameters(self):
        return self.regressor.parameters()

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = True
