from typing import Dict

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetV3FixedSlotDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_slots: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.backbone = backbone.features
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        self.neck = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.slot_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_slots * hidden_dim),
            nn.Hardswish(),
        )
        self.box_head = nn.Linear(hidden_dim, 4)
        self.objectness_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.constant_(self.objectness_head.bias, -2.0)
        nn.init.xavier_uniform_(self.box_head.weight)
        nn.init.constant_(self.box_head.bias, 0.0)
        with torch.no_grad():
            self.box_head.bias.copy_(torch.tensor([0.5, 0.5, 0.3, 0.3]))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.neck(features)
        pooled = self.pool(features)
        slot_features = self.slot_projection(pooled)
        slot_features = slot_features.view(images.shape[0], self.num_slots, self.hidden_dim)

        box_predictions = torch.sigmoid(self.box_head(slot_features))
        objectness_logits = self.objectness_head(slot_features).squeeze(-1)
        class_logits = self.class_head(slot_features)
        return {
            "boxes": box_predictions,
            "objectness_logits": objectness_logits,
            "class_logits": class_logits,
        }

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        modules = [self.neck, self.slot_projection, self.box_head, self.objectness_head, self.class_head]
        for module in modules:
            yield from module.parameters()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
