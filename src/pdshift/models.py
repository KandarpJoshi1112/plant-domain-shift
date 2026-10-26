"""Model factory."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import models

logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """Small from-scratch CNN.

    Replace this with the architecture from the original CNN-Plant-Disease-Classifier
    repository so that E1 is a faithful baseline.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Create a classifier.

    Args:
        name: ``custom_cnn`` or ``resnet18``.
        num_classes: Size of the shared label space.
        pretrained: Load ImageNet weights (ResNet only).
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model '{name}'")
