"""SignCNN Model Architecture for SignBridge.

A lightweight, efficient Convolutional Neural Network designed for real-time
sign language gesture recognition (ASL alphabet and utility gestures).
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, ReLU, and optional Pooling."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SignCNN(nn.Module):
    """Convolutional Neural Network for ASL alphabet gesture classification."""

    def __init__(self, num_classes: int = 29, in_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size

        # Feature extractor
        self.layer1 = ConvBlock(in_channels, 32, pool=True, dropout=0.05)   # -> (32, img_size/2, img_size/2)
        self.layer2 = ConvBlock(32, 64, pool=True, dropout=0.1)             # -> (64, img_size/4, img_size/4)
        self.layer3 = ConvBlock(64, 128, pool=True, dropout=0.15)           # -> (128, img_size/8, img_size/8)
        self.layer4 = ConvBlock(128, 256, pool=False, dropout=0.2)          # -> (256, img_size/8, img_size/8)

        # Adaptive pool to ensure fixed feature size regardless of input dimension
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_size = 256 * 4 * 4

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return {
            "model_type": "SignCNN",
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "img_size": self.img_size,
            "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
