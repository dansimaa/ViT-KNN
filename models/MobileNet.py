import torch
import torch.nn as nn
from torchvision import models


class MobileNetBinaryClassifier(nn.Module):
    """
    A binary classifier using MobileNetV2 as the base model.
    The classifier head is modified to output a single value.
    """

    def __init__(self, pretrained=True):
        """
        Load MobileNetV2 model.
        Replaces the default classification head with a 
        single neuron for binary classification.
        """
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 1
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.model(x)