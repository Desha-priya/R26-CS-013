#ResNet18 Model for Audio Phishing Detection


import torch
import torch.nn as nn
from torchvision import models

class ResNetSpoof(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)
    
if __name__ == "__main__":
    import torch

    model = ResNetSpoof()
    print("Model created")

    # Fake input [batch, channels, mel_bins, time_steps]
    x = torch.randn(2, 1, 80, 200)
    y = model(x)
    print("Output shape:", y.shape)