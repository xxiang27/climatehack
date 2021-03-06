import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12*128*128, 256, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(256, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256*6*6, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=24 * 64 * 64)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.view(-1, 12 * 128 * 128) / 1024.0
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.view(-1, 24, 64, 64) * 1024.0
