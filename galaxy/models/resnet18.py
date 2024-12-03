import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18()
        # self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Cloning pretrained weights of old layer
        weight = self.resnet.conv1.weight.clone()

        # Defining new layer
        self.resnet.conv1 = nn.Conv2d(
            2, self.resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Inserting pretrained weights from first 2 channels into new layer
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(weight[:, :2])

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


def load_model(num_classes=2):
    model = ResNet18(num_classes=num_classes)
    return model
