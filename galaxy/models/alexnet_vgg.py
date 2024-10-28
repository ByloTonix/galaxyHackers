import torch
import torch.nn as nn
from torchvision import models


class AlexNet_VGG(nn.Module):
    def __init__(self, num_ftrs, num_class=2):
        super(AlexNet_VGG, self).__init__()
        self.num_ftrs = num_ftrs

        VGG_fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
            nn.Softmax(dim=1),
        )

        self.classifier = VGG_fc

    def forward(self, x):
        x = self.classifier(x)
        return x


def load_model(num_class=2):

    model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    pretrained_weights = model_ft.features[0].weight.clone()

    new_features = nn.Sequential(*list(model_ft.features.children()))
    new_features[0] = nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2)

    # Inserting pretrained weights from first 2 channels into new layer
    with torch.no_grad():
        new_features[0].weight.data = nn.Parameter(pretrained_weights[:, :2])

    model_ft.features = new_features

    num_ftrs = model_ft.classifier[1].in_features
    half_in_size = round(num_ftrs / 2)

    model_ft.classifier = AlexNet_VGG(num_ftrs, num_class=num_class)

    return model_ft
