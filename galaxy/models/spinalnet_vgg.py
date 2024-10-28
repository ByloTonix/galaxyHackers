import torch
import torch.nn as nn
import torchvision.models as models


class SpinalNet_VGG(nn.Module):
    def __init__(self, num_ftrs, half_in_size, layer_width, num_class=2):
        super(SpinalNet_VGG, self).__init__()
        self.half_in_size = half_in_size

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_class), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0 : self.half_in_size])
        x2 = self.fc_spinal_layer2(
            torch.cat([x[:, self.half_in_size : 2 * self.half_in_size], x1], dim=1)
        )
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0 : self.half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(
            torch.cat([x[:, self.half_in_size : 2 * self.half_in_size], x3], dim=1)
        )

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x


def load_model(num_class=2):
    model_ft = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

    pretrained_weights = model_ft.features[0].weight.clone()

    new_features = nn.Sequential(*list(model_ft.features.children()))
    new_features[0] = nn.Conv2d(
        2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )

    # Inserting pretrained weights from first 2 channels into new layer
    with torch.no_grad():
        new_features[0].weight.data = nn.Parameter(pretrained_weights[:, :2])

    model_ft.features = new_features

    num_ftrs = model_ft.classifier[0].in_features
    half_in_size = round(num_ftrs / 2)
    layer_width = 512

    model_ft.classifier = SpinalNet_VGG(num_ftrs, half_in_size, layer_width, num_class)
    return model_ft
