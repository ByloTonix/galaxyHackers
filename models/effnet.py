import torch
import torch.nn as nn
import timm

class SpinalNet_EfficientNet(nn.Module):
    def __init__(self, num_ftrs, half_in_size, layer_width, num_class=1):
        super(SpinalNet_EfficientNet, self).__init__()
        self.half_in_size = half_in_size
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Linear(half_in_size, layer_width),
            nn.ReLU(inplace=True))
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.ReLU(inplace=True))
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.ReLU(inplace=True))
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.ReLU(inplace=True))
        self.fc_out = nn.Sequential(
            nn.Linear(layer_width * 4, num_class),
            nn.Softmax()
)

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:self.half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, self.half_in_size:2*self.half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:self.half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, self.half_in_size:2*self.half_in_size], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x

def load_model(num_class=1):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    num_ftrs = model.classifier.in_features
    half_in_size = round(num_ftrs / 2)
    layer_width = 200
    model.classifier = SpinalNet_EfficientNet(num_ftrs, half_in_size, layer_width, num_class)
    return model
