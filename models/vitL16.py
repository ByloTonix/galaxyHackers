import timm
import torch
import torch.nn as nn


class Sigm(nn.Module):
    def __init__(self, num_classes):
        super(Sigm, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(768, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc_out(x)
        return x


def load_model():
    model_ft = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=2
    )
    model_ft.head = Sigm(num_classes=2)

    pretrained_weights = model_ft.patch_embed.proj.weight.clone()

    model_ft.patch_embed.proj = nn.Conv2d(
        2, 768, kernel_size=(16, 16), stride=(16, 16)
    )

    # Inserting pretrained weights from first 2 channels into new layer
    with torch.no_grad():
        model_ft.patch_embed.proj.weight.data[:, :2] = pretrained_weights[:, :2]

    return model_ft
