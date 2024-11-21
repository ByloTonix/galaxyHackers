import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Output: 32 x 112 x 112
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Output: 64 x 56 x 56
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # Output: 128 x 28 x 28
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # Output: 256 x 14 x 14
        # Global average pooling
        x = self.global_avg_pool(x)  # Output: 256 x 1 x 1
        x = x.view(-1, 256)
        # Dropout for regularization
        x = self.dropout(x)
        # Fully connected layer
        x = self.fc(x)
        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)
        return x
    
def load_model(num_classes=2):
    model = Baseline(num_classes=num_classes)
    return model

