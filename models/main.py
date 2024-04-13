import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import os
import timm
import numpy as np

import argparse

from train import train, validate, continue_training
from plot_builder import plot_losses, plot_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

models = [
    ('ResNet18', timm.create_model('resnet18', pretrained=True)),
    ('ResNet50', timm.create_model('resnet50', pretrained=True)),
    ('EfficientNet', timm.create_model('efficientnet_b0', pretrained=True)),
    ('ViT', timm.create_model('vit_base_patch16_224', pretrained=True)),
    ('VGGNet', timm.create_model('vgg11', pretrained=True)),
    ('DenseNet', timm.create_model('densenet121', pretrained=True))
]

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--models', nargs='+', default=['ResNet18', 'ResNet50', 'EfficientNet', 'ViT', 'VGGNet', 'DenseNet'],
                    help='List of models to train (default: all)')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer (default: 0.0001)')
args = parser.parse_args()

selected_models = [(model_name, model) for model_name, model in models if model_name in args.models]

num_epochs = args.epochs
lr = args.lr

criterion = nn.CrossEntropyLoss()

results = {}
val_results = {}
for model_name, model in selected_models:
    optimizer = torch.optim.Adam(model.parameters(), lr)
    losses, epochs, accuracies = train(model, train_loader, criterion, optimizer, device, num_epochs)
    results[model_name] = {'losses': losses, 'epochs': epochs, 'accuracies': accuracies}

    val_losses, val_epochs, val_accuracies = validate(model, val_loader, criterion, device, num_epochs)
    val_results[model_name] = {'val_losses': val_losses, 'val_epochs': val_epochs, 'val_accuracies': val_accuracies}

# filepath = "/content/trained_models/ResNet_epoch_3.pth"
#
# for model_name, model in models:
#     losses, epochs, accuracies = continue_training(model, train_loader, criterion, optimizer, device, num_epochs, filepath)
#     results[model_name].update({'losses': losses, 'epochs': epochs, 'accuracies': accuracies})
#

os.makedirs('results', exist_ok=True)
for model_name, data in results.items():
    np.savez(f'results/{model_name}_results.npz', losses=data['losses'], epochs=data['epochs'], accuracies=data['accuracies'])

for model_name, data in val_results.items():
    np.savez(f'results/{model_name}_val_results.npz', losses=data['val_losses'], epochs=data['val_epochs'], accuracies=data['val_accuracies'])    

#TODO: make plot functions able to work with val_results
plot_losses(results, val_results)
plot_accuracies(results, val_results)

for model_name, data in results.items():
  losses = results[model_name]['losses']
  accuracies = results[model_name]['accuracies']

print("Stats for nerds")
print(f"Max loss: {np.max(losses)} \nAverage loss: {np.mean(losses)} \nMin loss: {np.min(losses)} \n\nMax accuracy: {np.max(accuracies)} \nAverage accuracy: {np.mean(accuracies)} \nMin accuracy: {np.min(accuracies)}")
print(f"\nLoss std: {np.std(losses)} \nAccuracy std: {np.std(accuracies)}")

