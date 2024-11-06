from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import data
import segmentation
import metrics
import argparse
import torch_optimizer as optimizer

from config import settings

import models.spinalnet_resnet as spinalnet_resnet
import models.effnet as effnet
import models.densenet as densenet
import models.spinalnet_vgg as spinalnet_vgg
import models.vitL16 as vitL16
import models.alexnet_vgg as alexnet_vgg
import models.resnet18 as resnet18

from train import Trainer
from data import DataPart

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, dataloaders = data.create_dataloaders()

train_loader = dataloaders[DataPart.TRAIN]
val_loader = dataloaders[DataPart.VALIDATE]
test_loader = dataloaders[DataPart.TEST_DR5]

all_models = [
    ("ResNet18", resnet18),
    ("EfficientNet", effnet),
    ("DenseNet", densenet),
    ("SpinalNet_ResNet", spinalnet_resnet),
    ("SpinalNet_VGG", spinalnet_vgg),
    ("ViTL16", vitL16),
    ("AlexNet_VGG", alexnet_vgg),
]

all_optimizers = [
    ("SGD", optim.SGD),
    ("Rprop", optim.Rprop),
    ("Adam", optim.Adam),
    ("NAdam", optim.NAdam),
    ("RAdam", optim.RAdam),
    ("AdamW", optim.AdamW),
    ("RMSprop", optim.RMSprop),
    ("DiffGrad", optimizer.DiffGrad),
]

parser = argparse.ArgumentParser(description="Model training")
parser.add_argument(
    "--models",
    nargs="+",
    default=[
        "ResNet18",
        "EfficientNet",
        "DenseNet",
        "SpinalNet_ResNet",
        "SpinalNet_VGG",
        "ViTL16",
        "AlexNet_VGG",
    ],
    help="List of models to train (default: all)",
)
parser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs to train (default: 5)"
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="Learning rate for optimizer (default: 0.0001)",
)
parser.add_argument(
    "--mm", type=float, default=0.9, help="Momentum for optimizer (default: 0.9)"
)
parser.add_argument(
    "--optimizer",
    choices=[name for name, _ in all_optimizers],
    default="Adam",
    help="Optimizer to use (default: Adam)",
)

args = parser.parse_args()

selected_models = [
    (model_name, model) for model_name, model in all_models if model_name in args.models
]

num_epochs = args.epochs
lr = args.lr
momentum = args.mm
optimizer_name = args.optimizer

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

results = {}
val_results = {}

classes = ("random", "clusters")

experiment = Experiment(
    api_key=settings.COMET_API_KEY,
    project_name="cluster-search",
    workspace=settings.COMET_WORKSPACE,
    auto_param_logging=False,
)

experiment.log_parameters(
    {
        "models": [name for name, _ in selected_models],
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "momentum": momentum,
        "optimizer": optimizer_name,
    }
)

for model_name, model in selected_models:

    model = model.load_model()
    optimizer_class = dict(all_optimizers)[optimizer_name]

    if optimizer_name in ["SGD", "RMSprop"]:
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    trainer.train(num_epochs)

    for step in range(trainer.global_step):
        experiment.log_metrics(
            {
                f"{model_name}_{optimizer_name}_train_loss": trainer.history[
                    "train_loss"
                ][step],
                f"{model_name}_{optimizer_name}_train_accuracy": trainer.history[
                    "train_acc"
                ][step],
            },
            step=step + 1,
        )

    for epoch in range(num_epochs):
        print(
            f"Epoch {epoch} - Val Loss: {trainer.history['val_loss'][epoch]}, Val Accuracy: {trainer.history['val_acc'][epoch]}"
        )
        experiment.log_metrics(
            {
                f"{model_name}_{optimizer_name}_val_loss": trainer.history["val_loss"][
                    epoch
                ],
                f"{model_name}_{optimizer_name}_val_accuracy": trainer.history[
                    "val_acc"
                ][epoch],
            },
            epoch=epoch,
        )

    train_table_data = [
        [step, trainer.history["train_loss"][step], trainer.history["train_acc"][step]]
        for step in range(trainer.global_step)
    ]
    val_table_data = [
        [epoch, trainer.history["val_loss"][epoch], trainer.history["val_acc"][epoch]]
        for epoch in range(num_epochs)
    ]

    experiment.log_table(
        filename=f"{model_name}_train_metrics.csv",
        tabular_data=train_table_data,
        headers=["Step", "Train Loss", "Train Accuracy"],
    )

    experiment.log_table(
        filename=f"{model_name}_val_metrics.csv",
        tabular_data=val_table_data,
        headers=["Epoch", "Validation Loss", "Validation Accuracy"],
    )

    predictions, *_ = trainer.test(test_loader)
    metrics.modelPerformance(model_name, 
                             optimizer_name, 
                             predictions, 
                             classes, 
                             train_table_data, 
                             val_table_data
                             )

    del model
    torch.cuda.empty_cache()

metrics.combine_metrics(selected_models, optimizer_name)

experiment.end()

for model_name, model in selected_models:
    segmentation.create_segmentation_plots(
        model, model_name, optimizer_name=optimizer_name
    )
