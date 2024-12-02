import argparse
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optimizer
from comet_ml import Experiment
from torch.optim.lr_scheduler import ExponentialLR

import data
import metrics
import models.alexnet_vgg as alexnet_vgg
import models.baseline as baseline
import models.densenet as densenet
import models.effnet as effnet
import models.resnet18 as resnet18
import models.spinalnet_resnet as spinalnet_resnet
import models.spinalnet_vgg as spinalnet_vgg
import models.vitL16 as vitL16
import segmentation
from config import settings

# import data.segmentation as segmentation
# import metrics.metrics as metrics
from data import DataPart
from train import Trainer


all_models = [
    ("Baseline", baseline),
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
    # ('Adagrad', optim.Adagrad),
    ("RMSprop", optim.RMSprop),
    # ('Adadelta', optim.Adadelta),
    ("DiffGrad", optimizer.DiffGrad),
    # ('LBFGS', optim.LBFGS)
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets, dataloaders = data.create_dataloaders()

train_loader = dataloaders[DataPart.TRAIN]
val_loader = dataloaders[DataPart.VALIDATE]
test_loader = dataloaders[DataPart.TEST_DR5]


parser = argparse.ArgumentParser(description="Model training")
parser.add_argument(
    "--models",
    nargs="+",
    default=[
        "Baseline",
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
parser.add_argument(
    "--segment", action="store_true", help="Run segmentation after training"
)

args = parser.parse_args()

selected_models = [
    (model_name, model) for model_name, model in all_models if model_name in args.models
]

num_epochs = args.epochs
lr = args.lr
momentum = args.mm
optimizer_name = args.optimizer

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
        "momentum": momentum,
        "optimizer": optimizer_name,
    }
)


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

results = {}
val_results = {}

classes = ("random", "clusters")

for model_name, model in selected_models:

    model = model.load_model()
    optimizer_class = dict(all_optimizers)[optimizer_name]

    if optimizer_name in ["SGD", "RMSprop"]:
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    scheduler = ExponentialLR(optimizer, gamma=0.8)

    trainer = Trainer(
        model_name=model_name,
        model=model,
        criterion=criterion,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        experiment=experiment,
    )

    trainer.find_lr(1e-6, 0.01, 70)

    try:
        trainer.train(num_epochs)

        train_table_data = trainer.train_table_data
        val_table_data = trainer.val_table_data

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

    finally:

        predictions, *_ = trainer.test(test_loader)
        metrics.modelPerformance(
            model_name,
            optimizer_name,
            predictions,
            classes,
            train_table_data,
            val_table_data,
        )

metrics.combine_metrics(selected_models, optimizer_name)
experiment.end()

del model
torch.cuda.empty_cache()
gc.collect()

if args.segment:
    for model_name, model in selected_models:
        segmentation.create_segmentation_plots(
            model, model_name, optimizer_name=optimizer_name
        )
