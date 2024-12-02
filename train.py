import os
from collections import defaultdict
from copy import deepcopy
from typing import Any
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from comet_ml import Experiment
from scipy.signal import savgol_filter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import sys

from config import settings


class Trainer:
    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        optimizer_name: str,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        experiment: Experiment,
        criterion: Any | None = None,
        lr_scheduler: LRScheduler | None = None,
        lr_scheduler_type: str = "per_epoch",
        batch_size: int = 128,
    ):

        self.model_name = model_name
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.experiment = experiment
        self.batch_size = batch_size

        self.train_table_data = []
        self.val_table_data = []

        self.history = defaultdict(list)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.model = self.model.to(self.device)

        self.global_step = 0
        self.cache = self.cache_states()

    def post_train_batch(self):
        # called after every train batch
        if self.lr_scheduler is not None and self.lr_scheduler_type == "per_batch":
            self.lr_scheduler.step()

    def post_val_batch(self):
        pass

    def post_train_stage(self):
        pass

    def post_val_stage(self):
        # called after every end of val stage (equals to epoch end)
        if self.lr_scheduler is not None and self.lr_scheduler_type == "per_epoch":
            self.lr_scheduler.step()

    def save_checkpoint(self):

        filename = f"best_weights_{self.model.__class__.__name__}_{self.optimizer.__class__.__name__}.pth"
        path = os.path.join(settings.BEST_MODELS_PATH, filename)

        torch.save(self.model.state_dict(), path)

    def log_metrics(
        self,
        loss,
        acc,
        mode: str = "train",
        step: int | None = None,
        epoch: int | None = None,
    ):

        loss_name = f"{self.model_name}_{self.optimizer_name}_{mode}_loss"
        acc_name = f"{self.model_name}_{self.optimizer_name}_{mode}_acc"

        metrics = {
            loss_name: loss,
            acc_name: acc,
        }

        if epoch is not None:
            self.experiment.log_metrics(metrics, epoch=epoch)
        elif step is not None:
            self.experiment.log_metrics(metrics, step=step)

        else:
            raise ValueError("No step or epoch given")

    def train(self, num_epochs: int):
        model = self.model
        optimizer = self.optimizer

        best_loss = float("inf")  # +inf

        for epoch in range(num_epochs):
            print(f"\nTraining: Epoch {epoch + 1}/{num_epochs}")

            torch.cuda.empty_cache()
            gc.collect()
            model.train()
            epoch_train_losses, epoch_train_accs = [], []

            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                *_, loss, acc = self.compute_all(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.post_train_batch()

                self.log_metrics(
                    loss=loss.item(), acc=acc, mode="train", step=self.global_step
                )

                epoch_train_losses.append(loss.item())
                epoch_train_accs.append(acc)

                self.global_step += 1

                sys.stdout.write(
                    f"\rBatch {batch_idx}/{len(self.train_dataloader)} | "
                    f"Loss: {np.mean(epoch_train_losses):.4f} | "
                    f"Accuracy: {np.mean(epoch_train_accs):.4f}"
                )
                sys.stdout.flush()

            print()

            train_loss = np.mean(epoch_train_losses)
            train_acc = np.mean(epoch_train_accs)
            self.train_table_data.append([epoch + 1, train_loss, train_acc])

            self.log_metrics(
                loss=train_loss, acc=train_acc, mode="train", epoch=epoch + 1
            )

            print(f"Validation: Epoch {epoch + 1}/{num_epochs}")

            model.eval()
            val_losses, val_accs = [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader, start=1):
                    *_, loss, acc = self.compute_all(batch)
                    val_losses.append(loss.item())
                    val_accs.append(acc)

                    sys.stdout.write(
                        f"\rBatch {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {np.mean(epoch_train_losses):.4f} | "
                        f"Accuracy: {np.mean(epoch_train_accs):.4f}"
                    )
                    sys.stdout.flush()

            print()

            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            self.post_val_stage()

            self.log_metrics(loss=val_loss, acc=val_acc, mode="val", epoch=epoch + 1)
            self.val_table_data.append([epoch + 1, val_loss, val_acc])

            print(
                f"Epoch {epoch + 1}/{num_epochs} Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_loss:
                self.save_checkpoint()
                best_loss = val_loss
                print(f"New best model saved with val_loss={val_loss:.4f}")

    def test(self, test_dataloader: DataLoader):
        torch.cuda.empty_cache()
        gc.collect()

        test_losses = []
        test_accs = []

        y_pred, y_probs, y_true, descriptions = (
            [],
            [],
            [],
            [],
        )  # y_true - the real class of object in the dataset
        y_negative_target_probs = []

        print("Testing...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader, start=1):
                logits, outputs, labels, loss, acc = self.compute_all(batch)

                test_losses.append(loss.item())
                test_accs.append(acc)

                y_probs.extend(logits[:, 1].data.cpu().numpy().ravel())
                y_negative_target_probs.extend(logits[:, 0].data.cpu().numpy().ravel())
                y_pred.extend(outputs.data.cpu().numpy().ravel())
                y_true.extend(labels.data.cpu().numpy().ravel())

                descriptions.append(pd.DataFrame(batch["description"]))

                sys.stdout.write(
                    f"\rBatch {batch_idx}/{len(test_dataloader)} | "
                    f"Loss: {np.mean(test_losses):.4f} | "
                    f"Accuracy: {np.mean(test_accs):.4f}"
                )

                sys.stdout.flush()

        predictions = pd.concat(descriptions).reset_index(drop=True)
        predictions["y_pred"] = y_pred
        predictions["y_probs"] = y_probs
        predictions["y_negative_probs"] = y_negative_target_probs
        predictions["y_true"] = y_true

        return (
            predictions,
            test_losses,
            test_accs,
        )

    def compute_all(
        self, batch
    ):  # удобно сделать функцию, в которой вычисляется лосс по пришедшему батчу
        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)
        logits = self.model(x)

        assert self.criterion is not None

        loss = self.criterion(logits[:, 1], y.float())

        assert logits.shape[1] == 2, logits.shape

        outputs = logits.argmax(axis=1)
        acc = (outputs == y).float().mean().cpu().numpy()

        return logits, outputs, y, loss, acc

    def cache_states(self):
        cache_dict = {
            "model_state": deepcopy(self.model.state_dict()),
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
        }

        return cache_dict

    def rollback_states(self):
        self.model.load_state_dict(self.cache["model_state"])
        self.optimizer.load_state_dict(self.cache["optimizer_state"])

    def find_lr(
        self,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        num_lrs: int = 20,
        smoothing_window=30,
        smooth_beta: float = 0.8,
    ) -> dict:
        lrs = np.geomspace(start=min_lr, stop=max_lr, num=num_lrs)
        logs = {"lr": [], "loss": [], "avg_loss": []}
        avg_loss = None
        model, optimizer = self.model, self.optimizer

        model.train()
        print("Finding optimal learning rate...")
        for idx, (lr, batch) in enumerate(zip(lrs, self.train_dataloader)):
            if idx >= num_lrs:
                break  # Stop after num_lrs steps

            # Apply new lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Train step
            *_, loss, _ = self.compute_all(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.cpu().detach().numpy()

            # Calculate smoothed loss
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss

            # Store values into logs
            logs["lr"].append(lr)
            logs["avg_loss"].append(avg_loss)
            logs["loss"].append(loss)

            # Print progress in one line
            sys.stdout.write(
                f"\rStep {idx + 1}/{num_lrs} | LR: {lr:.2E} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}"
            )
            sys.stdout.flush()

        print("\nFinished finding learning rate.")

        # Compute the logarithm of learning rates
        log_lrs = np.log10(logs["lr"])

        smoothed_losses = savgol_filter(
            logs["loss"], window_length=smoothing_window, polyorder=2
        )

        # Compute the derivative of the smoothed loss with respect to log_lr
        loss_derivatives = np.gradient(smoothed_losses, log_lrs)

        # Find the index where the derivative is minimum (most negative)
        optimal_idx = np.argmin(loss_derivatives)
        optimal_lr = logs["lr"][optimal_idx]

        logs.update({key: np.array(val) for key, val in logs.items()})

        plt.figure(figsize=(10, 6))

        plt.plot(logs["lr"], logs["loss"], label="Loss")
        plt.plot(logs["lr"], smoothed_losses, label="Smoothed loss")
        plt.axvline(
            x=optimal_lr,
            color="r",
            linestyle="--",
            label=f"Optimal LR: {optimal_lr:.2E}",
        )
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)

        # Mark the optimal learning rate
        # plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2E}')
        plt.legend()
        plt.show()
        self.rollback_states()

        self.optimizer.lr = optimal_lr

        return optimal_lr


class Predictor:

    def __init__(self, model: nn.Module, device):

        self.model = model
        self.model.eval()

        self.device = device

    def predict(self, dataloader: DataLoader):

        y_pred, y_prob, descriptions = [], [], []

        print("Predicting...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, start=1):
                logits, outputs = self.compute_all(batch)

                y_pred.extend(outputs.data.cpu().numpy().ravel())
                y_prob.extend(logits[:, 1].data.cpu().numpy().ravel())
                descriptions.append(pd.DataFrame(batch["description"]))

                sys.stdout.write(
                    f"\rBatch {batch_idx}/{len(dataloader)} | Predictions collected: {len(y_pred)}"
                )
                sys.stdout.flush()
                torch.cuda.empty_cache()
                gc.collect()

        print("\nPrediction complete.")

        predictions = pd.DataFrame(
            np.array([np.array(y_pred), np.array(y_prob)]).T,
            columns=["y_pred", "y_prob"],
        ).reset_index(drop=True)

        description_frame = pd.concat(descriptions).reset_index(drop=True)

        predictions = pd.concat([predictions, description_frame], axis=1)
        predictions = predictions.set_index("idx")
        predictions.index = predictions.index.astype(int)

        return predictions

    def compute_all(self, batch):

        x = batch["image"].to(self.device)

        logits = self.model(x)

        outputs = logits.argmax(axis=1)

        return logits, outputs
