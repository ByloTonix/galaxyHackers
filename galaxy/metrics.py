"""Script for model performance evaluation and metrics visualization."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from galaxy.config import settings
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def probabilities_hist(
    predictions_clusters: np.ndarray, 
    predictions_galaxies: np.ndarray, 
    predictions_stars: np.ndarray, 
    predictions_random: np.ndarray, 
    pdf: PdfPages
) -> None:
    """Plots histogram of prediction probabilities.

    Args:
        predictions_clusters (np.ndarray): Probabilities for the cluster class.
        predictions_non_clusters (np.ndarray): Probabilities for the non-cluster class.
        pdf (PdfPages): PDF file to save the plots.
    """
    bins = np.arange(0, 1.01, 0.05)
    plt.figure()
    plt.hist(predictions_clusters, bins, color="green", alpha=0.5, label="clusters")
    plt.hist(predictions_galaxies, bins, color="red", alpha=0.5, label="galaxies")
    plt.hist(predictions_stars, bins, color="blue", alpha=0.5, label="stars")
    plt.hist(predictions_random, bins, color="pink", alpha=0.5, label="random")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("Class prediction")
    pdf.savefig()
    plt.close()


def plot_roc_curve(pdf: PdfPages, predictions: pd.DataFrame) -> None:
    """Plots the ROC curve.

    Args:
        pdf (PdfPages): PDF file to save the plots.
        predictions (pd.DataFrame): DataFrame containing true labels and predicted probabilities.
    """
    fpr, tpr, _ = roc_curve(predictions.y_true, predictions.y_probs) #TODO: NEEDS CHANGE FOR MULTILABEL CLASSIFICATION
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label="")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate, FPR")
    plt.ylabel("True Positive Rate, TPR")
    plt.title("ROC curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    pdf.savefig()
    plt.close()


def plot_pr_curve(pdf: PdfPages, predictions: pd.DataFrame) -> float:
    """Plots the precision-recall curve.

    Args:
        pdf (PdfPages): PDF file to save the plots.
        predictions (pd.DataFrame): DataFrame containing true labels and predicted probabilities.

    Returns:
        float: Area under the precision-recall curve (PR AUC).
    """
    precisions, recalls, _ = precision_recall_curve(
        predictions.y_true, predictions.y_probs
    )
    # calculate Area Under the PR curve
    pr_auc = auc(recalls, precisions)
    plt.figure()
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    pdf.savefig()
    plt.close()
    return pr_auc


def plot_confusion_matrices(
    pdf: PdfPages, predictions: pd.DataFrame, classes: List[str]
):
    """Plots confusion matrices.

    Args:
        pdf (PdfPages): PDF file to save the plots.
        predictions (pd.DataFrame): DataFrame containing true and predicted labels.
        classes (List[str]): Class labels.

    Returns:
        Tuple[int, int, int, int]: Counts for TN, FP, FN, TP from the confusion matrix.
    """
    cm = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=range(len(classes)))
    weighted_cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 8))
    _ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.title("Confusion Matrix")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 8))
    _ = ConfusionMatrixDisplay(confusion_matrix=weighted_cm, display_labels=classes).plot()
    plt.title("Weighted Confusion Matrix")
    pdf.savefig()
    plt.close()
    return


def plot_red_shift(pdf, predictions: pd.DataFrame):

    red_shift_predictions = predictions.loc[predictions.red_shift.notna()]
    red_shift_predictions = red_shift_predictions.sort_values(by="red_shift")

    n_bins = 10
    # Create 10 equal-sized buckets based on red_shift
    red_shift_predictions["bucket"] = pd.qcut(
        red_shift_predictions["red_shift"], n_bins, duplicates="drop"
    )

    # Calculate recall for each bin
    recall_per_bin = red_shift_predictions.groupby("bucket", observed=False).apply(
        lambda x: recall_score(x["y_true"], x["y_pred"])
    )

    # Calculate proportions of red_shift_type within each bin
    proportions = (
        red_shift_predictions.groupby("bucket", observed=False)["red_shift_type"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    width = 0.3

    bars = []
    for i in range(proportions.shape[1]):
        bars.append(proportions.iloc[:, i] * recall_per_bin)

    bars[0].plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        ax=ax,
        color="skyblue",
        position=0,
        width=width,
        edgecolor="black",
    )
    for i in range(1, len(bars)):
        bars[i].plot(
            kind="bar",
            stacked=True,
            bottom=bars[i - 1],
            ax=ax,
            color=plt.cm.Paired(i),
            position=0,
            width=width,
            edgecolor="black",
        )

    bars = []
    for i in range(proportions.shape[1]):
        bars.append(proportions.iloc[:, i])

    bars[0].plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        ax=ax2,
        color="skyblue",
        position=1,
        width=width,
        edgecolor="black",
    )
    for i in range(1, len(bars)):
        bars[i].plot(
            kind="bar",
            stacked=True,
            bottom=bars[i - 1],
            ax=ax2,
            color=plt.cm.Paired(i),
            position=1,
            width=width,
            edgecolor="black",
        )

    plt.title("Recall by Red Shift Bins with Proportional Coloring by Red Shift Type")
    plt.xlabel("Red Shift Bin")
    plt.ylabel("Recall")
    plt.legend(
        proportions.columns,
        title="Red Shift Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def plot_loss_by_model(
    train_table_data: List[Tuple[int, float]], val_table_data: List[Tuple[int, float]], pdf: PdfPages
) -> None:
    """Plots loss curves for training and validation.

    Args:
        train_table_data (List[Tuple[int, float]]): Training loss data.
        val_table_data (List[Tuple[int, float]]): Validation loss data.
        pdf (PdfPages): PDF file to save the plots.
    """
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig.suptitle("Loss on train and validation")

    train_steps = [row[0] for row in train_table_data]
    train_losses = [row[1] for row in train_table_data]

    val_epochs = [row[0] for row in val_table_data]
    val_losses = [row[1] for row in val_table_data]

    ax1.plot(
        train_steps, train_losses, label="Train Loss (Steps)", marker=".", color="blue"
    )
    ax2.plot(
        val_epochs,
        val_losses,
        label="Validation Loss (Epochs)",
        marker=".",
        color="green",
    )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)

    pdf.savefig()
    plt.close()


def plot_accuracies_by_model(train_table_data, val_table_data, pdf):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig.suptitle("Accuracy on train and validation")

    train_steps = [row[0] for row in train_table_data]
    train_accuracies = [row[2] for row in train_table_data]

    val_epochs = [row[0] for row in val_table_data]
    val_accuracies = [row[2] for row in val_table_data]

    ax1.plot(train_steps, train_accuracies, label="train", marker=".", color="blue")
    ax2.plot(val_epochs, val_accuracies, label="valid", marker=".", color="green")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)

    pdf.savefig()
    plt.close()


def modelPerformance(
    model_name: str,
    optimizer_name: str,
    predictions: pd.DataFrame,
    classes: List[str],
    train_table_data: Optional[List[Tuple[int, float, float]]] = None,
    val_table_data: Optional[List[Tuple[int, float, float]]] = None,
    f_beta: float = 2.0,
) -> None:
    """
    Plots distributions of probabilities of classes, ROC and Precision-Recall curves, change of loss and accuracy throughout training,
    confusion matrix and its weighted version and saves them in .png files,
    counts accuracy, precision, recall, false positive rate and f1-score and saves them in .txt file

    Args:
        model_name (str): Name of the model.
        optimizer_name (str): Name of the optimizer.
        predictions (pd.DataFrame): DataFrame with true labels, predicted labels, and probabilities.
        classes (List[str]): Class labels.
        train_table_data (Optional[List[Tuple[int, float, float]]], optional): Training data for plotting. Defaults to None.
        val_table_data (Optional[List[Tuple[int, float, float]]], optional): Validation data for plotting. Defaults to None.
        f_beta (float, optional): Beta value for F-beta score calculation. Defaults to 2.0.
    """

    acc = accuracy_score(predictions.y_true, predictions.y_pred)
    precision = precision_score(predictions.y_true, predictions.y_pred, zero_division=0)
    recall = recall_score(predictions.y_true, predictions.y_pred)
    f1_measure = f1_score(predictions.y_true, predictions.y_pred)
    fbeta_measure = fbeta_score(predictions.y_true, predictions.y_pred, beta=f_beta)

    roc_auc = roc_auc_score(predictions.y_true, predictions.y_pred)

    model_path = Path(settings.METRICS_PATH, f"{model_name}_{optimizer_name}")
    os.makedirs(model_path, exist_ok=True)
    pdf = PdfPages(Path(model_path, f"{model_name}_metrics_plots.pdf"))
    # metrics_pdf_name = Path(model_path, 'metrics_plots.pdf')

    # plot probablities distribution
    probabilities_hist(predictions.y_probs, 
                       predictions.y_prob_class_0, #galaxy clusters
                       predictions.y_prob_class_1, #just galaxies
                       predictions.y_prob_class_2, #stars
                       predictions.y_prob_class_3, #random
                       pdf)

    # plot roc curve
    plot_roc_curve(pdf, predictions)

    # plot precision recall
    pr_auc = plot_pr_curve(pdf, predictions)

    # confusion matrices
    plot_confusion_matrices(pdf, predictions, classes)
    # fpr_measure = fp / (fp + tn) #TODO: should be adapted if wanted in multilabel classification

    if train_table_data is not None and val_table_data is not None:
        # change of loss throughout epochs
        plot_loss_by_model(train_table_data, val_table_data, pdf)

        # change of accuracies througout epochs
        plot_accuracies_by_model(train_table_data, val_table_data, pdf)

    # recall by red shift
    plot_red_shift(pdf, predictions)

    pdf.close()

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall (TPR)": recall,
        # "Fall-out (FPR)": fpr_measure,
        "PR AUC": pr_auc,
        "ROC AUC": roc_auc,
        "F-1 score": f1_measure,
        "Beta": f_beta,
        "F-beta score": fbeta_measure,
    }

    with open(Path(model_path, "metrics.json"), "w") as file:
        json.dump(metrics, file)


def combine_metrics(selected_models: List[Tuple[str, Any]], optimizer_name: str) -> pd.DataFrame:
    """Combines metrics for all selected models into a single CSV file.

    Args:
        selected_models (List[Tuple[str, Any]]): List of selected models.
        optimizer_name (str): Name of the optimizer.

    Returns:
        pd.DataFrame: Combined metrics DataFrame.
    """
    for model_name, _ in selected_models:

        all_metrics = {}
        for model_name, _ in selected_models:
            combination = f"{model_name}_{optimizer_name}"
            metrics_path = Path(settings.METRICS_PATH, combination, "metrics.json")

            with open(metrics_path) as file:
                all_metrics[combination] = json.load(file)

        metrics_frame = pd.DataFrame(all_metrics).T
        metrics_frame.index.name = "Combination"

        metrics_frame.to_csv(Path(settings.METRICS_PATH, "metrics.csv"))

    return metrics_frame
