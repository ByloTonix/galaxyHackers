import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from galaxy.config import settings


def probabilities_hist(predictions_clusters, predictions_non_clusters, model_path):
    bins = np.arange(0, 1.01, 0.05)
    plt.hist(predictions_clusters, bins, color="green", alpha=0.5, label="clusters")
    plt.hist(
        predictions_non_clusters, bins, color="red", alpha=0.5, label="non-clusters"
    )
    plt.legend(loc="upper right")
    plt.title("Class prediction")
    plt.savefig(Path(model_path, "probabilities_hist.png"))
    plt.close()


def plot_loss_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))

    # available flags for customizing: linestyle="--", linewidth=2, marker,
    plt.plot(result["epochs"], result["losses"], label="train", marker=".")
    plt.plot(
        val_result["val_epochs"], val_result["val_losses"], label="valid", marker="*"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(path)
    plt.close()


def plot_accuracies_by_model(
    model_name: str, result: dict, val_result: dict, path: str
):
    plt.figure(figsize=(10, 6))
    plt.plot(result["epochs"], result["accuracies"], label="train", marker=".")
    plt.plot(
        val_result["val_epochs"],
        val_result["val_accuracies"],
        label="valid",
        marker="*",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(path)
    plt.close()


def modelPerformance(
    model_name,
    optimizer_name,
    predictions: pd.DataFrame,
    classes,
    f_beta=2,
    #  result, val_result
):
    """
    Plots distributions of probabilities of classes, ROC and Precision-Recall curves, change of loss and accuracy throughout training,
    confusion matrix and its weighted version and saves them in .png files,
    counts accuracy, precision, recall, false positive rate and f1-score and saves them in .txt file
    """

    acc = accuracy_score(predictions.y_true, predictions.y_pred)
    precision = precision_score(predictions.y_true, predictions.y_pred)
    recall = recall_score(predictions.y_true, predictions.y_pred)
    f1_measure = f1_score(predictions.y_true, predictions.y_pred)
    fbeta_measure = fbeta_score(predictions.y_true, predictions.y_pred, beta=f_beta)

    cm = confusion_matrix(predictions.y_true, predictions.y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_measure = fp / (fp + tn)

    roc_auc = roc_auc_score(predictions.y_true, predictions.y_pred)

    model_path = Path(settings.METRICS_PATH, f"{model_name}_{optimizer_name}")
    os.makedirs(model_path, exist_ok=True)

    # plot probablities distribution
    probabilities_hist(predictions.y_probs, predictions.y_negative_probs, model_path)

    # plot roc curve

    fpr, tpr, _ = roc_curve(predictions.y_true, predictions.y_probs)
    plt.plot(fpr, tpr, linewidth=2, label="")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate, FPR")
    plt.ylabel("True Positive Rate, TPR")
    plt.title("ROC curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(Path(model_path, "roc_curve.png"))
    plt.close()
    # plot precision recall

    precisions, recalls, _ = precision_recall_curve(
        predictions.y_true, predictions.y_probs
    )
    # Step 6: Calculate Area Under the PR curve.
    pr_auc = auc(recalls, precisions)
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(Path(model_path, "precision_recall_curve.png"))
    plt.close()

    # confusion matrices
    e_00, e_11 = cm[0, 0] / (cm[0, 0] + cm[0, 1]), cm[1, 1] / (cm[1, 0] + cm[1, 1])
    weighted_cm = np.array([[e_00, 1 - e_00], [1 - e_11, e_11]])

    _ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.savefig(Path(model_path, "confusion_matrix.png"))
    plt.close()

    _ = ConfusionMatrixDisplay(
        confusion_matrix=weighted_cm, display_labels=classes
    ).plot()
    plt.savefig(Path(model_path, "weighted_confusion_matrix.png"))
    plt.close()

    red_shift_predictions = predictions.loc[predictions.red_shift.notna()]
    red_shift_predictions = red_shift_predictions.sort_values(by="red_shift")

    n_bins = 10
    # Assume df is your dataframe
    # Create 10 equal-sized buckets based on red_shift
    red_shift_predictions["bucket"] = pd.qcut(
        red_shift_predictions["red_shift"], n_bins
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

    fig = plt.figure()  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

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
    plt.savefig(Path(model_path, "redshift_recall.png"))
    plt.close()

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall (TPR)": recall,
        "Fall-out (FPR)": fpr_measure,
        "PR AUC": pr_auc,
        "ROC AUC": roc_auc,
        "F-1 score": f1_measure,
        "Beta": f_beta,
        "F-beta score": fbeta_measure,
    }

    with open(Path(model_path, "metrics.json"), "w") as file:
        json.dump(metrics, file)


def combine_metrics(selected_models: list, optimizer_name):
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
