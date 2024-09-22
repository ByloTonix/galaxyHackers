import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, fbeta_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pandas as pd

from config import settings

def probabilities_hist(predictions_clusters, predictions_non_clusters, pdf):
    bins = np.arange(0, 1.01, 0.05)
    plt.figure()
    plt.hist(predictions_clusters, bins, color='green', alpha=0.5, label='clusters')
    plt.hist(predictions_non_clusters, bins, color='red', alpha=0.5, label='non-clusters')
    plt.legend(loc='upper right')
    plt.title('Class prediction')
    # plt.savefig(metrics_pdf_name, format="pdf")
    pdf.savefig()
    plt.close()

# NOT YET IMPLEMENTED
def plot_loss_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))

    # available flags for customizing: linestyle="--", linewidth=2, marker,
    plt.plot(result['epochs'], result['losses'], label="train", marker=".")
    plt.plot(val_result['val_epochs'], val_result['val_losses'], label="valid", marker="*")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)
    plt.close()


def plot_roc_curve(pdf, predictions: pd.DataFrame):
    fpr, tpr, _ = roc_curve(predictions.y_true, predictions.y_probs)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate, FPR')
    plt.ylabel('True Positive Rate, TPR')
    plt.title('ROC curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig(metrics_pdf_name, format="pdf")
    pdf.savefig()
    plt.close()

def plot_pr_curve(pdf, predictions: pd.DataFrame):
    precisions, recalls, _ = precision_recall_curve(predictions.y_true, predictions.y_probs)
    # calculate Area Under the PR curve
    pr_auc = auc(recalls, precisions)
    plt.figure()
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig(metrics_pdf_name, format="pdf")
    pdf.savefig()
    plt.close()
    return pr_auc

def plot_confusion_matrices(pdf, predictions: pd.DataFrame, classes):
    cm = confusion_matrix(predictions.y_true, predictions.y_pred)
    tn, fp, fn, tp = cm.ravel()
    e_00, e_11 = cm[0, 0] / (cm[0, 0] + cm[0, 1]), cm[1, 1] / (cm[1, 0] + cm[1, 1])
    weighted_cm = np.array([[e_00, 1 - e_00], [1 - e_11, e_11]])

    plt.figure()
    _ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    pdf.savefig()
    plt.close()

    plt.figure()
    _ = ConfusionMatrixDisplay(confusion_matrix=weighted_cm, display_labels=classes).plot()
    # plt.savefig(metrics_pdf_name, format="pdf")
    pdf.savefig()
    plt.close()
    return tn, fp, fn, tp

def plot_red_shift(pdf, predictions: pd.DataFrame):
    red_shift_predictions = predictions.loc[predictions.red_shift.notna()]
    red_shift_predictions = red_shift_predictions.sort_values(by='red_shift')

    n_bins = 10
    # Create 10 equal-sized buckets based on red_shift
    red_shift_predictions['bucket'] = pd.qcut(red_shift_predictions['red_shift'], n_bins)

    # Calculate recall for each bin
    recall_per_bin = red_shift_predictions.groupby('bucket').apply(lambda x: recall_score(x['y_true'], x['y_pred']))

    # Calculate proportions of red_shift_type within each bin
    proportions = red_shift_predictions.groupby('bucket')['red_shift_type'].value_counts(normalize=True).unstack().fillna(0)

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    ax2 = ax.twinx() 

    width = 0.3

    bars = []
    for i in range(proportions.shape[1]):
        bars.append(proportions.iloc[:, i] * recall_per_bin)

    bars[0].plot(kind='bar', stacked=True, figsize=(10, 6), ax=ax, color='skyblue', position=0, width=width, edgecolor="black")
    for i in range(1, len(bars)):
        bars[i].plot(kind='bar', stacked=True, bottom=bars[i-1], ax=ax, color=plt.cm.Paired(i), position=0, width=width, edgecolor="black")

    bars = []
    for i in range(proportions.shape[1]):
        bars.append(proportions.iloc[:, i])

    bars[0].plot(kind='bar', stacked=True, figsize=(10, 6), ax=ax2, color='skyblue', position=1, width=width,edgecolor='black')
    for i in range(1, len(bars)):
        bars[i].plot(kind='bar', stacked=True, bottom=bars[i-1], ax=ax2, color=plt.cm.Paired(i), position=1, width=width,edgecolor='black')

    plt.title('Recall by Red Shift Bins with Proportional Coloring by Red Shift Type')
    plt.xlabel('Red Shift Bin')
    plt.ylabel('Recall')
    plt.legend(proportions.columns, title='Red Shift Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # plt.savefig(metrics_pdf_name, format="pdf")
    pdf.savefig(fig)
    plt.close()


# NOT YET IMPLEMENTED
def plot_accuracies_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(result['epochs'], result['accuracies'], label="train", marker=".")
    plt.plot(val_result['val_epochs'], val_result['val_accuracies'], label="valid", marker="*")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)
    plt.close()


def modelPerformance(model_name, optimizer_name,
                     predictions: pd.DataFrame,
                     classes, 
                     f_beta = 2,
                    #  result, val_result
                     ):
    '''
    Plots distributions of probabilities of classes, ROC and Precision-Recall curves, change of loss and accuracy throughout training,
    confusion matrix and its weighted version and saves them in .png files,
    counts accuracy, precision, recall, false positive rate and f1-score and saves them in .txt file
    '''
                         
    acc = accuracy_score(predictions.y_true, predictions.y_pred)
    precision = precision_score(predictions.y_true, predictions.y_pred)
    recall = recall_score(predictions.y_true, predictions.y_pred)
    f1_measure = f1_score(predictions.y_true, predictions.y_pred)
    fbeta_measure = fbeta_score(predictions.y_true, predictions.y_pred, beta=f_beta)

    roc_auc = roc_auc_score(predictions.y_true, predictions.y_pred)

    model_path = Path(settings.METRICS_PATH, f"{model_name}_{optimizer_name}")
    os.makedirs(model_path, exist_ok=True)
    pdf = PdfPages(Path(model_path, 'metrics_plots.pdf'))
    # metrics_pdf_name = Path(model_path, 'metrics_plots.pdf')

    # plot probablities distribution
    probabilities_hist(predictions.y_probs, predictions.y_negative_probs, pdf)

    # plot roc curve
    plot_roc_curve(pdf, predictions)

    # plot precision recall
    pr_auc = plot_pr_curve(pdf, predictions)

    # confusion matrices
    tn, fp, fn, tp = plot_confusion_matrices(pdf, predictions, classes)
    fpr_measure = fp/(fp+tn)

    # recall by red shift
    plot_red_shift(pdf, predictions)

    pdf.close()

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

    with open(Path(model_path, 'metrics.json'), "w") as file:
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