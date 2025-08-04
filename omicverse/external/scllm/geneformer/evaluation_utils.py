import logging
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from sklearn import preprocessing
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from tqdm.auto import trange

from .emb_extractor import make_colorbar

logger = logging.getLogger(__name__)


def preprocess_classifier_batch(cell_batch, max_len, label_name, gene_token_dict):
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    def pad_label_example(example):
        example[label_name] = np.pad(
            example[label_name],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=-100,
        )
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=gene_token_dict.get("<pad>"),
        )
        example["attention_mask"] = (
            example["input_ids"] != gene_token_dict.get("<pad>")
        ).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch


# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if rem == 0:
        return N
    else:
        return N - rem


def vote(logit_list):
    m = max(logit_list)
    logit_list.index(m)
    indices = [i for i, x in enumerate(logit_list) if x == m]
    if len(indices) > 1:
        return "tie"
    else:
        return indices[0]


def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def classifier_predict(model, classifier_type, evalset, forward_batch_size, gene_token_dict):
    if classifier_type == "gene":
        label_name = "labels"
    elif classifier_type == "cell":
        label_name = "label"

    predict_logits = []
    predict_labels = []
    model.eval()

    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible

    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])

    disable_progress_bar()  # disable progress bar for preprocess_classifier_batch mapping
    for i in trange(0, evalset_len, forward_batch_size):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(
            batch_evalset, max_evalset_len, label_name, gene_token_dict
        )
        padded_batch.set_format(type="torch")

        # For datasets>=4.0.0, convert to dict to avoid format issues
        if int(datasets.__version__.split(".")[0]) >= 4:
            padded_batch = padded_batch[:]
        
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch[label_name]
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_batch.to("cuda"),
                attention_mask=attn_msk_batch.to("cuda"),
                labels=label_batch.to("cuda"),
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]

    enable_progress_bar()
    logits_by_cell = torch.cat(predict_logits)
    last_dim = len(logits_by_cell.shape) - 1
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[last_dim])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [
        item
        for item in list(zip(all_logits.tolist(), all_labels.tolist()))
        if item[1] != -100
    ]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    return y_pred, y_true, logits_list


def get_metrics(y_pred, y_true, logits_list, num_classes, labels):
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(labels))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    roc_metrics = None  # roc metrics not reported for multiclass
    if num_classes == 2:
        y_score = [py_softmax(item)[1] for item in logits_list]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_wt = len(tpr)
        roc_auc = auc(fpr, tpr)
        roc_metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "interp_tpr": interp_tpr,
            "auc": roc_auc,
            "tpr_wt": tpr_wt,
        }
    return conf_mat, macro_f1, acc, roc_metrics


# get cross-validated mean and sd metrics
def get_cross_valid_roc_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count / sum(all_tpr_wt) for count in all_tpr_wt]
    all_weighted_tpr = [a * b for a, b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a * b for a, b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc - roc_auc) ** 2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd


# plot ROC curve
def plot_ROC(roc_metric_dict, model_style_dict, title, output_dir, output_prefix):
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    sns.set(font_scale=2)
    sns.set_style("white")
    lw = 3
    for model_name in roc_metric_dict.keys():
        mean_fpr = roc_metric_dict[model_name]["mean_fpr"]
        mean_tpr = roc_metric_dict[model_name]["mean_tpr"]
        color = model_style_dict[model_name]["color"]
        linestyle = model_style_dict[model_name]["linestyle"]
        if "roc_auc" not in roc_metric_dict[model_name].keys():
            all_roc_auc = roc_metric_dict[model_name]["all_roc_auc"]
            label = f"{model_name} (AUC {all_roc_auc:0.2f})"
        else:
            roc_auc = roc_metric_dict[model_name]["roc_auc"]
            roc_auc_sd = roc_metric_dict[model_name]["roc_auc_sd"]
            if len(roc_metric_dict[model_name]["all_roc_auc"]) > 1:
                label = f"{model_name} (AUC {roc_auc:0.2f} $\pm$ {roc_auc_sd:0.2f})"
            else:
                label = f"{model_name} (AUC {roc_auc:0.2f})"
        plt.plot(
            mean_fpr, mean_tpr, color=color, linestyle=linestyle, lw=lw, label=label
        )

    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    output_file = (Path(output_dir) / f"{output_prefix}_roc").with_suffix(".pdf")
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


# plot confusion matrix
def plot_confusion_matrix(
    conf_mat_df, title, output_dir, output_prefix, custom_class_order
):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {"axes.grid": False})
    if custom_class_order is not None:
        conf_mat_df = conf_mat_df.reindex(
            index=custom_class_order, columns=custom_class_order
        )
    display_labels = generate_display_labels(conf_mat_df)
    conf_mat = preprocessing.normalize(conf_mat_df.to_numpy(), norm="l1")
    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=display_labels
    )
    display.plot(cmap="Blues", values_format=".2g")
    plt.title(title)
    plt.show()

    output_file = (Path(output_dir) / f"{output_prefix}_conf_mat").with_suffix(".pdf")
    display.figure_.savefig(output_file, bbox_inches="tight")


def generate_display_labels(conf_mat_df):
    display_labels = []
    i = 0
    for label in conf_mat_df.index:
        display_labels += [f"{label}\nn={conf_mat_df.iloc[i,:].sum():.0f}"]
        i = i + 1
    return display_labels


def plot_predictions(predictions_df, title, output_dir, output_prefix, kwargs_dict):
    sns.set(font_scale=2)
    plt.figure(figsize=(10, 10), dpi=150)
    label_colors, label_color_dict = make_colorbar(predictions_df, "true")
    predictions_df = predictions_df.drop(columns=["true"])
    predict_colors_list = [label_color_dict[label] for label in predictions_df.columns]
    predict_label_list = [label for label in predictions_df.columns]
    predict_colors = pd.DataFrame(
        pd.Series(predict_colors_list, index=predict_label_list), columns=["predicted"]
    )

    default_kwargs_dict = {
        "row_cluster": False,
        "col_cluster": False,
        "row_colors": label_colors,
        "col_colors": predict_colors,
        "linewidths": 0,
        "xticklabels": False,
        "yticklabels": False,
        "center": 0,
        "cmap": "vlag",
    }

    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
    g = sns.clustermap(predictions_df, **default_kwargs_dict)

    plt.setp(g.ax_row_colors.get_xmajorticklabels(), rotation=45, ha="right")

    for label_color in list(label_color_dict.keys()):
        g.ax_col_dendrogram.bar(
            0, 0, color=label_color_dict[label_color], label=label_color, linewidth=0
        )

        g.ax_col_dendrogram.legend(
            title=f"{title}",
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 1),
            facecolor="white",
        )

    output_file = (Path(output_dir) / f"{output_prefix}_pred").with_suffix(".pdf")
    plt.savefig(output_file, bbox_inches="tight")
