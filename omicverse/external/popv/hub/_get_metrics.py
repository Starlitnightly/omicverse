import json
import os

import pandas as pd
from anndata import AnnData
from sklearn.metrics import f1_score


def _dataframe_to_markdown(df):
    # Create the header
    pretty_columns = [
        (
            "Consensus Prediction"
            if col == "popv_prediction"
            else (col[4:-11].replace("_", " ").capitalize() if col.startswith("popv") else col)
        )
        for col in df.columns
    ]
    header = "| Cell Type | " + " | ".join(pretty_columns) + " |"
    separator = "| --- | " + " | ".join("---" for _ in df.columns) + " |"

    # Format values and create rows, including the index
    rows = "\n".join(
        "| "
        + " | ".join([str(index)] + [f"{value:.2f}" if not isinstance(value, int) else f"{value}" for value in row])
        + " |"
        for index, row in zip(df.index, df.values, strict=True)
    )

    # Combine header, separator, and rows
    return f"{header}\n{separator}\n{rows}"


def create_criticism_report(
    adata: AnnData | None = None,
    label_key: str | None = "_reference_labels_annotation",
    save_folder: str | None = None,
) -> dict:
    """
    Compute and store accuracy metrics for a model.

    Parameters
    ----------
    adata
        AnnData to compute metrics on.
    label_key
        Key in adata.obs to use as cell type labels. If None, will use the original label key from
        the model.
    save_folder
        Path to folder for storing the metrics. Preferred to store in save_path folder of model.
    """
    predictions = adata.uns["prediction_keys"] + ["popv_prediction"]
    annotations = adata.obs[[*predictions, label_key]].astype("category")
    query_annotations = annotations.loc[adata.obs["_dataset"] == "query"]
    ref_annotations = annotations.loc[adata.obs["_dataset"] == "ref"]

    # Function to compute F1 score for each column
    def compute_f1_scores_per_label(df, ground_truth_col, labels, prediction_cols):
        results = pd.DataFrame(index=labels, columns=prediction_cols)
        # Compute F1 scores per label for each prediction column
        results = pd.DataFrame(index=labels, columns=["N cells", *prediction_cols])
        # Compute the cell count for each label
        label_counts = df[ground_truth_col].value_counts()
        results["N cells"] = [label_counts.get(label, 0) for label in labels]
        for col in prediction_cols:
            # Compare each label using vectorized logic
            for label in labels:
                y_true = df[ground_truth_col] == label
                y_pred = df[col] == label
                results.loc[label, col] = f1_score(y_true, y_pred)

        return results

    # Compute F1 scores
    labels = adata.obs[label_key].value_counts().index
    f1_scores_per_label_query = compute_f1_scores_per_label(query_annotations, label_key, labels, predictions)
    f1_scores_per_label_ref = compute_f1_scores_per_label(ref_annotations, label_key, labels, predictions)
    md_query_accuracy = _dataframe_to_markdown(f1_scores_per_label_query)
    md_ref_accuracy = _dataframe_to_markdown(f1_scores_per_label_ref)

    markdown_dict = {
        "query_accuracy": md_query_accuracy,
        "ref_accuracy": md_ref_accuracy,
    }

    save_path = os.path.join(save_folder, "accuracies.json")

    with open(save_path, "w") as f:
        json.dump(markdown_dict, f, indent=4)
