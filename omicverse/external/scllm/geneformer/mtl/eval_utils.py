import os
import json
import torch
import pandas as pd

from .data import prepare_test_loader
from .model import GeneformerMultiTask

def evaluate_test_dataset(model, device, test_loader, cell_id_mapping, config):
    task_pred_labels = {task_name: [] for task_name in config["task_names"]}
    task_pred_probs = {task_name: [] for task_name in config["task_names"]}
    cell_ids = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, logits, _ = model(input_ids, attention_mask)
            for sample_idx in range(len(batch["input_ids"])):
                cell_id = cell_id_mapping[batch["cell_id"][sample_idx].item()]
                cell_ids.append(cell_id)
                for i, task_name in enumerate(config["task_names"]):
                    pred_label = torch.argmax(logits[i][sample_idx], dim=-1).item()
                    pred_prob = (
                        torch.softmax(logits[i][sample_idx], dim=-1).cpu().numpy()
                    )
                    task_pred_labels[task_name].append(pred_label)
                    task_pred_probs[task_name].append(pred_prob)

    # Save test predictions with cell IDs and probabilities to CSV
    test_results_dir = config["results_dir"]
    os.makedirs(test_results_dir, exist_ok=True)
    test_preds_file = os.path.join(test_results_dir, "test_preds.csv")

    rows = []
    for sample_idx in range(len(cell_ids)):
        row = {"Cell ID": cell_ids[sample_idx]}
        for task_name in config["task_names"]:
            row[f"{task_name} Prediction"] = task_pred_labels[task_name][sample_idx]
            row[f"{task_name} Probabilities"] = ",".join(
                map(str, task_pred_probs[task_name][sample_idx])
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(test_preds_file, index=False)
    print(f"Test predictions saved to {test_preds_file}")


def load_and_evaluate_test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, cell_id_mapping, num_labels_list = prepare_test_loader(config)
    model_directory = os.path.join(config["model_save_path"], "GeneformerMultiTask")
    hyperparams_path = os.path.join(model_directory, "hyperparameters.json")

    # Load the saved best hyperparameters
    with open(hyperparams_path, "r") as f:
        best_hyperparams = json.load(f)

    # Extract the task weights if present, otherwise set to None
    task_weights = best_hyperparams.get("task_weights", None)
    normalized_task_weights = task_weights if task_weights else []

    # Print the loaded hyperparameters
    print("Loaded hyperparameters:")
    for param, value in best_hyperparams.items():
        if param == "task_weights":
            print(f"normalized_task_weights: {value}")
        else:
            print(f"{param}: {value}")

    best_model_path = os.path.join(model_directory, "pytorch_model.bin")
    best_model = GeneformerMultiTask(
        config["pretrained_path"],
        num_labels_list,
        dropout_rate=best_hyperparams["dropout_rate"],
        use_task_weights=config["use_task_weights"],
        task_weights=normalized_task_weights,
    )
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    evaluate_test_dataset(best_model, device, test_loader, cell_id_mapping, config)
    print("Evaluation completed.")