from typing import Dict, List, Optional, Union
import json
import os
import pickle
import random
import torch
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from transformers import AutoConfig, BertConfig, BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from contextlib import contextmanager


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_wandb(config):
    if config.get("use_wandb", False):
        import wandb
        wandb.init(
            project=config.get("wandb_project", "geneformer_multitask"),
            name=config.get("run_name", "experiment"),
            config=config,
            reinit=True,
        )


def create_model(config, num_labels_list, device, is_distributed=False, local_rank=0):
    """Create and initialize the model based on configuration."""
    from .model import GeneformerMultiTask
    
    model = GeneformerMultiTask(
        config["pretrained_path"],
        num_labels_list,
        dropout_rate=config.get("dropout_rate", 0.1),
        use_task_weights=config.get("use_task_weights", False),
        task_weights=config.get("task_weights", None),
        max_layers_to_freeze=config.get("max_layers_to_freeze", 0),
        use_attention_pooling=config.get("use_attention_pooling", False),
    )
    
    # Move model to device
    model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    return model


def setup_optimizer_and_scheduler(model, config, total_steps):
    """Set up optimizer and learning rate scheduler."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=config["learning_rate"],
        eps=config.get("adam_epsilon", 1e-8)
    )
    
    # Prepare scheduler
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    scheduler_map = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup
    }
    
    scheduler_fn = scheduler_map.get(config["lr_scheduler_type"])
    if not scheduler_fn:
        raise ValueError(f"Unsupported scheduler type: {config['lr_scheduler_type']}")
    
    scheduler = scheduler_fn(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    return optimizer, scheduler


def save_model(model, model_save_directory):
    """Save model weights and configuration."""
    os.makedirs(model_save_directory, exist_ok=True)

    # Handle DDP model
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model
    
    model_state_dict = model_to_save.state_dict()

    model_save_path = os.path.join(model_save_directory, "pytorch_model.bin")
    torch.save(model_state_dict, model_save_path)

    # Save the model configuration
    model_to_save.config.to_json_file(os.path.join(model_save_directory, "config.json"))

    print(f"Model and configuration saved to {model_save_directory}")


def save_hyperparameters(model_save_directory, hyperparams):
    """Save hyperparameters to a JSON file."""
    hyperparams_path = os.path.join(model_save_directory, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f)
    print(f"Hyperparameters saved to {hyperparams_path}")


def calculate_metrics(labels=None, preds=None, task_data=None, metric_type="task_specific", return_format="dict"):
    if metric_type == "single":
        # Calculate metrics for a single task
        if labels is None or preds is None:
            raise ValueError("Labels and predictions must be provided for single task metrics")
        
        task_name = None
        if isinstance(labels, dict) and len(labels) == 1:
            task_name = list(labels.keys())[0]
            labels = labels[task_name]
            preds = preds[task_name]
            
        f1 = f1_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        
        if return_format == "tuple":
            return f1, accuracy
        
        result = {"f1": f1, "accuracy": accuracy}
        if task_name:
            return {task_name: result}
        return result
        
    elif metric_type == "task_specific":
        # Calculate metrics for multiple tasks
        if task_data:
            result = {}
            for task_name, (task_labels, task_preds) in task_data.items():
                f1 = f1_score(task_labels, task_preds, average="macro")
                accuracy = accuracy_score(task_labels, task_preds)
                result[task_name] = {"f1": f1, "accuracy": accuracy}
            return result
        elif isinstance(labels, dict) and isinstance(preds, dict):
            result = {}
            for task_name in labels:
                if task_name in preds:
                    f1 = f1_score(labels[task_name], preds[task_name], average="macro")
                    accuracy = accuracy_score(labels[task_name], preds[task_name])
                    result[task_name] = {"f1": f1, "accuracy": accuracy}
            return result
        else:
            raise ValueError("For task_specific metrics, either task_data or labels and preds dictionaries must be provided")
            
    elif metric_type == "combined":
        # Calculate combined metrics across all tasks
        if labels is None or preds is None:
            raise ValueError("Labels and predictions must be provided for combined metrics")
            
        # Handle label encoding for non-numeric labels
        if not all(isinstance(x, (int, float)) for x in labels + preds):
            le = LabelEncoder()
            le.fit(labels + preds)
            labels = le.transform(labels)
            preds = le.transform(preds)
            
        f1 = f1_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        
        if return_format == "tuple":
            return f1, accuracy
        return {"f1": f1, "accuracy": accuracy}
        
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def get_layer_freeze_range(pretrained_path):
    if not pretrained_path:
        return {"min": 0, "max": 0}
    
    config = AutoConfig.from_pretrained(pretrained_path)
    total_layers = config.num_hidden_layers
    return {"min": 0, "max": total_layers - 1}


def prepare_training_environment(config):
    """
    Prepare the training environment by setting seed and loading data.
    
    Returns:
        tuple: (device, train_loader, val_loader, train_cell_id_mapping, 
               val_cell_id_mapping, num_labels_list)
    """
    from .data import prepare_data_loaders
    
    # Set seed for reproducibility
    set_seed(config["seed"])

    # Set up device - for non-distributed training
    if not config.get("distributed_training", False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # For distributed training, device will be set per process
        device = None
    
    # Load data using the streaming dataset
    data = prepare_data_loaders(config)
    
    # For distributed training, we'll set up samplers later in the distributed worker
    # Don't create DistributedSampler here as process group isn't initialized yet
    
    return (
        device,
        data["train_loader"],
        data["val_loader"],
        data["train_cell_mapping"],
        data["val_cell_mapping"],
        data["num_labels_list"],
    )


# Optuna hyperparameter optimization utilities
def save_trial_callback(study, trial, trials_result_path):
    """
    Callback to save Optuna trial results to a file.
    
    Args:
        study: Optuna study object
        trial: Current trial object
        trials_result_path: Path to save trial results
    """
    with open(trials_result_path, "a") as f:
        f.write(
            f"Trial {trial.number}: Value (F1 Macro): {trial.value}, Params: {trial.params}\n"
        )


def create_optuna_study(objective, n_trials: int, trials_result_path: str, tensorboard_log_dir: str) -> optuna.Study:
    """Create and run an Optuna study with TensorBoard logging."""
    from optuna.integration import TensorBoardCallback
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[
            lambda study, trial: save_trial_callback(study, trial, trials_result_path),
            TensorBoardCallback(dirname=tensorboard_log_dir, metric_name="F1 Macro")
        ]
    )
    return study


@contextmanager
def setup_logging(config):
    from torch.utils.tensorboard import SummaryWriter
    run_name = config.get("run_name", "manual_run")
    log_dir = os.path.join(config["tensorboard_log_dir"], run_name)
    writer = SummaryWriter(log_dir=log_dir)
    try:
        yield writer
    finally:
        writer.close()


def log_training_step(loss, writer, config, epoch, steps_per_epoch, batch_idx):
    """Log training step metrics to TensorBoard and optionally W&B."""
    writer.add_scalar(
        "Training Loss", loss, epoch * steps_per_epoch + batch_idx
    )
    if config.get("use_wandb", False):
        import wandb
        wandb.log({"Training Loss": loss})


def log_validation_metrics(task_metrics, val_loss, config, writer, epoch):
    """Log validation metrics to console, TensorBoard, and optionally W&B."""
    for task_name, metrics in task_metrics.items():
        print(
            f"{task_name} - Validation F1 Macro: {metrics['f1']:.4f}, Validation Accuracy: {metrics['accuracy']:.4f}"
        )
        if config.get("use_wandb", False):
            import wandb
            wandb.log(
                {
                    f"{task_name} Validation F1 Macro": metrics["f1"],
                    f"{task_name} Validation Accuracy": metrics["accuracy"],
                }
            )

    writer.add_scalar("Validation Loss", val_loss, epoch)
    for task_name, metrics in task_metrics.items():
        writer.add_scalar(f"{task_name} - Validation F1 Macro", metrics["f1"], epoch)
        writer.add_scalar(
            f"{task_name} - Validation Accuracy", metrics["accuracy"], epoch
        )


def load_label_mappings(results_dir: str, task_names: List[str]) -> Dict[str, Dict]:
    """Load or initialize task label mappings."""
    label_mappings_path = os.path.join(results_dir, "task_label_mappings_val.pkl")
    if os.path.exists(label_mappings_path):
        with open(label_mappings_path, 'rb') as f:
            return pickle.load(f)
    return {task_name: {} for task_name in task_names}


def create_prediction_row(sample_idx: int, val_cell_indices: Dict, task_true_labels: Dict, 
                         task_pred_labels: Dict, task_pred_probs: Dict, task_names: List[str], 
                         inverted_mappings: Dict, val_cell_mapping: Dict) -> Dict:
    """Create a row for validation predictions."""
    batch_cell_idx = val_cell_indices.get(sample_idx)
    cell_id = val_cell_mapping.get(batch_cell_idx, f"unknown_cell_{sample_idx}") if batch_cell_idx is not None else f"unknown_cell_{sample_idx}"
    
    row = {"Cell ID": cell_id}
    for task_name in task_names:
        if task_name in task_true_labels and sample_idx < len(task_true_labels[task_name]):
            true_idx = task_true_labels[task_name][sample_idx]
            pred_idx = task_pred_labels[task_name][sample_idx]
            true_label = inverted_mappings.get(task_name, {}).get(true_idx, f"Unknown-{true_idx}")
            pred_label = inverted_mappings.get(task_name, {}).get(pred_idx, f"Unknown-{pred_idx}")
            
            row.update({
                f"{task_name}_true_idx": true_idx,
                f"{task_name}_pred_idx": pred_idx,
                f"{task_name}_true_label": true_label,
                f"{task_name}_pred_label": pred_label
            })
            
            if task_name in task_pred_probs and sample_idx < len(task_pred_probs[task_name]):
                probs = task_pred_probs[task_name][sample_idx]
                if isinstance(probs, (list, np.ndarray)) or (hasattr(probs, '__iter__') and not isinstance(probs, str)):
                    prob_list = list(probs) if not isinstance(probs, list) else probs
                    row[f"{task_name}_all_probs"] = ",".join(map(str, prob_list))
                    for class_idx, prob in enumerate(prob_list):
                        class_label = inverted_mappings.get(task_name, {}).get(class_idx, f"Unknown-{class_idx}")
                        row[f"{task_name}_prob_{class_label}"] = prob
                else:
                    row[f"{task_name}_all_probs"] = str(probs)
    
    return row


def save_validation_predictions(
    val_cell_indices,
    task_true_labels,
    task_pred_labels,
    task_pred_probs,
    config,
    trial_number=None,
):
    """Save validation predictions to a CSV file with class labels and probabilities."""
    os.makedirs(config["results_dir"], exist_ok=True)
    
    if trial_number is not None:
        os.makedirs(os.path.join(config["results_dir"], f"trial_{trial_number}"), exist_ok=True)
        val_preds_file = os.path.join(config["results_dir"], f"trial_{trial_number}/val_preds.csv")
    else:
        val_preds_file = os.path.join(config["results_dir"], "manual_run_val_preds.csv")

    if not val_cell_indices or not task_true_labels:
        pd.DataFrame().to_csv(val_preds_file, index=False)
        return

    try:
        label_mappings = load_label_mappings(config["results_dir"], config["task_names"])
        inverted_mappings = {task: {idx: label for label, idx in mapping.items()} for task, mapping in label_mappings.items()}
        val_cell_mapping = config.get("val_cell_mapping", {})
        
        # Determine maximum number of samples
        max_samples = max(
            [len(val_cell_indices)] + 
            [len(task_true_labels[task]) for task in task_true_labels]
        )
        
        rows = [
            create_prediction_row(
                sample_idx, val_cell_indices, task_true_labels, task_pred_labels, 
                task_pred_probs, config["task_names"], inverted_mappings, val_cell_mapping
            )
            for sample_idx in range(max_samples)
        ]
        
        pd.DataFrame(rows).to_csv(val_preds_file, index=False)
    except Exception as e:
        pd.DataFrame([{"Error": str(e)}]).to_csv(val_preds_file, index=False)


def setup_distributed_environment(rank, world_size, config):
    """
    Setup the distributed training environment.
    
    Args:
        rank (int): The rank of the current process
        world_size (int): Total number of processes
        config (dict): Configuration dictionary
    """
    os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config.get('master_port', '12355')
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set the device for this process
    torch.cuda.set_device(rank)


def train_distributed(trainer_class, config, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list, trial_number=None, shared_dict=None):
    """Run distributed training across multiple GPUs with fallback to single GPU."""
    world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        print("Distributed training requested but only one GPU found. Falling back to single GPU training.")
        config["distributed_training"] = False
        trainer = trainer_class(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer.device = device
        train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list = trainer.setup(
            train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list
        )
        val_loss, model = trainer.train(
            train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list
        )
        model_save_directory = os.path.join(config["model_save_path"], "GeneformerMultiTask")
        save_model(model, model_save_directory)
        save_hyperparameters(model_save_directory, {
            **get_config_value(config, "manual_hyperparameters", {}),
            "dropout_rate": config["dropout_rate"],
            "use_task_weights": config["use_task_weights"],
            "task_weights": config["task_weights"],
            "max_layers_to_freeze": config["max_layers_to_freeze"],
            "use_attention_pooling": config["use_attention_pooling"],
        })
        
        if shared_dict is not None:
            shared_dict['val_loss'] = val_loss
            task_true_labels, task_pred_labels, task_pred_probs = collect_validation_predictions(model, val_loader, device, config)
            shared_dict['task_metrics'] = calculate_metrics(labels=task_true_labels, preds=task_pred_labels, metric_type="task_specific")
            shared_dict['model_state_dict'] = {k: v.cpu() for k, v in model.state_dict().items()}
        
        return val_loss, model
    
    print(f"Using distributed training with {world_size} GPUs")
    mp.spawn(
        _distributed_worker,
        args=(world_size, trainer_class, config, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list, trial_number, shared_dict),
        nprocs=world_size,
        join=True
    )
    
    if trial_number is None and shared_dict is None:
        model_save_directory = os.path.join(config["model_save_path"], "GeneformerMultiTask")
        model_path = os.path.join(model_save_directory, "pytorch_model.bin")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_model(config, num_labels_list, device)
        model.load_state_dict(torch.load(model_path))
        return 0.0, model
    
    return None


def _distributed_worker(rank, world_size, trainer_class, config, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list, trial_number=None, shared_dict=None):
    """Worker function for distributed training."""
    setup_distributed_environment(rank, world_size, config)
    config["local_rank"] = rank
    
    # Set up distributed samplers
    from torch.utils.data import DistributedSampler
    from .data import get_data_loader
    
    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    train_loader = get_data_loader(train_loader.dataset, config["batch_size"], sampler=train_sampler, shuffle=False)
    val_loader = get_data_loader(val_loader.dataset, config["batch_size"], sampler=val_sampler, shuffle=False)
    
    if rank == 0:
        print(f"Rank {rank}: Training {len(train_sampler)} samples, Validation {len(val_sampler)} samples")
        print(f"Total samples across {world_size} GPUs: Training {len(train_sampler) * world_size}, Validation {len(val_sampler) * world_size}")
    
    # Create and setup trainer
    trainer = trainer_class(config)
    train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list = trainer.setup(
        train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list
    )
    
    # Train the model
    val_loss, model = trainer.train(
        train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list
    )
    
    # Save model only from the main process
    if rank == 0:
        model_save_directory = os.path.join(config["model_save_path"], "GeneformerMultiTask")
        save_model(model, model_save_directory)
        
        save_hyperparameters(model_save_directory, {
            **get_config_value(config, "manual_hyperparameters", {}),
            "dropout_rate": config["dropout_rate"],
            "use_task_weights": config["use_task_weights"],
            "task_weights": config["task_weights"],
            "max_layers_to_freeze": config["max_layers_to_freeze"],
            "use_attention_pooling": config["use_attention_pooling"],
        })
        
        # For Optuna trials, store results in shared dictionary
        if shared_dict is not None:
            shared_dict['val_loss'] = val_loss
            
            # Run validation on full dataset from rank 0 for consistent metrics
            full_val_loader = get_data_loader(val_loader.dataset, config["batch_size"], sampler=None, shuffle=False)
            
            # Get validation predictions using our utility function
            task_true_labels, task_pred_labels, task_pred_probs = collect_validation_predictions(
                model, full_val_loader, trainer.device, config
            )
            
            # Calculate metrics
            task_metrics = calculate_metrics(labels=task_true_labels, preds=task_pred_labels, metric_type="task_specific")
            shared_dict['task_metrics'] = task_metrics
            
            # Store model state dict
            if isinstance(model, DDP):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            shared_dict['model_state_dict'] = {k: v.cpu() for k, v in model_state_dict.items()}
    
    # Clean up distributed environment
    dist.destroy_process_group()


def save_model_without_heads(model_directory):
    """
    Save a version of the fine-tuned model without classification heads.
    
    Args:
        model_directory (str): Path to the directory containing the fine-tuned model
    """
    import torch
    from transformers import BertConfig, BertModel
    
    # Load the full model
    model_path = os.path.join(model_directory, "pytorch_model.bin")
    config_path = os.path.join(model_directory, "config.json")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model files not found in {model_directory}")
    
    # Load the configuration
    config = BertConfig.from_json_file(config_path)
    
    # Load the model state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create a new model without heads
    base_model = BertModel(config)
    
    # Filter out the classification head parameters
    base_model_state_dict = {}
    for key, value in state_dict.items():
        # Only keep parameters that belong to the base model (not classification heads)
        if not key.startswith('classification_heads') and not key.startswith('attention_pool'):
            base_model_state_dict[key] = value
    
    # Load the filtered state dict into the base model
    base_model.load_state_dict(base_model_state_dict, strict=False)
    
    # Save the model without heads
    output_dir = os.path.join(model_directory, "model_without_heads")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model weights
    torch.save(base_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save the configuration
    base_model.config.to_json_file(os.path.join(output_dir, "config.json"))
    
    print(f"Model without classification heads saved to {output_dir}")
    return output_dir


def get_config_value(config: Dict, key: str, default=None):
    
    return config.get(key, default)


def collect_validation_predictions(model, val_loader, device, config) -> tuple:
    task_true_labels = {}
    task_pred_labels = {}
    task_pred_probs = {}
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = [batch["labels"][task_name].to(device) for task_name in config["task_names"]]
            _, logits, _ = model(input_ids, attention_mask, labels)
            
            for sample_idx in range(len(batch["input_ids"])):
                for i, task_name in enumerate(config["task_names"]):
                    if task_name not in task_true_labels:
                        task_true_labels[task_name] = []
                        task_pred_labels[task_name] = []
                        task_pred_probs[task_name] = []
                    
                    true_label = batch["labels"][task_name][sample_idx].item()
                    pred_label = torch.argmax(logits[i][sample_idx], dim=-1).item()
                    pred_prob = torch.softmax(logits[i][sample_idx], dim=-1).cpu().numpy()
                    
                    task_true_labels[task_name].append(true_label)
                    task_pred_labels[task_name].append(pred_label)
                    task_pred_probs[task_name].append(pred_prob)
    
    return task_true_labels, task_pred_labels, task_pred_probs
