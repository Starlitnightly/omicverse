import os
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna
import functools
import time

from .model import GeneformerMultiTask
from .utils import (
    calculate_metrics,
    get_layer_freeze_range,
    set_seed,
    initialize_wandb,
    create_model,
    setup_optimizer_and_scheduler,
    save_model,
    save_hyperparameters,
    prepare_training_environment,
    log_training_step,
    log_validation_metrics,
    save_validation_predictions,
    setup_logging,
    setup_distributed_environment,
    train_distributed
)


class Trainer:
    """Trainer class for multi-task learning"""
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.is_distributed = config.get("distributed_training", False)
        self.local_rank = config.get("local_rank", 0)
        self.is_main_process = not self.is_distributed or self.local_rank == 0
        
    def train_epoch(self, train_loader, epoch):
        """Train the model for one epoch."""
        epoch_start = time.time()
        self.model.train()
        
        # For distributed training, we need to be aware of the global batch count
        if self.is_distributed:
            # Get world size for reporting
            world_size = dist.get_world_size()
            # Calculate total batches across all GPUs
            total_batches_global = len(train_loader) * world_size if self.local_rank == 0 else len(train_loader)
        else:
            world_size = 1
            total_batches_global = len(train_loader)
        
        progress_bar = None
        if self.is_main_process:
            # Use the global batch count for progress reporting in distributed mode
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", 
                               total=len(train_loader))
            iterator = progress_bar
            
            # Report distributed training information
            if self.is_distributed:
                print(f"Distributed training: {world_size} GPUs, {len(train_loader)} batches per GPU, "
                      f"{total_batches_global} total batches globally")
        else:
            iterator = train_loader
            
        batch_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        
        # Get gradient accumulation steps from config (default to 1 if not specified)
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        # Zero gradients at the beginning
        self.optimizer.zero_grad()
        
        # Track loss for the entire epoch
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(iterator):
            batch_start = time.time()
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = [
                batch["labels"][task_name].to(self.device) for task_name in self.config["task_names"]
            ]

            forward_start = time.time()
            loss, _, _ = self.model(input_ids, attention_mask, labels)
            
            # Scale loss by accumulation steps for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Track loss - store the unscaled loss for reporting
            unscaled_loss = loss.item() * (1 if accumulation_steps == 1 else accumulation_steps)
            total_loss += unscaled_loss
            num_batches += 1
            accumulated_loss += loss.item()  # For gradient accumulation tracking
            
            backward_start = time.time()
            
            # Use no_sync() for all but the last accumulation step to avoid unnecessary communication
            if self.is_distributed and accumulation_steps > 1:
                # If this is not the last accumulation step or the last batch
                if (batch_idx + 1) % accumulation_steps != 0 and (batch_idx + 1) != len(train_loader):
                    with self.model.no_sync():
                        loss.backward()
                else:
                    loss.backward()
            else:
                # Non-distributed training or accumulation_steps=1
                loss.backward()
            
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            # Only update weights after accumulation_steps or at the end of the epoch
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.config["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

                optimizer_start = time.time()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                optimizer_end = time.time()
                optimizer_times.append(optimizer_end - optimizer_start)
                
                # Log after optimizer step
                if self.is_main_process:
                    # Calculate running average loss
                    avg_loss = total_loss / num_batches
                    
                    log_training_step(avg_loss, self.writer, self.config, epoch, len(train_loader), batch_idx)
                    
                    # Update progress bar with just the running average loss
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                accumulated_loss = 0.0
            else:
                optimizer_times.append(0)  # No optimizer step taken
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
        
        epoch_end = time.time()
        
        # Calculate average loss for the epoch
        epoch_avg_loss = total_loss / num_batches
        
        # If distributed, gather losses from all processes to compute global average
        if self.is_distributed:
            # Create a tensor to hold the loss
            loss_tensor = torch.tensor([epoch_avg_loss], device=self.device)
            # Gather losses from all processes
            all_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_losses, loss_tensor)
            # Compute the global average loss across all processes
            epoch_avg_loss = torch.mean(torch.stack(all_losses)).item()
        
        if self.is_main_process:
            # douhble check if batch_size has already been adjusted for world_size in the config
            # This avoids double-counting the effective batch size
            per_gpu_batch_size = self.config['batch_size']
            total_effective_batch = per_gpu_batch_size * accumulation_steps * world_size
            
            print(f"Epoch {epoch+1} timing:")
            print(f"  Total epoch time: {epoch_end - epoch_start:.2f}s")
            print(f"  Average batch time: {sum(batch_times)/len(batch_times):.4f}s")
            print(f"  Average forward time: {sum(forward_times)/len(forward_times):.4f}s")
            print(f"  Average backward time: {sum(backward_times)/len(backward_times):.4f}s")
            print(f"  Average optimizer time: {sum([t for t in optimizer_times if t > 0])/max(1, len([t for t in optimizer_times if t > 0])):.4f}s")
            print(f"  Gradient accumulation steps: {accumulation_steps}")
            print(f"  Batch size per GPU: {per_gpu_batch_size}")
            print(f"  Effective global batch size: {total_effective_batch}")
            print(f"  Average training loss: {epoch_avg_loss:.4f}")
            if self.is_distributed:
                print(f"  Total batches processed across all GPUs: {total_batches_global}")
                print(f"  Communication optimization: Using no_sync() for gradient accumulation")

        return epoch_avg_loss  # Return the average loss for the epoch

    def validate_model(self, val_loader):
        val_start = time.time()
        self.model.eval()
        val_loss = 0.0
        task_true_labels = {task_name: [] for task_name in self.config["task_names"]}
        task_pred_labels = {task_name: [] for task_name in self.config["task_names"]}
        task_pred_probs = {task_name: [] for task_name in self.config["task_names"]}
        
        val_cell_ids = {}
        sample_counter = 0

        batch_times = []
        
        # Print validation dataset size
        if self.is_main_process:
            print(f"Validation dataset size: {len(val_loader.dataset)} samples")
            print(f"Number of validation batches: {len(val_loader)}")

            if self.is_distributed:
                world_size = dist.get_world_size()
                print(f"Distributed validation: {world_size} GPUs")
                if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'num_samples'):
                    samples_per_gpu = val_loader.sampler.num_samples
                    print(f"Each GPU processes {samples_per_gpu} validation samples")
                    print(f"Total validation samples processed: {samples_per_gpu * world_size}")
        
        with torch.no_grad():
            for batch in val_loader:
                batch_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = [
                    batch["labels"][task_name].to(self.device)
                    for task_name in self.config["task_names"]
                ]
                loss, logits, _ = self.model(input_ids, attention_mask, labels)
                val_loss += loss.item()

                if "cell_id" in batch:
                    for i, cell_id in enumerate(batch["cell_id"]):
                        # Store the actual index for later mapping to unique_cell_id
                        val_cell_ids[sample_counter + i] = cell_id.item()
                
                for sample_idx in range(len(batch["input_ids"])):
                    for i, task_name in enumerate(self.config["task_names"]):
                        true_label = batch["labels"][task_name][sample_idx].item()
                        pred_label = torch.argmax(logits[i][sample_idx], dim=-1).item()
                        # Store the full probability distribution
                        pred_prob = torch.softmax(logits[i][sample_idx], dim=-1).cpu().numpy().tolist()
                        task_true_labels[task_name].append(true_label)
                        task_pred_labels[task_name].append(pred_label)
                        task_pred_probs[task_name].append(pred_prob)
                
                # Update current index for cell ID tracking
                sample_counter += len(batch["input_ids"])
                
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)

        # norm validation loss by the number of batches
        val_loss /= len(val_loader)
        
        # distributed, gather results from all processes
        if self.is_distributed:
            # Create tensors to hold the local results
            loss_tensor = torch.tensor([val_loss], device=self.device)
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss_tensor)
            
            # Compute average loss across all processes
            val_loss = torch.mean(torch.cat(gathered_losses)).item()
            
            world_size = dist.get_world_size()
            
            if self.is_main_process:
                print(f"Collected predictions from rank {self.local_rank}")
                print(f"Number of samples processed by this rank: {sample_counter}")
            
        val_end = time.time()
        
        if self.is_main_process:
            print(f"Validation timing:")
            print(f"  Total validation time: {val_end - val_start:.2f}s")
            print(f"  Average batch time: {sum(batch_times)/len(batch_times):.4f}s")
            print(f"  Collected {len(val_cell_ids)} cell indices from validation data")
            print(f"  Processed {sample_counter} total samples during validation")
            
            # Print number of samples per task
            for task_name in self.config["task_names"]:
                print(f"  Task {task_name}: {len(task_true_labels[task_name])} samples")
        
        return val_loss, task_true_labels, task_pred_labels, task_pred_probs, val_cell_ids

    def train(self, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list):
        """Train the model and return validation loss and trained model."""
        if self.config.get("use_wandb", False) and self.is_main_process:
            initialize_wandb(self.config)

        # Create model
        self.model = create_model(self.config, num_labels_list, self.device, self.is_distributed, self.local_rank)
        
        # Setup optimizer and scheduler
        total_steps = len(train_loader) * self.config["epochs"]
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(self.model, self.config, total_steps)

        # Training loop
        if self.is_main_process:
            epoch_progress = tqdm(range(self.config["epochs"]), desc="Training Progress")
        else:
            epoch_progress = range(self.config["epochs"])
        
        best_val_loss = float('inf')
        train_losses = []
        
        with setup_logging(self.config) as self.writer:
            for epoch in epoch_progress:
                if self.is_distributed:
                    train_loader.sampler.set_epoch(epoch)
                    
                train_loss = self.train_epoch(train_loader, epoch)
                train_losses.append(train_loss)
                
                # Run validation after each epoch if configured to do so
                if self.config.get("validate_each_epoch", False):
                    val_loss, _, _, _, _ = self.validate_model(val_loader)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    
                    if self.is_main_process:
                        epoch_progress.set_postfix({
                            "train_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "best_val_loss": f"{best_val_loss:.4f}"
                        })
                else:
                    if self.is_main_process:
                        epoch_progress.set_postfix({
                            "train_loss": f"{train_loss:.4f}"
                        })

            val_loss, task_true_labels, task_pred_labels, task_pred_probs, val_cell_ids = self.validate_model(val_loader)
            task_metrics = calculate_metrics(labels=task_true_labels, preds=task_pred_labels, metric_type="task_specific")

            if self.is_main_process:
                log_validation_metrics(task_metrics, val_loss, self.config, self.writer, self.config["epochs"])

                # Save validation predictions
                save_validation_predictions(
                    val_cell_ids, 
                    task_true_labels, 
                    task_pred_labels, 
                    task_pred_probs, 
                    {**self.config, "val_cell_mapping": val_cell_id_mapping}  # Include the mapping
                )

                if self.config.get("use_wandb", False):
                    import wandb
                    wandb.finish()

                print(f"\nTraining Summary:")
                print(f"  Final Training Loss: {train_losses[-1]:.4f}")
                print(f"  Final Validation Loss: {val_loss:.4f}")
                for task_name, metrics in task_metrics.items():
                    print(f"  {task_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
        return val_loss, self.model  # Return both the validation loss and the trained model

    def setup(self, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list):
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = create_model(self.config, num_labels_list, self.device)
        
        # war model w DDP
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
            # communication hook to optimize gradient synchronization
            from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
            
            # default hook which maintains full precision
            self.model.register_comm_hook(
                state=None,
                hook=comm_hooks.allreduce_hook
            )
            
            print(f"Rank {self.local_rank}: Registered communication hook for optimized gradient synchronization")

            print(f"Rank {self.local_rank}: Using samplers created in distributed worker")
            print(f"Rank {self.local_rank}: Training dataset has {len(train_loader.dataset)} samples")
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'num_samples'):
                print(f"Rank {self.local_rank}: This GPU will process {train_loader.sampler.num_samples} training samples per epoch")
            
            if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'num_samples'):
                print(f"Rank {self.local_rank}: This GPU will process {val_loader.sampler.num_samples} validation samples")
        
        # Set up optimizer and scheduler
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(
            self.model, self.config, len(train_loader)
        )

        if self.is_main_process and self.config.get("use_wandb", False):
            initialize_wandb(self.config)
        
        return train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list


def train_model(config, device, train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list):
    """Train a model with the given configuration and data."""
    # Check if distributed training is enabled
    if config.get("distributed_training", False):
        # Check if we have multiple GPUs
        if torch.cuda.device_count() > 1:
            result = train_distributed(
                Trainer, 
                config, 
                train_loader, 
                val_loader, 
                train_cell_id_mapping, 
                val_cell_id_mapping, 
                num_labels_list
            )
            if result is not None:
                return result
        else:
            print("Distributed training requested but only one GPU found. Falling back to single GPU training.")
            config["distributed_training"] = False
    
    # Non-distributed training
    trainer = Trainer(config)
    trainer.device = device
    return trainer.train(train_loader, val_loader, train_cell_id_mapping, val_cell_id_mapping, num_labels_list)


def objective(
    trial,
    train_loader,
    val_loader,
    train_cell_id_mapping,
    val_cell_id_mapping,
    num_labels_list,
    config,
    device,
):
    """Objective function for Optuna hyperparameter optimization."""
    set_seed(config["seed"])
    initialize_wandb(config)

    trial_config = config.copy()
    
    # Suggest hyperparameters for this trial
    for param_name, param_config in config["hyperparameters"].items():
        if param_name == "lr_scheduler_type":
            trial_config[param_name] = trial.suggest_categorical(
                param_name, param_config["choices"]
            )
        elif param_name == "task_weights" and config["use_task_weights"]:
            weights = [
                trial.suggest_float(
                    f"task_weight_{i}",
                    param_config["low"],
                    param_config["high"],
                )
                for i in range(len(num_labels_list))
            ]
            weight_sum = sum(weights)
            trial_config[param_name] = [w / weight_sum for w in weights]
        elif "log" in param_config and param_config["log"]:
            trial_config[param_name] = trial.suggest_float(
                param_name, param_config["low"], param_config["high"], log=True
            )
        else:
            trial_config[param_name] = trial.suggest_float(
                param_name, param_config["low"], param_config["high"]
            )
    
    # Set appropriate max layers to freeze based on pretrained model
    if "max_layers_to_freeze" in trial_config:
        freeze_range = get_layer_freeze_range(trial_config["pretrained_path"])
        trial_config["max_layers_to_freeze"] = int(trial.suggest_int(
            "max_layers_to_freeze", 
            freeze_range["min"], 
            freeze_range["max"]
        ))

    trial_config["run_name"] = f"trial_{trial.number}"
    
    # Handle distributed training for this trial
    if trial_config.get("distributed_training", False) and torch.cuda.device_count() > 1:
        manager = mp.Manager()
        shared_dict = manager.dict()
        
        train_distributed(
            Trainer,
            trial_config, 
            train_loader, 
            val_loader, 
            train_cell_id_mapping, 
            val_cell_id_mapping, 
            num_labels_list,
            trial.number, 
            shared_dict
        )
        
        val_loss = shared_dict.get('val_loss', float('inf'))
        task_metrics = shared_dict.get('task_metrics', {})
        
        trial.set_user_attr("model_state_dict", shared_dict.get('model_state_dict', {}))
        trial.set_user_attr("task_weights", trial_config["task_weights"])
        
        if config.get("use_wandb", False):
            import wandb
            wandb.log({
                "trial_number": trial.number,
                "val_loss": val_loss,
                **{f"{task_name}_f1": metrics["f1"] for task_name, metrics in task_metrics.items()},
                **{f"{task_name}_accuracy": metrics["accuracy"] for task_name, metrics in task_metrics.items()},
            })
            wandb.finish()
            
        return val_loss
    
    with setup_logging(trial_config) as writer:
        trainer = Trainer(trial_config)
        trainer.device = device
        trainer.writer = writer
        
        # Create model with trial hyperparameters
        trainer.model = create_model(trial_config, num_labels_list, device)
        total_steps = len(train_loader) * config["epochs"]
        trainer.optimizer, trainer.scheduler = setup_optimizer_and_scheduler(trainer.model, trial_config, total_steps)

        # Training loop
        for epoch in range(config["epochs"]):
            trainer.train_epoch(train_loader, epoch)

        val_loss, task_true_labels, task_pred_labels, task_pred_probs, val_cell_ids = trainer.validate_model(val_loader)
        task_metrics = calculate_metrics(labels=task_true_labels, preds=task_pred_labels, metric_type="task_specific")

        # Log metrics
        log_validation_metrics(task_metrics, val_loss, trial_config, writer, config["epochs"])

        # Save validation predictions
        save_validation_predictions(
            val_cell_ids,
            task_true_labels,
            task_pred_labels,
            task_pred_probs,
            {**trial_config, "val_cell_mapping": val_cell_id_mapping},
            trial.number,
        )

        # Store model state dict and task weights in trial user attributes
        trial.set_user_attr("model_state_dict", trainer.model.state_dict())
        trial.set_user_attr("task_weights", trial_config["task_weights"])

        # Report intermediate value to Optuna
        trial.report(val_loss, config["epochs"])
        if trial.should_prune():
            raise optuna.TrialPruned()

        if config.get("use_wandb", False):
            import wandb
            wandb.log(
                {
                    "trial_number": trial.number,
                    "val_loss": val_loss,
                    **{f"{task_name}_f1": metrics["f1"] for task_name, metrics in task_metrics.items()},
                    **{f"{task_name}_accuracy": metrics["accuracy"] for task_name, metrics in task_metrics.items()},
                    **{k: v for k, v in trial_config.items() if k in [
                        "learning_rate", "warmup_ratio", "weight_decay", "dropout_rate",
                        "lr_scheduler_type", "use_attention_pooling", "max_layers_to_freeze"
                    ]},
                }
            )
            wandb.finish()

    return val_loss


def run_manual_tuning(config):
    """Run training with manually specified hyperparameters."""
    (
        device,
        train_loader,
        val_loader,
        train_cell_id_mapping,
        val_cell_id_mapping,
        num_labels_list,
    ) = prepare_training_environment(config)

    print("\nManual hyperparameters being used:")
    for key, value in config["manual_hyperparameters"].items():
        print(f"{key}: {value}")
    print() 

    # Update config with manual hyperparameters
    for key, value in config["manual_hyperparameters"].items():
        config[key] = value

    # Train the model
    val_loss, trained_model = train_model(
        config,
        device,
        train_loader,
        val_loader,
        train_cell_id_mapping,
        val_cell_id_mapping,
        num_labels_list,
    )

    print(f"\nValidation loss with manual hyperparameters: {val_loss}")

    # Save the trained model - only if not using distributed training
    # (distributed training saves the model in the worker)
    if not config.get("distributed_training", False):
        model_save_directory = os.path.join(
            config["model_save_path"], "GeneformerMultiTask"
        )
        save_model(trained_model, model_save_directory)

        # Save the hyperparameters
        hyperparams_to_save = {
            **config["manual_hyperparameters"],
            "dropout_rate": config["dropout_rate"],
            "use_task_weights": config["use_task_weights"],
            "task_weights": config["task_weights"],
            "max_layers_to_freeze": config["max_layers_to_freeze"],
            "use_attention_pooling": config["use_attention_pooling"],
        }
        save_hyperparameters(model_save_directory, hyperparams_to_save)

    return val_loss


def run_optuna_study(config):
    """Run hyperparameter optimization using Optuna."""
    # Prepare training environment
    (
        device,
        train_loader,
        val_loader,
        train_cell_id_mapping,
        val_cell_id_mapping,
        num_labels_list,
    ) = prepare_training_environment(config)

    # If manual hyperparameters are specified, use them instead of running Optuna
    if config.get("use_manual_hyperparameters", False):
        return run_manual_tuning(config)

    # Create a partial function with fixed arguments for the objective
    objective_with_config_and_data = functools.partial(
        objective,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cell_id_mapping=train_cell_id_mapping,
        val_cell_id_mapping=val_cell_id_mapping,
        num_labels_list=num_labels_list,
        config=config,
        device=device,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        study_name=config["study_name"],
        # storage=config["storage"],
        load_if_exists=True,
    )

    study.optimize(objective_with_config_and_data, n_trials=config["n_trials"])

    # After finding the best trial
    best_params = study.best_trial.params
    best_task_weights = study.best_trial.user_attrs["task_weights"]
    print("Saving the best model and its hyperparameters...")

    # Create a model with the best hyperparameters
    best_model = GeneformerMultiTask(
        config["pretrained_path"],
        num_labels_list,
        dropout_rate=best_params["dropout_rate"],
        use_task_weights=config["use_task_weights"],
        task_weights=best_task_weights,
        max_layers_to_freeze=best_params.get("max_layers_to_freeze", 0),
        use_attention_pooling=best_params.get("use_attention_pooling", False),
    )

    # Get the best model state dictionary
    best_model_state_dict = study.best_trial.user_attrs["model_state_dict"]

    best_model_state_dict = {
        k.replace("module.", ""): v for k, v in best_model_state_dict.items()
    }

    best_model.load_state_dict(best_model_state_dict, strict=False)

    model_save_directory = os.path.join(
        config["model_save_path"], "GeneformerMultiTask"
    )
    save_model(best_model, model_save_directory)

    save_hyperparameters(model_save_directory, {**best_params, "task_weights": best_task_weights})

    return study.best_trial.value  # Return the best validation loss
