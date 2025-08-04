import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

from .collators import DataCollatorForMultitaskCellClassification


class StreamingMultiTaskDataset(Dataset):
    
    def __init__(self, dataset_path, config, is_test=False, dataset_type=""):
        """Initialize the streaming dataset."""
        self.dataset = load_from_disk(dataset_path)
        self.config = config
        self.is_test = is_test
        self.dataset_type = dataset_type
        self.cell_id_mapping = {}
        
        # Setup task and column mappings
        self.task_names = [f"task{i+1}" for i in range(len(config["task_columns"]))]
        self.task_to_column = dict(zip(self.task_names, config["task_columns"]))
        config["task_names"] = self.task_names
        
        # Check if unique_cell_id column exists in the dataset
        self.has_unique_cell_ids = "unique_cell_id" in self.dataset.column_names
        print(f"{'Found' if self.has_unique_cell_ids else 'No'} unique_cell_id column in {dataset_type} dataset")
        
        # Setup label mappings
        self.label_mappings_path = os.path.join(
            config["results_dir"],
            f"task_label_mappings{'_val' if dataset_type == 'validation' else ''}.pkl"
        )
        
        if not is_test:
            self._validate_columns()
            self.task_label_mappings, self.num_labels_list = self._create_label_mappings()
            self._save_label_mappings()
        else:
            # Load existing mappings for test data
            self.task_label_mappings = self._load_label_mappings()
            self.num_labels_list = [len(mapping) for mapping in self.task_label_mappings.values()]
    
    def _validate_columns(self):
        """Ensures required columns are present in the dataset."""
        missing_columns = [col for col in self.task_to_column.values() 
                          if col not in self.dataset.column_names]
        if missing_columns:
            raise KeyError(
                f"Missing columns in {self.dataset_type} dataset: {missing_columns}. "
                f"Available columns: {self.dataset.column_names}"
            )
    
    def _create_label_mappings(self):
        """Creates label mappings for the dataset."""
        task_label_mappings = {}
        num_labels_list = []
        
        for task, column in self.task_to_column.items():
            unique_values = sorted(set(self.dataset[column]))
            mapping = {label: idx for idx, label in enumerate(unique_values)}
            task_label_mappings[task] = mapping
            num_labels_list.append(len(unique_values))
        
        return task_label_mappings, num_labels_list
    
    def _save_label_mappings(self):
        """Saves label mappings to a pickle file."""
        with open(self.label_mappings_path, "wb") as f:
            pickle.dump(self.task_label_mappings, f)
    
    def _load_label_mappings(self):
        """Loads label mappings from a pickle file."""
        with open(self.label_mappings_path, "rb") as f:
            return pickle.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        record = self.dataset[idx]
        
        # Store cell ID mapping
        if self.has_unique_cell_ids:
            unique_cell_id = record["unique_cell_id"]
            self.cell_id_mapping[idx] = unique_cell_id
        else:
            self.cell_id_mapping[idx] = f"cell_{idx}"
        
        # Create transformed record
        transformed_record = {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "cell_id": idx,
        }
        
        # Add labels
        if not self.is_test:
            label_dict = {
                task: self.task_label_mappings[task][record[column]]
                for task, column in self.task_to_column.items()
            }
        else:
            label_dict = {task: -1 for task in self.config["task_names"]}
        
        transformed_record["label"] = label_dict
        
        return transformed_record


def get_data_loader(dataset, batch_size, sampler=None, shuffle=True):
    """Create a DataLoader with the given dataset and parameters."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=0,
        pin_memory=True,
        collate_fn=DataCollatorForMultitaskCellClassification(),
    )


def prepare_data_loaders(config, include_test=False):
    """Prepare data loaders for training, validation, and optionally test."""
    result = {}
    
    # Process train data
    train_dataset = StreamingMultiTaskDataset(
        config["train_path"], 
        config, 
        dataset_type="train"
    )
    result["train_loader"] = get_data_loader(train_dataset, config["batch_size"])
    
    # Store the cell ID mapping from the dataset
    result["train_cell_mapping"] = {k: v for k, v in train_dataset.cell_id_mapping.items()}
    print(f"Collected {len(result['train_cell_mapping'])} cell IDs from training dataset")
    
    result["num_labels_list"] = train_dataset.num_labels_list
    
    # Process validation data
    val_dataset = StreamingMultiTaskDataset(
        config["val_path"], 
        config, 
        dataset_type="validation"
    )
    result["val_loader"] = get_data_loader(val_dataset, config["batch_size"])
    
    # Store the complete cell ID mapping for validation
    for idx in range(len(val_dataset)):
        _ = val_dataset[idx]
    
    result["val_cell_mapping"] = {k: v for k, v in val_dataset.cell_id_mapping.items()}
    print(f"Collected {len(result['val_cell_mapping'])} cell IDs from validation dataset")
    
    # Validate label mappings
    validate_label_mappings(config)
    
    # Process test data if requested
    if include_test and "test_path" in config:
        test_dataset = StreamingMultiTaskDataset(
            config["test_path"], 
            config, 
            is_test=True,
            dataset_type="test"
        )
        result["test_loader"] = get_data_loader(test_dataset, config["batch_size"])
        
        for idx in range(len(test_dataset)):
            _ = test_dataset[idx]
        
        result["test_cell_mapping"] = {k: v for k, v in test_dataset.cell_id_mapping.items()}
        print(f"Collected {len(result['test_cell_mapping'])} cell IDs from test dataset")
    
    return result


def validate_label_mappings(config):
    """Ensures train and validation label mappings are consistent."""
    train_mappings_path = os.path.join(config["results_dir"], "task_label_mappings.pkl")
    val_mappings_path = os.path.join(config["results_dir"], "task_label_mappings_val.pkl")
    
    with open(train_mappings_path, "rb") as f:
        train_mappings = pickle.load(f)
    
    with open(val_mappings_path, "rb") as f:
        val_mappings = pickle.load(f)

    for task_name in config["task_names"]:
        if train_mappings[task_name] != val_mappings[task_name]:
            raise ValueError(
                f"Mismatch in label mappings for task '{task_name}'.\n"
                f"Train Mapping: {train_mappings[task_name]}\n"
                f"Validation Mapping: {val_mappings[task_name]}"
            )


# Legacy functions for backward compatibility
def preload_and_process_data(config):
    """Preloads and preprocesses train and validation datasets."""
    data = prepare_data_loaders(config)
    
    return (
        data["train_loader"].dataset,
        data["train_cell_mapping"],
        data["val_loader"].dataset,
        data["val_cell_mapping"],
        data["num_labels_list"]
    )


def preload_data(config):
    """Preprocesses train and validation data for trials."""
    data = prepare_data_loaders(config)
    return data["train_loader"], data["val_loader"]


def load_and_preprocess_test_data(config):
    """Loads and preprocesses test data."""
    test_dataset = StreamingMultiTaskDataset(
        config["test_path"], 
        config, 
        is_test=True,
        dataset_type="test"
    )
    
    return (
        test_dataset,
        test_dataset.cell_id_mapping,
        test_dataset.num_labels_list
    )


def prepare_test_loader(config):
    """Prepares DataLoader for test data."""
    data = prepare_data_loaders(config, include_test=True)
    return data["test_loader"], data["test_cell_mapping"], data["num_labels_list"]