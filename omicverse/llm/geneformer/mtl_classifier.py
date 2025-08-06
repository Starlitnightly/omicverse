"""
Geneformer multi-task cell classifier.

**Input data:**

| Single-cell transcriptomes as Geneformer rank value encodings with cell state labels for each task in Geneformer .dataset format (generated from single-cell RNAseq data by tokenizer.py). Must contain "unique_cell_id" column for logging.

**Usage:**

.. code-block :: python

    >>> from geneformer import MTLClassifier
    >>> mc = MTLClassifier(task_columns = ["task1", "task2"],
    ...                 study_name = "mtl",
    ...                 pretrained_path = "/path/pretrained/model",
    ...                 train_path = "/path/train/set",
    ...                 val_path = "/path/eval/set",
    ...                 test_path = "/path/test/set",
    ...                 model_save_path = "/results/directory/save_path",
    ...                 trials_result_path = "/results/directory/results.txt",
    ...                 results_dir = "/results/directory",
    ...                 tensorboard_log_dir = "/results/tblogdir",
    ...                 hyperparameters = hyperparameters)
    >>> mc.run_optuna_study()
    >>> mc.load_and_evaluate_test_model()
    >>> mc.save_model_without_heads()
"""

import logging
import os

from .mtl import eval_utils, utils, train

logger = logging.getLogger(__name__)


class MTLClassifier:
    valid_option_dict = {
        "task_columns": {list},
        "train_path": {None, str},
        "val_path": {None, str},
        "test_path": {None, str},
        "pretrained_path": {None, str},
        "model_save_path": {None, str},
        "results_dir": {None, str},
        "batch_size": {None, int},
        "n_trials": {None, int},
        "study_name": {None, str},
        "max_layers_to_freeze": {None, dict},
        "epochs": {None, int},
        "tensorboard_log_dir": {None, str},
        "distributed_training": {None, bool},
        "master_addr": {None, str},
        "master_port": {None, str},
        "use_attention_pooling": {None, bool},
        "use_task_weights": {None, bool},
        "hyperparameters": {None, dict},
        "manual_hyperparameters": {None, dict},
        "use_manual_hyperparameters": {None, bool},
        "use_wandb": {None, bool},
        "wandb_project": {None, str},
        "gradient_clipping": {None, bool},
        "max_grad_norm": {None, int, float},
        "seed": {None, int},
        "trials_result_path": {None, str},
        "gradient_accumulation_steps": {None, int},
    }

    def __init__(
        self,
        task_columns=None,
        train_path=None,
        val_path=None,
        test_path=None,
        pretrained_path=None,
        model_save_path=None,
        results_dir=None,
        trials_result_path=None,
        batch_size=4,
        n_trials=15,
        study_name="mtl",
        max_layers_to_freeze=None,
        epochs=1,
        tensorboard_log_dir="/results/tblogdir",
        distributed_training=False,
        master_addr="localhost",
        master_port="12355",
        use_attention_pooling=True,
        use_task_weights=True,
        hyperparameters=None,  # Default is None
        manual_hyperparameters=None,  # Default is None
        use_manual_hyperparameters=False,  # Default is False
        use_wandb=False,
        wandb_project=None,
        gradient_clipping=False,
        max_grad_norm=None,
        gradient_accumulation_steps=1,  # Add this line with default value 1
        seed=42,  # Default seed value
    ):
        """
        Initialize Geneformer multi-task classifier.
        
        **Parameters:**
        
        task_columns : list
            | List of tasks for cell state classification
            | Input data columns are labeled with corresponding task names
        study_name : None, str
            | Study name for labeling output files
        pretrained_path : None, str
            | Path to pretrained model
        train_path : None, str
            | Path to training dataset with task columns and "unique_cell_id" column
        val_path : None, str
            | Path to validation dataset with task columns and "unique_cell_id" column
        test_path : None, str
            | Path to test dataset with task columns and "unique_cell_id" column
        model_save_path : None, str
            | Path to directory to save output model (either full model or model without heads)
        trials_result_path : None, str
            | Path to directory to save hyperparameter tuning trial results
        results_dir : None, str
            | Path to directory to save results
        tensorboard_log_dir : None, str
            | Path to directory for Tensorboard logging results
        distributed_training : None, bool
            | Whether to use distributed data parallel training across multiple GPUs
        master_addr : None, str
            | Master address for distributed training (default: localhost)
        master_port : None, str
            | Master port for distributed training (default: 12355)
        use_attention_pooling : None, bool
            | Whether to use attention pooling
        use_task_weights : None, bool
            | Whether to use task weights
        batch_size : None, int
            | Batch size to use
        n_trials : None, int
            | Number of trials for hyperparameter tuning
        epochs : None, int
            | Number of epochs for training
        max_layers_to_freeze : None, dict
            | Dictionary with keys "min" and "max" indicating the min and max layers to freeze from fine-tuning (int)
            | 0: no layers will be frozen; 2: first two layers will be frozen; etc.
        hyperparameters : None, dict
            | Dictionary of categorical max and min for each hyperparameter for tuning
            | For example:
            | {"learning_rate": {"type":"float", "low":"1e-5", "high":"1e-3", "log":True}, "task_weights": {...}, ...}
        manual_hyperparameters : None, dict
            | Dictionary of manually set value for each hyperparameter
            | For example:
            | {"learning_rate": 0.001, "task_weights": [1, 1], ...}
        use_manual_hyperparameters : None, bool
            | Whether to use manually set hyperparameters
        use_wandb : None, bool
            | Whether to use Weights & Biases for logging
        wandb_project : None, str
            | Weights & Biases project name
        gradient_clipping : None, bool
            | Whether to use gradient clipping
        max_grad_norm : None, int, float
            | Maximum norm for gradient clipping
        gradient_accumulation_steps : None, int
            | Number of steps to accumulate gradients before performing a backward/update pass
        seed : None, int
            | Random seed
        """

        self.task_columns = task_columns
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.pretrained_path = pretrained_path
        self.model_save_path = model_save_path
        self.results_dir = results_dir
        self.trials_result_path = trials_result_path
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.study_name = study_name
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if max_layers_to_freeze is None:
            # Dynamically determine the range of layers to freeze
            layer_freeze_range = utils.get_layer_freeze_range(pretrained_path)
            self.max_layers_to_freeze = {"min": 1, "max": layer_freeze_range["max"]}
        else:
            self.max_layers_to_freeze = max_layers_to_freeze

        self.epochs = epochs
        self.tensorboard_log_dir = tensorboard_log_dir
        self.distributed_training = distributed_training
        self.master_addr = master_addr
        self.master_port = master_port
        self.use_attention_pooling = use_attention_pooling
        self.use_task_weights = use_task_weights
        self.hyperparameters = (
            hyperparameters
            if hyperparameters is not None
            else {
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-3,
                    "log": True,
                },
                "warmup_ratio": {"type": "float", "low": 0.005, "high": 0.01},
                "weight_decay": {"type": "float", "low": 0.01, "high": 0.1},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.7},
                "lr_scheduler_type": {"type": "categorical", "choices": ["cosine"]},
                "task_weights": {"type": "float", "low": 0.1, "high": 2.0},
            }
        )
        self.manual_hyperparameters = (
            manual_hyperparameters
            if manual_hyperparameters is not None
            else {
                "learning_rate": 0.001,
                "warmup_ratio": 0.01,
                "weight_decay": 0.1,
                "dropout_rate": 0.1,
                "lr_scheduler_type": "cosine",
                "use_attention_pooling": False,
                "task_weights": [1, 1],
                "max_layers_to_freeze": 2,
            }
        )
        self.use_manual_hyperparameters = use_manual_hyperparameters
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.gradient_clipping = gradient_clipping
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        if self.use_manual_hyperparameters:
            logger.warning(
                "Hyperparameter tuning is highly recommended for optimal results."
            )

        self.validate_options()

        # set up output directories
        if self.results_dir is not None:
            self.trials_results_path = f"{self.results_dir}/results.txt".replace(
                "//", "/"
            )

        for output_dir in [self.model_save_path, self.results_dir]:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        self.config = {
            key: value
            for key, value in self.__dict__.items()
            if key in self.valid_option_dict
        }

    def validate_options(self):
        # confirm arguments are within valid options and compatible with each other
        for attr_name, valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if not isinstance(attr_value, (list, dict)):
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int, float, list, dict, bool, str]) and isinstance(
                    attr_value, option
                ):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. "
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise ValueError(
                f"Invalid option for {attr_name}. Valid options for {attr_name}: {valid_options}"
            )

    def run_manual_tuning(self):
        """
        Manual hyperparameter tuning and multi-task fine-tuning of pretrained model.
        """
        required_variable_names = [
            "train_path",
            "val_path",
            "pretrained_path",
            "model_save_path",
            "results_dir",
        ]
        required_variables = [
            self.train_path,
            self.val_path,
            self.pretrained_path,
            self.model_save_path,
            self.results_dir,
        ]
        req_var_dict = dict(zip(required_variable_names, required_variables))
        self.validate_additional_options(req_var_dict)

        if not self.use_manual_hyperparameters:
            raise ValueError(
                "Manual hyperparameters are not enabled. Set use_manual_hyperparameters to True."
            )

        # Ensure manual_hyperparameters are set in the config
        self.config["manual_hyperparameters"] = self.manual_hyperparameters
        self.config["use_manual_hyperparameters"] = True

        train.run_manual_tuning(self.config)

    def validate_additional_options(self, req_var_dict):
        missing_variable = False
        for variable_name, variable in req_var_dict.items():
            if variable is None:
                logger.warning(
                    f"Please provide value to MTLClassifier for required variable {variable_name}"
                )
                missing_variable = True
        if missing_variable is True:
            raise ValueError("Missing required variables for MTLClassifier")

    def run_optuna_study(
        self,
    ):
        """
        Hyperparameter optimization and/or multi-task fine-tuning of pretrained model.
        """

        required_variable_names = [
            "train_path",
            "val_path",
            "pretrained_path",
            "model_save_path",
            "results_dir",
        ]
        required_variables = [
            self.train_path,
            self.val_path,
            self.pretrained_path,
            self.model_save_path,
            self.results_dir,
        ]
        req_var_dict = dict(zip(required_variable_names, required_variables))
        self.validate_additional_options(req_var_dict)

        train.run_optuna_study(self.config)

    def load_and_evaluate_test_model(
        self,
    ):
        """
        Loads previously fine-tuned multi-task model and evaluates on test data.
        """

        required_variable_names = ["test_path", "model_save_path", "results_dir"]
        required_variables = [self.test_path, self.model_save_path, self.results_dir]
        req_var_dict = dict(zip(required_variable_names, required_variables))
        self.validate_additional_options(req_var_dict)

        eval_utils.load_and_evaluate_test_model(self.config)

    # def save_model_without_heads(
    #     self,
    # ):
    #     """
    #     Save previously fine-tuned multi-task model without classification heads.
    #     """

    #     required_variable_names = ["model_save_path"]
    #     required_variables = [self.model_save_path]
    #     req_var_dict = dict(zip(required_variable_names, required_variables))
    #     self.validate_additional_options(req_var_dict)

    #     utils.save_model_without_heads(
    #         os.path.join(self.model_save_path, "GeneformerMultiTask")
    #     )
