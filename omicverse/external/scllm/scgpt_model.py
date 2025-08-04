"""
scGPT model implementation with simplified interface.
"""

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from anndata import AnnData
from scipy.sparse import issparse

try:
    from .base import SCLLMBase, ModelConfig, TaskConfig
except ImportError:
    from base import SCLLMBase, ModelConfig, TaskConfig
# Import scGPT components with error handling
try:
    from .scgpt.model import TransformerModel
    from .scgpt.tokenizer.gene_tokenizer import GeneVocab
    from .scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from .scgpt.preprocess import Preprocessor
    from .scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
    from .scgpt.utils import set_seed
    _scgpt_imports_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"scGPT components not available: {e}")
    _scgpt_imports_available = False
    
    # Create placeholder classes/functions
    class TransformerModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("scGPT TransformerModel not available due to missing dependencies")
    
    class GeneVocab:
        @classmethod
        def from_file(cls, *args, **kwargs):
            raise ImportError("scGPT GeneVocab not available due to missing dependencies")
    
    def tokenize_and_pad_batch(*args, **kwargs):
        raise ImportError("scGPT tokenization not available due to missing dependencies")
    
    def random_mask_value(*args, **kwargs):
        raise ImportError("scGPT masking not available due to missing dependencies")
    
    class Preprocessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("scGPT Preprocessor not available due to missing dependencies")
    
    def masked_mse_loss(*args, **kwargs):
        raise ImportError("scGPT losses not available due to missing dependencies")
    
    def criterion_neg_log_bernoulli(*args, **kwargs):
        raise ImportError("scGPT losses not available due to missing dependencies")
    
    def set_seed(*args, **kwargs):
        import random
        import numpy as np
        import torch
        if args:
            seed = args[0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


class ScGPTModel(SCLLMBase):
    """
    Simplified scGPT model interface.
    
    This class provides an easy-to-use interface for scGPT model operations
    including loading, preprocessing, training, and inference.
    """
    
    def __init__(self, device: Optional[str] = None, seed: int = 0):
        """
        Initialize scGPT model.
        
        Args:
            device: Device to run the model on
            seed: Random seed for reproducibility
        """
        super().__init__("scGPT", device)
        self.seed = seed
        set_seed(seed)
        
        # Model components
        self.model = None
        self.vocab = None
        self.preprocessor = None
        self.config = None
        
        # Default parameters matching Tutorial exactly
        self.default_config = ModelConfig(
            embsize=128,  # Tutorial uses 128, not 512
            nhead=4,      # Tutorial uses 4, not 8
            d_hid=128,    # Tutorial uses 128, not 512
            nlayers=4,    # Tutorial uses 4, not 12
            nlayers_cls=3,  # Tutorial uses 3
            dropout=0.2,  # Tutorial uses 0.2
            n_bins=51,    # Tutorial uses 51 bins
            max_seq_len=3001,  # Tutorial max sequence length
            pad_token="<pad>",
            special_tokens=["<pad>", "<cls>", "<eoc>"],
            input_style="binned",
            output_style="binned", 
            input_emb_style="continuous",  # Tutorial uses continuous
            cell_emb_style="cls",  # Tutorial uses cls token
            fast_transformer=True,
            pre_norm=False,
            # Additional Tutorial-specific parameters
            explicit_zero_prob=False,
            use_fast_transformer=True,
            fast_transformer_backend="flash",
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            # Integration-specific parameters (Tutorial_Integration)
            do_dab=False,            # Domain Adversarial Batch correction
            do_mvc=False,            # Masked Value Prediction (GEPC)
            do_ecs=False,            # Elastic Cell Similarity
            use_batch_labels=False,  # Whether to use batch labels
            domain_spec_batchnorm=False,  # Domain-Specific Batch Normalization
            num_batch_labels=None,   # Number of batch types
            dab_weight=1.0,          # DAB loss weight
            ecs_weight=10.0,         # ECS loss weight
            gepc_weight=1.0,         # GEPC (MVC) loss weight
            mask_ratio=0.4,          # Masking ratio for integration (higher than annotation)
        )
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load a pre-trained scGPT model.
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional parameters
        """
        if not _scgpt_imports_available:
            raise ImportError("Cannot load scGPT model: required dependencies not available")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Load vocabulary
        vocab_file = model_path / "vocab.json"
        if vocab_file.exists():
            self.vocab = GeneVocab.from_file(vocab_file)
            print(f"Loaded vocabulary with {len(self.vocab)} genes")
        else:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        
        # Add special tokens
        for token in self.default_config.special_tokens:
            if token not in self.vocab:
                # Note: GeneVocab may not have append_token method
                # This is a simplified approach
                print(f"Warning: Special token {token} not in vocabulary")
        
        self.vocab.set_default_index(self.vocab[self.default_config.pad_token])
        
        # Load model configuration
        config_file = model_path / "args.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Update default config with loaded config
            self.config = copy.deepcopy(self.default_config)
            self.config.update(**model_config)
            self.config.update(**kwargs)  # Override with user parameters
            
            print(f"Loaded model config from {config_file}")
            print(f"Key config parameters:")
            for key in ['embsize', 'nheads', 'd_hid', 'nlayers', 'n_layers_cls']:
                if key in model_config:
                    print(f"  {key}: {model_config[key]}")
        else:
            self.config = self.default_config
            self.config.update(**kwargs)
            print("No config file found, using default configuration")
        
        # Determine number of classes - try to infer from loaded model weights
        n_cls = kwargs.get('n_cls')
        
        if not n_cls:
            # Try to get from config
            n_cls = (getattr(self.config, 'n_cls', None) or 
                    getattr(self.config, 'num_types', None))
        
        if not n_cls:
            # Try to infer from model weights if they exist
            model_file = model_path / "best_model.pt"
            if not model_file.exists():
                model_file = model_path / "model.pt"
            
            if model_file.exists():
                try:
                    checkpoint = torch.load(model_file, map_location="cpu")
                    print(f"Analyzing model checkpoint for n_cls inference...")
                    
                    # Look for classifier output layer - be more specific
                    classifier_candidates = []
                    for key, tensor in checkpoint.items():
                        print(f"  {key}: {tensor.shape}")
                        if ('cls' in key.lower() and 'weight' in key):
                            if 'out' in key.lower() or key.endswith('weight'):
                                if len(tensor.shape) == 2:
                                    classifier_candidates.append((key, tensor.shape[0]))
                    
                    # Choose the most likely classifier layer
                    if classifier_candidates:
                        # Sort by potential n_cls and choose reasonable one
                        classifier_candidates.sort(key=lambda x: x[1])
                        for key, potential_n_cls in classifier_candidates:
                            if potential_n_cls > 1 and potential_n_cls < 1000:
                                n_cls = potential_n_cls
                                print(f"✓ Inferred n_cls={n_cls} from {key} shape")
                                break
                    else:
                        print("⚠ No classifier layers found in checkpoint")
                        
                except Exception as e:
                    print(f"Could not infer n_cls from model weights: {e}")
        
        # Use default if still not found
        if not n_cls:
            n_cls = 50
            print(f"Using default n_cls={n_cls}")
        else:
            print(f"Using n_cls={n_cls} classes")
        
        # Store n_cls for later use
        self.n_cls = n_cls
        
        # Initialize model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.config.embsize,
            nhead=self.config.nhead,
            d_hid=self.config.d_hid,
            nlayers=self.config.nlayers,
            nlayers_cls=self.config.nlayers_cls,
            n_cls=n_cls,  # Use proper number of classes
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.config.pad_token,
            pad_value=-2,
            do_mvc=False,  # Default to False for basic usage
            do_dab=False,  # Default to False for basic usage
            use_batch_labels=False,  # Default to False for basic usage
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            input_emb_style=self.config.input_emb_style,
            n_input_bins=self.config.n_bins + 2,
            cell_emb_style=self.config.cell_emb_style,
            mvc_decoder_style="inner product",
            ecs_threshold=0.3,
            explicit_zero_prob=False,
            use_fast_transformer=self.config.fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=self.config.pre_norm,
        )
        
        # Check if user wants to override classifier for multi-class prediction
        force_multiclass = kwargs.get('force_multiclass', False)
        expected_n_cls = kwargs.get('expected_n_cls', None)
        
        # Load model weights
        model_file = model_path / "best_model.pt"
        if not model_file.exists():
            model_file = model_path / "model.pt"
        
        if model_file.exists():
            try:
                checkpoint = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"✓ Successfully loaded all model weights from {model_file}")
                
                # Check if classifier weights were loaded
                classifier_keys = [k for k in checkpoint.keys() if 'cls' in k.lower() or 'classifier' in k.lower()]
                print(f"Classifier layers found: {len(classifier_keys)}")
                if classifier_keys:
                    for key in classifier_keys[:3]:  # Show first 3
                        print(f"  {key}: {checkpoint[key].shape}")
                
            except Exception as e:
                print(f"Warning: Could not load all model weights: {e}")
                print("Attempting to load compatible weights only...")
                
                # Load compatible weights only
                model_dict = self.model.state_dict()
                checkpoint = torch.load(model_file, map_location=self.device)
                
                compatible_dict = {}
                incompatible_dict = {}
                
                for k, v in checkpoint.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                        else:
                            incompatible_dict[k] = f"shape mismatch: {v.shape} vs {model_dict[k].shape}"
                    else:
                        incompatible_dict[k] = "key not found in model"
                
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)
                
                print(f"✓ Loaded compatible weights: {len(compatible_dict)}/{len(checkpoint)} parameters")
                if incompatible_dict:
                    print(f"⚠ Incompatible weights ({len(incompatible_dict)}):")
                    for k, reason in list(incompatible_dict.items())[:5]:  # Show first 5
                        print(f"  {k}: {reason}")
                    if len(incompatible_dict) > 5:
                        print(f"  ... and {len(incompatible_dict) - 5} more")
        
        # Check if the loaded model has the right number of classes
        if hasattr(self.model, 'cls_decoder') and hasattr(self.model.cls_decoder, 'out_layer'):
            actual_n_cls = self.model.cls_decoder.out_layer.out_features
            print(f"Loaded model has {actual_n_cls} output classes")
            
            if actual_n_cls == 1 and (force_multiclass or expected_n_cls):
                target_n_cls = expected_n_cls or n_cls
                print(f"⚠ Model has only 1 output class, but {target_n_cls} classes expected")
                print(f"This suggests the model is a pre-trained model that needs fine-tuning")
                print(f"For inference, you may need a model that was fine-tuned for your specific task")
                
                if force_multiclass:
                    print(f"🔄 Reinitializing classifier with {target_n_cls} classes...")
                    # Reinitialize the classifier layer
                    import torch.nn as nn
                    if hasattr(self.model.cls_decoder, '_decoder'):
                        # Get the input dimension from the previous layer
                        prev_dim = self.model.cls_decoder._decoder[-2].out_features
                    else:
                        prev_dim = self.model.d_model
                    
                    # Replace the output layer
                    self.model.cls_decoder.out_layer = nn.Linear(prev_dim, target_n_cls)
                    self.model.cls_decoder.out_layer.to(self.device)
                    print(f"✓ Classifier reinitialized with {target_n_cls} output classes")
                    print("⚠ Note: This classifier is randomly initialized and needs training!")
        
        self.model.to(self.device)
        self.is_loaded = True
        print(f"Model loaded successfully on {self.device}")
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess data for scGPT following Tutorial implementation exactly.
        
        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters
                skip_normalization (bool): Force skip normalization (default: auto-detect)
                force_normalization (bool): Force apply normalization even if detected as normalized
                data_is_raw (bool): Whether input data is raw counts (default: True)
                
        Returns:
            Preprocessed AnnData object
        """
        adata_processed = adata.copy()
        
        # Step 1: Filter genes by vocabulary BEFORE preprocessing (like Tutorial)
        if self.vocab is not None:
            print("Filtering genes by vocabulary...")
            adata_processed.var["id_in_vocab"] = [
                1 if gene in self.vocab else -1 
                for gene in adata_processed.var_names
            ]
            
            genes_in_vocab = (adata_processed.var["id_in_vocab"] >= 0).sum()
            total_genes = len(adata_processed.var)
            print(f"Matched {genes_in_vocab}/{total_genes} genes in vocabulary")
            
            if genes_in_vocab == 0:
                raise ValueError("No genes matched the vocabulary! Check gene naming conventions.")
            
            # Keep only genes in vocabulary
            adata_processed = adata_processed[:, adata_processed.var["id_in_vocab"] >= 0]
            print(f"After vocabulary filtering: {adata_processed.n_vars} genes retained")
        
        # Step 2: Initialize preprocessor with Tutorial-exact parameters
        if self.preprocessor is None:
            # Get data_is_raw flag - crucial for Tutorial compatibility
            data_is_raw = kwargs.get('data_is_raw', True)  # Tutorial assumes raw data by default
            
            self.preprocessor = Preprocessor(
                use_key="X",  # Use adata.X as input
                filter_gene_by_counts=kwargs.get('filter_gene_by_counts', False),
                filter_cell_by_counts=kwargs.get('filter_cell_by_counts', False),
                normalize_total=kwargs.get('normalize_total', 1e4),  # Tutorial uses 1e4
                result_normed_key="X_normed",
                log1p=kwargs.get('log1p', data_is_raw),  # Tutorial: log1p if data_is_raw
                result_log1p_key="X_log1p",
                subset_hvg=kwargs.get('subset_hvg', False),  # Tutorial: False
                hvg_flavor=kwargs.get('hvg_flavor', "seurat_v3" if data_is_raw else "cell_ranger"),
                binning=self.config.n_bins,  # Tutorial: n_bins = 51
                result_binned_key="X_binned",
            )
            print(f"Initialized preprocessor with Tutorial settings:")
            print(f"  n_bins: {self.config.n_bins}")
            print(f"  normalize_total: {self.preprocessor.normalize_total}")
            print(f"  log1p: {self.preprocessor.log1p}")
            print(f"  data_is_raw: {data_is_raw}")
        
        # Step 3: Smart preprocessing with user control
        skip_preprocessing = False
        
        # Check user preferences
        skip_normalization = kwargs.get('skip_normalization', None)
        force_normalization = kwargs.get('force_normalization', False)
        
        # Check if already fully preprocessed
        if hasattr(adata_processed, 'layers') and 'X_binned' in adata_processed.layers:
            print("Data appears to be already preprocessed (X_binned exists), skipping preprocessing")
            skip_preprocessing = True
        else:
            # Inspect data to detect normalization status
            if issparse(adata_processed.X):
                X_data = adata_processed.X.toarray()
            else:
                X_data = adata_processed.X
            
            # Calculate statistics to detect normalization
            cell_totals = X_data.sum(axis=1)
            mean_total = cell_totals.mean()
            median_total = np.median(cell_totals)
            
            print(f"Data inspection:")
            print(f"  Cell total counts - Mean: {mean_total:.1f}, Median: {median_total:.1f}")
            print(f"  Data range: [{X_data.min():.3f}, {X_data.max():.3f}]")
            print(f"  Data type: {X_data.dtype}")
            
            # Auto-detect normalization status
            auto_detected_normalized = False
            if 9000 <= mean_total <= 11000:  # Close to 10k normalization
                auto_detected_normalized = True
                print("  🔍 Auto-detected: Data appears normalized to ~10k")
            elif 900000 <= mean_total <= 1100000:  # Close to 1M normalization  
                auto_detected_normalized = True
                print("  🔍 Auto-detected: Data appears normalized to ~1M")
            elif mean_total < 1000:  # Very low counts, likely already log-transformed
                auto_detected_normalized = True
                print("  🔍 Auto-detected: Data appears log-transformed (low values)")
            else:
                print("  🔍 Auto-detected: Data appears to be raw counts")
            
            # Determine final normalization decision
            should_skip_normalization = False
            
            if force_normalization:
                print("  ⚡ User override: FORCING normalization")
                should_skip_normalization = False
            elif skip_normalization is True:
                print("  ⚡ User override: SKIPPING normalization")
                should_skip_normalization = True
            elif skip_normalization is False:
                print("  ⚡ User override: APPLYING normalization")
                should_skip_normalization = False
            else:
                # Use auto-detection
                should_skip_normalization = auto_detected_normalized
                if should_skip_normalization:
                    print("  ✓ Decision: Will SKIP normalization (auto-detected as normalized)")
                else:
                    print("  ✓ Decision: Will APPLY normalization (auto-detected as raw)")
            
            # Adjust preprocessor settings if skipping normalization
            if should_skip_normalization:
                print("  🔄 Adjusting preprocessor to skip normalization...")
                # Store original settings
                original_normalize_total = self.preprocessor.normalize_total
                original_log1p = self.preprocessor.log1p
                
                # Modify settings
                self.preprocessor.normalize_total = None  # Skip normalization
                self.preprocessor.log1p = False
                print("  🔄 Also skipping log1p (data appears log-transformed)")
                
                print(f"  Modified settings: normalize_total={self.preprocessor.normalize_total}, log1p={self.preprocessor.log1p}")
            else:
                print("  ✓ Will apply normalization as configured")
        
        if not skip_preprocessing:
            print("Applying preprocessing pipeline...")
            self.preprocessor(adata_processed, batch_key=kwargs.get('batch_key', None))
            
            # Restore original settings if they were modified
            if 'original_normalize_total' in locals():
                self.preprocessor.normalize_total = original_normalize_total
                self.preprocessor.log1p = original_log1p
                print("  ✓ Restored original preprocessor settings")
            
            print("Preprocessing completed")
            
            # Debug: Check preprocessing results
            if 'X_binned' in adata_processed.layers:
                binned_data = adata_processed.layers['X_binned']
                print(f"Final binned data shape: {binned_data.shape}")
                print(f"Final binned data range: [{binned_data.min():.3f}, {binned_data.max():.3f}]")
                print(f"Unique values in binned data: {len(np.unique(binned_data))}")
                
                # Verify binning is correct (should be integers from 0 to n_bins-1, plus special values)
                unique_vals = np.unique(binned_data)
                print(f"Binned values range: {unique_vals[:5]}...{unique_vals[-5:]} (showing first and last 5)")
                
                if binned_data.max() > self.config.n_bins:
                    print(f"  ⚠ WARNING: Binned values exceed n_bins ({self.config.n_bins})")
                if binned_data.min() < -2:
                    print(f"  ⚠ WARNING: Unexpected negative values in binned data")
        else:
            print("Using existing preprocessed data")
        
        return adata_processed
    
    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using the scGPT model.
        
        Args:
            adata: Input AnnData object
            task: Task type ('annotation', 'embedding', 'integration')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        adata_processed = self.preprocess(adata, **kwargs)
        
        if task == "annotation":
            return self._predict_annotation(adata_processed, **kwargs)
        elif task == "embedding":
            return self._predict_embedding(adata_processed, **kwargs)
        elif task == "integration":
            return self._predict_integration(adata_processed, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Available tasks: annotation, embedding, integration")
    
    def _predict_annotation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Predict cell type annotations following Tutorial exactly."""
        # Get gene expression data from binned layer (Tutorial approach)
        input_layer_key = "X_binned"
        
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found in adata.layers")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        print(f"Input data shape: {all_counts.shape}")
        print(f"Input data range: [{all_counts.min():.3f}, {all_counts.max():.3f}]")
        
        # Get genes and their vocab IDs (following Tutorial exactly)
        genes = adata.var_names.tolist()
        
        # Tutorial approach: get vocab IDs for all genes
        gene_ids = []
        for gene in genes:
            if gene in self.vocab:
                gene_ids.append(self.vocab[gene])
            else:
                # This should not happen if preprocessing was done correctly
                raise ValueError(f"Gene {gene} not found in vocabulary")
        
        gene_ids = np.array(gene_ids, dtype=int)
        print(f"Gene IDs shape: {gene_ids.shape}, range: [{gene_ids.min()}, {gene_ids.max()}]")
        
        # Tutorial exact tokenization
        print("Tokenizing data...")
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,  # Tutorial uses -2 for padding
            append_cls=True,  # Tutorial appends CLS token
            include_zero_gene=kwargs.get('include_zero_gene', False),
        )
        
        print(f"Tokenized genes shape: {tokenized_data['genes'].shape}")
        print(f"Tokenized values shape: {tokenized_data['values'].shape}")
        
        # Tutorial exact masking (even with mask_ratio=0.0)
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        print(f"Applying masking with mask_ratio={mask_ratio}...")
        
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,  # Tutorial uses -1 for masking
            pad_value=-2,   # Tutorial uses -2 for padding
        )
        
        print(f"After masking - values shape: {input_values.shape}")
        print(f"After masking - values range: [{input_values.min()}, {input_values.max()}]")
        
        # Create dataset exactly like Tutorial
        dataset_dict = {
            "gene_ids": tokenized_data["genes"],  # Note: Tutorial uses "genes" key
            "values": input_values,
            "target_values": tokenized_data["values"],
        }
        
        # Create data loader - Tutorial approach
        batch_size = kwargs.get('batch_size', 32)
        dataset = SimpleDataset(dataset_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Created dataloader with {len(dataloader)} batches, batch_size={batch_size}")
        
        # Make predictions following Tutorial exactly
        self.model.eval()
        all_predictions = []
        all_embeddings = []  
        all_logits = []
        
        print(f"Starting prediction loop...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Get batch data and move to device
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                
                # Create padding mask (Tutorial approach)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                if batch_idx == 0:
                    print(f"First batch shapes:")
                    print(f"  input_gene_ids: {input_gene_ids.shape}")
                    print(f"  input_values: {input_values.shape}")
                    print(f"  padding_mask: {src_key_padding_mask.shape}")
                    print(f"  padding_mask_sum: {src_key_padding_mask.sum()}")
                
                # Tutorial exact model call - critical parameters
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,  # Tutorial: None for inference
                    CLS=True,           # Tutorial: True for classification
                    CCE=False,          # Tutorial: False during evaluation
                    MVC=False,          # Tutorial: False during evaluation
                    ECS=False,          # Tutorial: False during evaluation 
                    do_sample=False,    # Tutorial: False during inference
                )
                
                # Process model outputs - following Tutorial exactly
                if batch_idx == 0:
                    print(f"Model output keys: {list(output_dict.keys())}")
                
                if "cls_output" in output_dict:
                    logits = output_dict["cls_output"]  # Tutorial: cls_output
                    predictions = logits.argmax(1).cpu().numpy()
                    all_predictions.append(predictions)
                    all_logits.append(logits.cpu().numpy())
                    
                    # Detailed debug info for first batch
                    if batch_idx == 0:
                        print(f"First batch classification debug:")
                        print(f"  Logits shape: {logits.shape}")
                        print(f"  Logits dtype: {logits.dtype}")
                        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                        print(f"  Logits mean: {logits.mean().item():.3f}")
                        print(f"  Logits std: {logits.std().item():.3f}")
                        
                        # Check each class logit
                        mean_logits_per_class = logits.mean(dim=0)
                        print(f"  Mean logits per class: {mean_logits_per_class.cpu().numpy()}")
                        
                        print(f"  Predictions sample: {predictions[:10]}")
                        print(f"  Unique predictions in batch: {np.unique(predictions)}")
                        
                        # Check if model is producing reasonable outputs
                        if logits.std().item() < 0.1:
                            print(f"  ⚠ WARNING: Very low logits variance ({logits.std().item():.4f})")
                            print(f"  This suggests the model may not be working correctly")
                        
                        # Check for gradient/parameter issues
                        if torch.isnan(logits).any():
                            print(f"  ❌ ERROR: NaN values detected in logits!")
                        if torch.isinf(logits).any():
                            print(f"  ❌ ERROR: Inf values detected in logits!")
                
                # Get cell embeddings if available
                if "cell_emb" in output_dict:
                    embeddings = output_dict["cell_emb"].cpu().numpy()
                    all_embeddings.append(embeddings)
                    
                    if batch_idx == 0:
                        print(f"Cell embeddings shape: {embeddings.shape}")
                        print(f"Cell embeddings range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
                elif batch_idx == 0:
                    print("  ⚠ No cell embeddings found in output")
        
        results = {}
        if all_predictions:
            predictions = np.concatenate(all_predictions)
            results["predictions"] = predictions
            
            # Prediction analysis
            unique_preds, counts = np.unique(predictions, return_counts=True)
            print(f"\nPrediction Analysis:")
            print(f"  Total cells: {len(predictions)}")
            print(f"  Unique predictions: {len(unique_preds)}")
            print(f"  Prediction distribution:")
            for pred, count in zip(unique_preds, counts):
                percentage = count / len(predictions) * 100
                print(f"    Class {pred}: {count} cells ({percentage:.1f}%)")
            
            if len(unique_preds) == 1:
                print(f"  ⚠ WARNING: All cells predicted as same class ({unique_preds[0]})")
                print(f"    This suggests the model may not be working correctly.")
                
                if all_logits:
                    all_logits_concat = np.concatenate(all_logits)
                    print(f"  Logits analysis:")
                    print(f"    Logits shape: {all_logits_concat.shape}")
                    print(f"    Logits mean: {all_logits_concat.mean():.3f}")
                    print(f"    Logits std: {all_logits_concat.std():.3f}")
                    
                    # Check if one class dominates
                    mean_logits_per_class = all_logits_concat.mean(axis=0)
                    dominant_class = mean_logits_per_class.argmax()
                    print(f"    Dominant class: {dominant_class} (logit: {mean_logits_per_class[dominant_class]:.3f})")
                    
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings)
            results["embeddings"] = embeddings
            print(f"  Cell embeddings shape: {embeddings.shape}")
        
        return results
    
    def _predict_embedding(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Get cell embeddings."""
        # Similar to annotation but only return embeddings
        result = self._predict_annotation(adata, **kwargs)
        return {"embeddings": result.get("embeddings")}
    
    def _predict_integration(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Perform batch integration following Tutorial_Integration exactly."""
        # Check for batch information
        batch_key = kwargs.get('batch_key', 'batch')
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs. "
                           f"Integration requires batch labels.")
        
        print(f"🔄 Starting batch integration using batch key: '{batch_key}'")
        
        # Get gene expression data from binned layer
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found in adata.layers")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        print(f"Input data shape: {all_counts.shape}")
        print(f"Input data range: [{all_counts.min():.3f}, {all_counts.max():.3f}]")
        
        # Process batch labels
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(batch_labels)
        num_batches = len(unique_batches)
        
        print(f"Batch information:")
        print(f"  Number of batches: {num_batches}")
        print(f"  Batch distribution: {np.bincount(batch_labels)}")
        
        # Get genes and their vocab IDs 
        genes = adata.var_names.tolist()
        gene_ids = []
        for gene in genes:
            if gene in self.vocab:
                gene_ids.append(self.vocab[gene])
            else:
                raise ValueError(f"Gene {gene} not found in vocabulary")
        
        gene_ids = np.array(gene_ids, dtype=int)
        
        # Tutorial exact tokenization for integration
        print("Tokenizing data for integration...")
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=kwargs.get('include_zero_gene', False),
        )
        
        # Integration uses higher masking ratio (Tutorial: 0.4)
        mask_ratio = kwargs.get('mask_ratio', 0.4)
        print(f"Applying masking for integration with mask_ratio={mask_ratio}...")
        
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        
        # Create dataset with batch labels
        dataset_dict = {
            "gene_ids": tokenized_data["genes"],
            "values": input_values,
            "target_values": tokenized_data["values"],
            "batch_labels": torch.from_numpy(batch_labels).long(),
        }
        
        # Create data loader
        batch_size = kwargs.get('batch_size', 32)
        dataset = SimpleDataset(dataset_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Created dataloader with {len(dataloader)} batches for integration")
        
        # Make predictions with batch-aware model
        self.model.eval()
        all_embeddings = []
        
        print(f"Starting integration inference...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                batch_labels_tensor = batch_data["batch_labels"].to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                if batch_idx == 0:
                    print(f"First batch debug info:")
                    print(f"  input_gene_ids: {input_gene_ids.shape}")
                    print(f"  input_values: {input_values.shape}")
                    print(f"  batch_labels: {batch_labels_tensor.shape}")
                    print(f"  unique batch labels in batch: {torch.unique(batch_labels_tensor)}")
                
                # Tutorial exact model call for integration
                # Note: For integration, we enable batch-aware features
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels_tensor if self.config.use_batch_labels else None,
                    CLS=False,          # Integration focuses on embeddings, not classification
                    CCE=False,          # Contrastive cell embedding can be enabled if needed
                    MVC=self.config.do_mvc,  # Masked Value Prediction (GEPC)
                    ECS=self.config.do_ecs,  # Elastic Cell Similarity  
                    do_sample=False,    # No sampling during inference
                )
                
                # Extract cell embeddings for integration
                if "cell_emb" in output_dict:
                    embeddings = output_dict["cell_emb"].cpu().numpy()
                    all_embeddings.append(embeddings)
                    
                    if batch_idx == 0:
                        print(f"Cell embeddings shape: {embeddings.shape}")
                        print(f"Cell embeddings range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
                else:
                    print(f"⚠ Warning: No cell embeddings found in model output")
                    # Fallback: use the last hidden state
                    if "encoder_output" in output_dict:
                        # Use CLS token embedding as fallback
                        encoder_output = output_dict["encoder_output"]
                        cls_embeddings = encoder_output[:, 0, :].cpu().numpy()  # CLS token
                        all_embeddings.append(cls_embeddings)
        
        results = {}
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings)
            results["embeddings"] = embeddings
            results["batch_labels"] = batch_labels
            results["integrated_embeddings"] = embeddings  # Same as embeddings for compatibility
            
            print(f"✓ Integration completed!")
            print(f"  Final embeddings shape: {embeddings.shape}")
            print(f"  Number of batches integrated: {num_batches}")
            
            # Add batch statistics
            results["integration_stats"] = {
                "num_batches": num_batches,
                "batch_distribution": np.bincount(batch_labels).tolist(),
                "total_cells": len(batch_labels)
            }
        else:
            raise RuntimeError("No embeddings were generated during integration")
        
        return results
    
    def integrate(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using scGPT model.
        
        This method automatically chooses the appropriate integration approach:
        - If model is fine-tuned for integration: uses trained integration capabilities
        - If model is pre-trained only: uses embeddings + post-hoc correction methods
        
        Args:
            adata: Input data with batch information
            batch_key: Column name for batch labels
            **kwargs: Additional parameters including:
                - correction_method: Post-hoc correction method ('combat', 'mnn', 'center_scale', 'none')
                - Other integration parameters
            
        Returns:
            Dictionary with integrated embeddings and batch statistics
        """
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs")
        
        # Check if model is fine-tuned for integration
        has_integration_training = hasattr(self, '_integration_trained') and self._integration_trained
        
        if has_integration_training:
            print(f"🔄 Using fine-tuned integration model...")
            # Use the trained integration capabilities
            return self._predict_integration(adata, batch_key=batch_key, **kwargs)
        else:
            print(f"🔄 Using pre-trained model with post-hoc batch correction...")
            return self._apply_post_hoc_integration(adata, batch_key=batch_key, **kwargs)
    
    def _predict_integration(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """Use trained integration model for prediction."""
        print("   Using trained integration model for batch correction...")
        
        # This would use the actual integration model prediction
        # Similar to the predict method but for integration task
        adata_processed = self.preprocess(adata, add_cls_token=True)
        
        # Get integrated embeddings from the trained model
        # Implementation would depend on the trained integration model
        embeddings = self.get_embeddings(adata_processed, **kwargs)
        
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        
        results = {
            'embeddings': embeddings,
            'batch_labels': batch_labels,
            'integration_method': 'trained_model',
            'integration_stats': {
                'num_batches': len(np.unique(batch_labels)),
                'batch_distribution': np.bincount(batch_labels).tolist(),
                'total_cells': len(batch_labels),
                'method': 'fine_tuned_integration'
            }
        }
        
        print("✓ Integration completed using fine-tuned model")
        return results
    
    def _apply_post_hoc_integration(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """Apply post-hoc batch correction methods to pre-trained embeddings."""
        
        # Get pre-trained embeddings
        print("   Extracting pre-trained cell embeddings...")
        embeddings = self.get_embeddings(adata, **kwargs)
        
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(batch_labels)
        num_batches = len(unique_batches)
        
        print(f"   Found {num_batches} batches with distribution: {np.bincount(batch_labels)}")
        
        # Apply post-hoc batch correction methods
        correction_method = kwargs.get('correction_method', 'combat')
        
        if correction_method == 'combat':
            corrected_embeddings = self._apply_combat_correction(embeddings, batch_labels, **kwargs)
        elif correction_method == 'mnn':
            corrected_embeddings = self._apply_mnn_correction(embeddings, batch_labels, **kwargs)
        elif correction_method == 'center_scale':
            corrected_embeddings = self._apply_center_scale_correction(embeddings, batch_labels, **kwargs)
        elif correction_method == 'none':
            print("   Using embeddings without additional correction")
            corrected_embeddings = embeddings
        else:
            print(f"   Unknown correction method '{correction_method}', using center_scale")
            corrected_embeddings = self._apply_center_scale_correction(embeddings, batch_labels, **kwargs)
        
        results = {
            'embeddings': corrected_embeddings,
            'original_embeddings': embeddings,
            'batch_labels': batch_labels,
            'correction_method': correction_method,
            'integration_stats': {
                'num_batches': num_batches,
                'batch_distribution': np.bincount(batch_labels).tolist(),
                'total_cells': len(batch_labels),
                'correction_applied': correction_method,
                'method': 'post_hoc_correction'
            }
        }
        
        print(f"✓ Integration completed using {correction_method} post-hoc correction")
        return results
    
    def _apply_combat_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply ComBat-style batch correction to embeddings."""
        print("   Applying ComBat-style correction...")
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Simple ComBat-like correction
            corrected = embeddings.copy()
            
            # Standardize within each batch
            for batch_id in np.unique(batch_labels):
                batch_mask = batch_labels == batch_id
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:  # Need at least 2 samples
                    # Center and scale within batch
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8  # Avoid division by zero
                    
                    # Remove batch-specific mean and scale
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std
            
            # Global standardization
            global_scaler = StandardScaler()
            corrected = global_scaler.fit_transform(corrected)
            
            print(f"     ComBat-style correction applied to {embeddings.shape[0]} cells")
            return corrected
            
        except Exception as e:
            print(f"     ComBat correction failed: {e}, using original embeddings")
            return embeddings
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply MNN-inspired batch correction."""
        print("   Applying MNN-inspired correction...")
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = np.unique(batch_labels)
            
            if len(unique_batches) < 2:
                print("     Only one batch found, no correction needed")
                return corrected
            
            # Find mutual nearest neighbors between batches
            k = kwargs.get('mnn_k', 10)
            
            for i, batch1 in enumerate(unique_batches[:-1]):
                for batch2 in unique_batches[i+1:]:
                    mask1 = batch_labels == batch1
                    mask2 = batch_labels == batch2
                    
                    data1 = embeddings[mask1]
                    data2 = embeddings[mask2]
                    
                    if data1.shape[0] > k and data2.shape[0] > k:
                        # Find nearest neighbors
                        nn1 = NearestNeighbors(n_neighbors=min(k, data2.shape[0]))
                        nn2 = NearestNeighbors(n_neighbors=min(k, data1.shape[0]))
                        
                        nn1.fit(data2)
                        nn2.fit(data1)
                        
                        # Find MNNs and apply correction
                        distances1, indices1 = nn1.kneighbors(data1)
                        distances2, indices2 = nn2.kneighbors(data2)
                        
                        # Simple correction: move towards batch centroid
                        centroid1 = data1.mean(axis=0)
                        centroid2 = data2.mean(axis=0)
                        correction_vector = (centroid1 + centroid2) / 2
                        
                        # Apply partial correction
                        alpha = kwargs.get('mnn_alpha', 0.3)
                        corrected[mask1] = (1 - alpha) * data1 + alpha * correction_vector
                        corrected[mask2] = (1 - alpha) * data2 + alpha * correction_vector
            
            print(f"     MNN-inspired correction applied")
            return corrected
            
        except Exception as e:
            print(f"     MNN correction failed: {e}, using original embeddings")
            return embeddings
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply simple centering and scaling correction."""
        print("   Applying center-scale correction...")
        
        corrected = embeddings.copy()
        unique_batches = np.unique(batch_labels)
        
        # Calculate global statistics
        global_mean = embeddings.mean(axis=0)
        global_std = embeddings.std(axis=0) + 1e-8
        
        # Correct each batch
        for batch_id in unique_batches:
            batch_mask = batch_labels == batch_id
            batch_data = corrected[batch_mask]
            
            if batch_data.shape[0] > 1:
                # Center to global mean
                batch_mean = batch_data.mean(axis=0)
                batch_centered = batch_data - batch_mean + global_mean
                
                # Scale to global variance
                batch_std = batch_data.std(axis=0) + 1e-8
                scale_factor = global_std / batch_std
                scale_factor = np.clip(scale_factor, 0.5, 2.0)  # Avoid extreme scaling
                
                corrected[batch_mask] = batch_centered * scale_factor
        
        print(f"     Center-scale correction applied to {len(unique_batches)} batches")
        return corrected
    
    def fine_tune(self, 
                  train_adata: AnnData,
                  valid_adata: Optional[AnnData] = None,
                  task: str = "annotation",
                  **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the scGPT model for cell type annotation.
        
        Args:
            train_adata: Training data with 'celltype' in .obs
            valid_adata: Validation data (optional)
            task: Task type ('annotation')
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input data
        if 'celltype' not in train_adata.obs:
            raise ValueError("train_adata must have 'celltype' column in .obs")
        
        print(f"🚀 Starting fine-tuning for {task} task...")
        
        # Get training parameters
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)
        mask_ratio = kwargs.get('mask_ratio', 0.0)  # No masking for annotation task
        
        # Prepare cell type mapping
        unique_celltypes = train_adata.obs['celltype'].astype('category').cat.categories
        celltype_to_id = {ct: i for i, ct in enumerate(unique_celltypes)}
        id_to_celltype = {i: ct for i, ct in enumerate(unique_celltypes)}
        n_celltypes = len(unique_celltypes)
        
        print(f"Found {n_celltypes} cell types: {list(unique_celltypes)}")
        
        # Add celltype_id to data
        train_adata.obs['celltype_id'] = train_adata.obs['celltype'].map(celltype_to_id)
        if valid_adata is not None:
            if 'celltype' not in valid_adata.obs:
                raise ValueError("valid_adata must have 'celltype' column in .obs")
            valid_adata.obs['celltype_id'] = valid_adata.obs['celltype'].map(celltype_to_id)
        
        # Preprocess data
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None
        
        # Update model's classifier if needed
        if hasattr(self.model, 'cls_decoder') and hasattr(self.model.cls_decoder, 'out_layer'):
            current_n_cls = self.model.cls_decoder.out_layer.out_features
            if current_n_cls != n_celltypes:
                print(f"🔄 Updating classifier: {current_n_cls} → {n_celltypes} classes")
                # Get input dimension
                if hasattr(self.model.cls_decoder, '_decoder') and len(self.model.cls_decoder._decoder) > 0:
                    prev_layer = self.model.cls_decoder._decoder[-2]
                    if hasattr(prev_layer, 'out_features'):
                        prev_dim = prev_layer.out_features
                    else:
                        prev_dim = self.model.d_model
                else:
                    prev_dim = self.model.d_model
                
                # Replace classifier
                import torch.nn as nn
                self.model.cls_decoder.out_layer = nn.Linear(prev_dim, n_celltypes)
                self.model.cls_decoder.out_layer.to(self.device)
                print("✓ Classifier updated")
        
        # Prepare training data
        train_results = self._prepare_training_data(train_processed, mask_ratio)
        valid_results = self._prepare_training_data(valid_processed, mask_ratio) if valid_processed is not None else None
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self._train_epoch(
                train_results, optimizer, criterion, batch_size
            )
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            
            # Validation
            if valid_results is not None:
                val_loss, val_acc = self._validate_epoch(valid_results, criterion, batch_size)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    print(f"✓ New best validation accuracy: {best_accuracy:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_model_state = copy.deepcopy(self.model.state_dict())
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✓ Loaded best model with accuracy: {best_accuracy:.4f}")
        
        # Store celltype mapping for inference
        self.celltype_mapping = {
            'celltype_to_id': celltype_to_id,
            'id_to_celltype': id_to_celltype,
            'n_celltypes': n_celltypes
        }
        
        print("🎉 Fine-tuning completed!")
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'celltype_mapping': self.celltype_mapping,
            'n_celltypes': n_celltypes
        }
    
    def train_integration(self, 
                         train_adata: AnnData,
                         valid_adata: Optional[AnnData] = None,
                         batch_key: str = "batch",
                         **kwargs) -> Dict[str, Any]:
        """
        Train the scGPT model for batch integration following Tutorial_Integration.
        
        Args:
            train_adata: Training data with batch labels in .obs[batch_key]
            valid_adata: Validation data (optional)
            batch_key: Column name for batch labels (default: "batch")
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input data
        if batch_key not in train_adata.obs:
            raise ValueError(f"train_adata must have '{batch_key}' column in .obs")
        
        print(f"🚀 Starting scGPT integration training...")
        
        # Get training parameters (Tutorial_Integration defaults)
        epochs = kwargs.get('epochs', 20)  # Integration typically needs more epochs
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)
        mask_ratio = kwargs.get('mask_ratio', 0.4)  # Higher masking for integration
        dab_weight = kwargs.get('dab_weight', 1.0)
        ecs_weight = kwargs.get('ecs_weight', 10.0)
        gepc_weight = kwargs.get('gepc_weight', 1.0)
        
        # Process batch labels
        train_batch_labels = train_adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(train_batch_labels)
        num_batches = len(unique_batches)
        
        print(f"Batch information:")
        print(f"  Number of batches: {num_batches}")
        print(f"  Training cells distribution: {np.bincount(train_batch_labels)}")
        
        if valid_adata is not None:
            if batch_key not in valid_adata.obs:
                raise ValueError(f"valid_adata must have '{batch_key}' column in .obs")
            valid_batch_labels = valid_adata.obs[batch_key].astype('category').cat.codes.values
            print(f"  Validation cells distribution: {np.bincount(valid_batch_labels)}")
        
        # Update model configuration for integration
        self.config.use_batch_labels = True
        self.config.num_batch_labels = num_batches
        self.config.do_dab = kwargs.get('do_dab', True)
        self.config.do_mvc = kwargs.get('do_mvc', True)  # GEPC
        self.config.do_ecs = kwargs.get('do_ecs', True)
        self.config.domain_spec_batchnorm = kwargs.get('domain_spec_batchnorm', True)
        
        print(f"Integration training configuration:")
        print(f"  epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")
        print(f"  mask_ratio: {mask_ratio}")
        print(f"  DAB: {self.config.do_dab}, MVC: {self.config.do_mvc}, ECS: {self.config.do_ecs}")
        print(f"  DSBN: {self.config.domain_spec_batchnorm}")
        print(f"  Loss weights - DAB: {dab_weight}, ECS: {ecs_weight}, GEPC: {gepc_weight}")
        
        # Check if model needs to be reconfigured for integration
        if hasattr(self.model, 'use_batch_labels') and not self.model.use_batch_labels:
            print("⚠ Warning: Model was not initialized with batch label support.")
            print("  You may need to reload the model with integration parameters.")
        
        # Preprocess data
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None
        
        # Prepare integration training data
        train_results = self._prepare_integration_data(train_processed, train_batch_labels, mask_ratio)
        valid_results = self._prepare_integration_data(valid_processed, valid_batch_labels, mask_ratio) if valid_processed is not None else None
        
        # Setup training components
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Integration uses multiple loss functions
        mse_criterion = torch.nn.MSELoss()
        dab_criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_loss = float('inf')
        best_model_state = None
        training_history = {
            'train_loss': [], 'train_mse': [], 'train_dab': [], 'train_ecs': [],
            'val_loss': [], 'val_mse': [], 'val_dab': [], 'val_ecs': []
        }
        
        for epoch in range(epochs):
            print(f"\n📊 Integration Epoch {epoch+1}/{epochs}")
            
            # Training
            train_losses = self._train_integration_epoch(
                train_results, optimizer, mse_criterion, dab_criterion, 
                batch_size, dab_weight, ecs_weight, gepc_weight
            )
            
            for key, value in train_losses.items():
                training_history[f'train_{key}'].append(value)
            
            # Validation
            if valid_results is not None:
                val_losses = self._validate_integration_epoch(
                    valid_results, mse_criterion, dab_criterion, 
                    batch_size, dab_weight, ecs_weight, gepc_weight
                )
                
                for key, value in val_losses.items():
                    training_history[f'val_{key}'].append(value)
                
                total_val_loss = val_losses['total_loss']
                print(f"Train Total: {train_losses['total_loss']:.4f}, Val Total: {total_val_loss:.4f}")
                print(f"Train MSE: {train_losses['mse']:.4f}, Val MSE: {val_losses['mse']:.4f}")
                if train_losses['dab'] > 0:
                    print(f"Train DAB: {train_losses['dab']:.4f}, Val DAB: {val_losses['dab']:.4f}")
                if train_losses['ecs'] > 0:
                    print(f"Train ECS: {train_losses['ecs']:.4f}, Val ECS: {val_losses['ecs']:.4f}")
                
                # Save best model
                if total_val_loss < best_loss:
                    best_loss = total_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    print(f"✓ New best validation loss: {best_loss:.4f}")
            else:
                total_train_loss = train_losses['total_loss']
                print(f"Train Total: {total_train_loss:.4f}")
                if total_train_loss < best_loss:
                    best_loss = total_train_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✓ Loaded best model with loss: {best_loss:.4f}")
        
        # Store integration metadata
        self.integration_metadata = {
            'batch_key': batch_key,
            'num_batches': num_batches,
            'batch_labels': {i: f"batch_{i}" for i in range(num_batches)}
        }
        
        # Mark as integration trained
        self._integration_trained = True
        
        print("🎉 Integration training completed!")
        
        return {
            'best_loss': best_loss,
            'training_history': training_history,
            'integration_metadata': self.integration_metadata,
            'num_batches': num_batches
        }
    
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Get cell embeddings.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings
        """
        result = self.predict(adata, task="embedding", **kwargs)
        return result["embeddings"]
    
    def _prepare_training_data(self, adata: AnnData, mask_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """Prepare data for training."""
        # Get expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        genes = adata.var_names.tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        
        # Tokenize
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Apply masking
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        
        # Get labels
        celltype_labels = torch.from_numpy(adata.obs['celltype_id'].values.astype(int)).long()
        
        return {
            "gene_ids": tokenized_data["genes"],
            "values": input_values,
            "target_values": tokenized_data["values"],
            "celltype_labels": celltype_labels,
        }
    
    def _prepare_integration_data(self, adata: AnnData, batch_labels: np.ndarray, mask_ratio: float = 0.4) -> Dict[str, torch.Tensor]:
        """Prepare data for integration training."""
        # Get expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        genes = adata.var_names.tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        
        # Tokenize
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Apply masking (higher ratio for integration)
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        
        return {
            "gene_ids": tokenized_data["genes"],
            "values": input_values,
            "target_values": tokenized_data["values"],
            "batch_labels": torch.from_numpy(batch_labels).long(),
        }
    
    def _train_integration_epoch(self, data: Dict[str, torch.Tensor], optimizer, mse_criterion, dab_criterion, 
                               batch_size: int, dab_weight: float, ecs_weight: float, gepc_weight: float) -> Dict[str, float]:
        """Train integration for one epoch."""
        self.model.train()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        total_mse = 0.0
        total_dab = 0.0
        total_ecs = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
            
            optimizer.zero_grad()
            
            # Integration model forward pass
            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.config.use_batch_labels else None,
                CLS=False,                    # No classification for integration
                CCE=False,                    # Can be enabled if needed
                MVC=self.config.do_mvc,      # Masked Value Prediction (GEPC)
                ECS=self.config.do_ecs,      # Elastic Cell Similarity
                do_sample=False,
            )
            
            # Compute integration losses
            loss = 0.0
            mse_loss = 0.0
            dab_loss = 0.0
            ecs_loss = 0.0
            
            # 1. MLM (Masked Language Modeling) loss - reconstruction
            if "mlm_output" in output_dict:
                # Create mask for loss computation
                masked_positions = (input_values == -1)  # -1 is mask_value
                if masked_positions.any():
                    mse_loss = mse_criterion(
                        output_dict["mlm_output"][masked_positions],
                        target_values[masked_positions]
                    )
                    loss += mse_loss
            
            # 2. GEPC (MVC) loss - Gene Expression Prediction for Cells
            if self.config.do_mvc and "mvc_output" in output_dict:
                masked_positions = (input_values == -1)
                if masked_positions.any():
                    gepc_loss = mse_criterion(
                        output_dict["mvc_output"][masked_positions],
                        target_values[masked_positions]
                    )
                    loss += gepc_weight * gepc_loss
            
            # 3. DAB (Domain Adversarial Batch) loss
            if self.config.do_dab and "dab_output" in output_dict:
                dab_loss = dab_criterion(output_dict["dab_output"], batch_labels)
                loss += dab_weight * dab_loss
            
            # 4. ECS (Elastic Cell Similarity) loss
            if self.config.do_ecs and "loss_ecs" in output_dict:
                ecs_loss = output_dict["loss_ecs"]
                loss += ecs_weight * ecs_loss
            
            if torch.isnan(loss):
                print("⚠ Warning: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
            total_dab += dab_loss.item() if isinstance(dab_loss, torch.Tensor) else dab_loss
            total_ecs += ecs_loss.item() if isinstance(ecs_loss, torch.Tensor) else ecs_loss
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'dab': total_dab / num_batches,
            'ecs': total_ecs / num_batches,
        }
    
    def _validate_integration_epoch(self, data: Dict[str, torch.Tensor], mse_criterion, dab_criterion,
                                  batch_size: int, dab_weight: float, ecs_weight: float, gepc_weight: float) -> Dict[str, float]:
        """Validate integration for one epoch."""
        self.model.eval()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_mse = 0.0
        total_dab = 0.0
        total_ecs = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if self.config.use_batch_labels else None,
                    CLS=False,
                    CCE=False,
                    MVC=self.config.do_mvc,
                    ECS=self.config.do_ecs,
                    do_sample=False,
                )
                
                # Compute validation losses (same as training)
                loss = 0.0
                mse_loss = 0.0
                dab_loss = 0.0
                ecs_loss = 0.0
                
                if "mlm_output" in output_dict:
                    masked_positions = (input_values == -1)
                    if masked_positions.any():
                        mse_loss = mse_criterion(
                            output_dict["mlm_output"][masked_positions],
                            target_values[masked_positions]
                        )
                        loss += mse_loss
                
                if self.config.do_mvc and "mvc_output" in output_dict:
                    masked_positions = (input_values == -1)
                    if masked_positions.any():
                        gepc_loss = mse_criterion(
                            output_dict["mvc_output"][masked_positions],
                            target_values[masked_positions]
                        )
                        loss += gepc_weight * gepc_loss
                
                if self.config.do_dab and "dab_output" in output_dict:
                    dab_loss = dab_criterion(output_dict["dab_output"], batch_labels)
                    loss += dab_weight * dab_loss
                
                if self.config.do_ecs and "loss_ecs" in output_dict:
                    ecs_loss = output_dict["loss_ecs"]
                    loss += ecs_weight * ecs_loss
                
                total_loss += loss.item()
                total_mse += mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
                total_dab += dab_loss.item() if isinstance(dab_loss, torch.Tensor) else dab_loss
                total_ecs += ecs_loss.item() if isinstance(ecs_loss, torch.Tensor) else ecs_loss
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'dab': total_dab / num_batches,
            'ecs': total_ecs / num_batches,
        }
    
    def _train_epoch(self, data: Dict[str, torch.Tensor], optimizer, criterion, batch_size: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_data in dataloader:
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)
            
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
            
            optimizer.zero_grad()
            
            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
            
            if "cls_output" in output_dict:
                logits = output_dict["cls_output"]
                loss = criterion(logits, celltype_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = logits.argmax(1)
                total_correct += (predictions == celltype_labels).sum().item()
                total_samples += celltype_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, data: Dict[str, torch.Tensor], criterion, batch_size: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                
                if "cls_output" in output_dict:
                    logits = output_dict["cls_output"]
                    loss = criterion(logits, celltype_labels)
                    
                    total_loss += loss.item()
                    predictions = logits.argmax(1)
                    total_correct += (predictions == celltype_labels).sum().item()
                    total_samples += celltype_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def predict_celltypes(self, query_adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict cell types for query data using fine-tuned model.
        
        Args:
            query_adata: Query data to predict
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not hasattr(self, 'celltype_mapping'):
            raise ValueError("Model has not been fine-tuned. Call fine_tune() first.")
        
        print("🔍 Predicting cell types for query data...")
        
        # Preprocess query data
        query_processed = self.preprocess(query_adata, **kwargs)
        
        # Get predictions
        results = self._predict_annotation(query_processed, **kwargs)
        
        if 'predictions' in results:
            # Convert IDs to cell type names
            id_to_celltype = self.celltype_mapping['id_to_celltype']
            predicted_celltypes = [id_to_celltype.get(pred, f"Unknown_{pred}") 
                                 for pred in results['predictions']]
            
            results['predicted_celltypes'] = predicted_celltypes
            
            print(f"✓ Predicted cell types for {len(predicted_celltypes)} cells")
            
            # Show prediction summary
            from collections import Counter
            type_counts = Counter(predicted_celltypes)
            print("Prediction summary:")
            for celltype, count in type_counts.most_common():
                percentage = count / len(predicted_celltypes) * 100
                print(f"  {celltype}: {count} cells ({percentage:.1f}%)")
        
        return results
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """Save scGPT model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path / "model.pt")
        
        if self.vocab is not None:
            self.vocab.save_json(save_path / "vocab.json")
        
        if self.config is not None:
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        
        # Save celltype mapping if available
        if hasattr(self, 'celltype_mapping'):
            with open(save_path / "celltype_mapping.json", 'w') as f:
                json.dump(self.celltype_mapping, f, indent=2)
            print("✓ Saved celltype mapping")
    
    def load_celltype_mapping(self, model_path: Union[str, Path]) -> None:
        """Load celltype mapping from saved model."""
        model_path = Path(model_path)
        mapping_file = model_path / "celltype_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                self.celltype_mapping = json.load(f)
            print(f"✓ Loaded celltype mapping with {self.celltype_mapping['n_celltypes']} types")
        else:
            print("⚠ No celltype mapping found")


class SimpleDataset(Dataset):
    """Simple dataset class for tokenized data."""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        # Determine the key to use for length
        self.length_key = None
        for key in ["genes", "gene_ids"]:
            if key in data:
                self.length_key = key
                break
        if self.length_key is None:
            raise ValueError("No suitable key found for dataset length")
    
    def __len__(self):
        return self.data[self.length_key].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}