"""
Unified output utilities for SCLLM models.

This module provides consistent progress bars, status messages, and formatting
across all SCLLM models (scGPT, scFoundation, Geneformer, CellPLM).
"""

import sys
from typing import Optional, Any, Dict, List
from tqdm import tqdm
import contextlib
from io import StringIO

try:
    from .message_standards import MessageStandards, DebugFilter, ModelTerminology
except ImportError:
    from message_standards import MessageStandards, DebugFilter, ModelTerminology


class SCLLMOutput:
    """Unified output manager for SCLLM models."""
    
    # Unified emoji and formatting standards
    EMOJIS = {
        # Model operations
        'loading': '[Loading]',
        'saving': '[Saving]', 
        'loaded': '[Loaded]',
        'failed': '[Failed]',
        'warning': '[Warning]',
        'info': '[ℹ️Info]',
        
        # Training/Processing
        'training': '[Training]',
        'predicting': '[Predicting]',
        'embedding': '[Embedding]',
        'integrating': '[Integrating]',
        'fine_tuning': '[Fine-tuning]',
        'preprocessing': '[Preprocessing]',
        'postprocessing': '[Postprocessing]',
        
        # Progress
        'epoch': '[♻️Epoch]',
        'batch': '[Batch]',
        'complete': '[✅Complete]',
        'best': '[🏆Best]',
        
        # Data
        'cells': '[🔬Cells]',
        'genes': '[🧬Genes]',
        'batches': '[Batches]',
    }
    
    @staticmethod
    def status(message: str, emoji_key: Optional[str] = None, indent: int = 0) -> None:
        """Print status message with consistent formatting."""
        prefix = "    " * indent
        if emoji_key and emoji_key in SCLLMOutput.EMOJIS:
            emoji = SCLLMOutput.EMOJIS[emoji_key]
            print(f"{prefix}{emoji} {message}")
        else:
            print(f"{prefix}{message}")
    
    @staticmethod
    def progress_bar(iterable=None, total=None, desc="Processing", 
                    model_name: Optional[str] = None, silent: bool = True, **kwargs) -> tqdm:
        """Create unified progress bar.
        
        Args:
            iterable: Iterable to wrap with progress bar
            total: Total iterations if iterable is None
            desc: Description for the progress bar
            model_name: Model name to add to description
            silent: If True, progress bar will not remain after completion (default: True)
            **kwargs: Additional arguments passed to tqdm
        """
        # Set default parameters for consistency
        defaults = {
            'ncols': 100,
            'unit': 'it',
            'leave': not silent,  # Progress bar disappears when silent=True
            'dynamic_ncols': True,
        }
        defaults.update(kwargs)
        
        # Add model name to description if provided
        if model_name:
            desc = f"[{model_name}] {desc}"
        
        return tqdm(iterable=iterable, total=total, desc=desc, **defaults)
    
    @staticmethod
    def progress_range(start_or_stop, stop=None, step=1, desc="Processing", 
                      model_name: Optional[str] = None, silent: bool = True, **kwargs) -> tqdm:
        """Create progress bar for range iterations (similar to trange).
        
        Args:
            start_or_stop: If stop is None, this is the stop value (range from 0). 
                          Otherwise, this is the start value.
            stop: Stop value (exclusive)
            step: Step size
            desc: Description for the progress bar
            model_name: Model name to add to description
            silent: If True, progress bar will not remain after completion (default: True)
            **kwargs: Additional arguments passed to tqdm
        """
        if stop is None:
            start, stop = 0, start_or_stop
        else:
            start = start_or_stop
            
        # Create range and wrap with progress bar
        return SCLLMOutput.progress_bar(
            iterable=range(start, stop, step),
            desc=desc,
            model_name=model_name,
            silent=silent,
            **kwargs
        )
    
    @staticmethod
    def section_header(title: str, model_name: Optional[str] = None) -> None:
        """Print section header with consistent formatting."""
        if model_name:
            title = f"[{model_name}] {title}"
        print(f"\n{'='*60}")
        print(f"{title.upper()}")
        print(f"{'='*60}")
    
    @staticmethod
    def subsection(title: str, indent: int = 0) -> None:
        """Print subsection with consistent formatting."""
        prefix = "    " * indent
        print(f"\n{prefix}{'-'*40}")
        print(f"{prefix}{title}")
        print(f"{prefix}{'-'*40}")
    
    @staticmethod
    def model_info(model_name: str, info: Dict[str, Any], indent: int = 0) -> None:
        """Print model information in consistent format."""
        SCLLMOutput.status(f"{model_name} Model Information:", 'info', indent)
        for key, value in info.items():
            SCLLMOutput.status(f"{key}: {value}", indent=indent+1)
    
    @staticmethod
    def training_metrics(epoch: int, total_epochs: int, metrics: Dict[str, float], 
                        model_name: Optional[str] = None, indent: int = 0) -> None:
        """Print training metrics in consistent format."""
        prefix = "    " * indent
        model_prefix = f"[{model_name}] " if model_name else ""
        
        print(f"{prefix}📊 {model_prefix}Epoch {epoch}/{total_epochs}")
        for metric_name, value in metrics.items():
            print(f"{prefix}  {metric_name}: {value:.4f}")
    
    @staticmethod
    def data_summary(adata, model_name: Optional[str] = None, indent: int = 0) -> None:
        """Print data summary in consistent format."""
        SCLLMOutput.status(f"Data Summary:", 'cells', indent)
        SCLLMOutput.status(f"Cells: {adata.n_obs:,}", indent=indent+1)
        SCLLMOutput.status(f"Genes: {adata.n_vars:,}", indent=indent+1)
        
        # Show batch information if available
        if 'batch' in adata.obs.columns:
            batch_counts = adata.obs['batch'].value_counts()
            SCLLMOutput.status(f"Batches: {len(batch_counts)}", indent=indent+1)
            for batch, count in batch_counts.items():
                SCLLMOutput.status(f"  {batch}: {count:,} cells", indent=indent+1)
    
    @staticmethod
    def results_summary(results: Dict[str, Any], model_name: Optional[str] = None, 
                       indent: int = 0) -> None:
        """Print results summary in consistent format."""
        SCLLMOutput.status(MessageStandards.RESULTS_SUMMARY, 'complete', indent)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    SCLLMOutput.status(f"{key}: {value:.4f}", indent=indent+1)
                else:
                    SCLLMOutput.status(f"{key}: {value:,}", indent=indent+1)
            else:
                SCLLMOutput.status(f"{key}: {value}", indent=indent+1)
    
    # Convenience methods using MessageStandards
    @staticmethod
    def loading_model(model_name: str) -> None:
        """Standard model loading message."""
        SCLLMOutput.status(MessageStandards.LOADING_MODEL, 'loading')
    
    @staticmethod
    def model_loaded(model_name: str) -> None:
        """Standard model loaded message."""
        SCLLMOutput.status(MessageStandards.MODEL_LOADED, 'loaded')
    
    @staticmethod
    def preprocessing_data() -> None:
        """Standard preprocessing message."""
        SCLLMOutput.status(MessageStandards.PREPROCESSING_START, 'preprocessing')
    
    @staticmethod
    def preprocessing_complete() -> None:
        """Standard preprocessing complete message."""
        SCLLMOutput.status(MessageStandards.PREPROCESSING_COMPLETE, 'loaded')
    
    @staticmethod
    def running_inference() -> None:
        """Standard inference message."""
        SCLLMOutput.status(MessageStandards.PREDICTING_START, 'predicting')
    
    @staticmethod
    def extracting_embeddings() -> None:
        """Standard embedding extraction message."""
        SCLLMOutput.status(MessageStandards.EMBEDDING_START, 'embedding')
    
    @staticmethod
    def format_data_info(n_obs: int, n_vars: int) -> str:
        """Format data information consistently."""
        return MessageStandards.format_data_info(n_obs, n_vars)
    
    @staticmethod
    def format_embedding_info(shape: tuple) -> str:
        """Format embedding information consistently."""
        return MessageStandards.format_embedding_info(shape)


class ModelProgressManager:
    """Context manager for model operations with consistent progress tracking."""
    
    def __init__(self, operation: str, model_name: str, total_steps: Optional[int] = None, silent: bool = True):
        self.operation = operation
        self.model_name = model_name
        self.total_steps = total_steps
        self.silent = silent
        self.pbar = None
        self.step_count = 0
    
    def __enter__(self):
        if not self.silent:
            SCLLMOutput.status(f"Starting {self.operation}...", 'info')
        if self.total_steps:
            self.pbar = SCLLMOutput.progress_bar(
                total=self.total_steps,
                desc=self.operation,
                model_name=self.model_name,
                silent=self.silent
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        
        if not self.silent:
            if exc_type is None:
                SCLLMOutput.status(f"{self.operation} completed successfully!", 'complete')
            else:
                SCLLMOutput.status(f"{self.operation} failed: {exc_val}", 'failed')
    
    def update(self, n: int = 1, **kwargs):
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(n)
            # Update postfix with additional info
            if kwargs:
                self.pbar.set_postfix(**kwargs)
        self.step_count += n
    
    def set_description(self, desc: str):
        """Update progress bar description."""
        if self.pbar:
            self.pbar.set_description(f"[{self.model_name}] {desc}")


# Convenience functions for specific operations
def loading_model(model_name: str, model_path: str = "") -> None:
    """Standard model loading message."""
    path_msg = f" from {model_path}" if model_path else ""
    SCLLMOutput.status(f"Loading {model_name} model{path_msg}...", 'loading')

def model_loaded(model_name: str, info: Optional[Dict[str, Any]] = None) -> None:
    """Standard model loaded message."""
    SCLLMOutput.status(f"{model_name} model loaded successfully", 'loaded')
    if info:
        for key, value in info.items():
            SCLLMOutput.status(f"{key}: {value}", indent=1)

def batch_progress(start: int, total_length: int, batch_size: int, desc: str = "Processing batches",
                  model_name: Optional[str] = None, silent: bool = True, **kwargs) -> tqdm:
    """Create progress bar for batch processing iterations.
    
    Args:
        start: Starting index
        total_length: Total number of items to process
        batch_size: Size of each batch
        desc: Description for the progress bar
        model_name: Model name to add to description
        silent: If True, progress bar will not remain after completion (default: True)
        **kwargs: Additional arguments passed to tqdm
    
    Returns:
        tqdm progress bar for batch processing
    """
    return SCLLMOutput.progress_range(
        start, total_length, batch_size,
        desc=desc,
        model_name=model_name,
        silent=silent,
        **kwargs
    )

def operation_start(operation: str, model_name: str, data_info: Optional[Dict[str, Any]] = None) -> None:
    """Standard operation start message."""
    emoji_map = {
        'predict_celltypes': 'predicting',
        'get_embeddings': 'embedding', 
        'fine_tune': 'fine_tuning',
        'integrate': 'integrating',
        'preprocess': 'preprocessing',
    }
    emoji = emoji_map.get(operation, 'info')
    
    SCLLMOutput.status(f"Starting {operation}...", emoji)
    if data_info:
        for key, value in data_info.items():
            SCLLMOutput.status(f"{key}: {value}", indent=1)

def operation_complete(operation: str, results: Optional[Dict[str, Any]] = None) -> None:
    """Standard operation complete message."""
    SCLLMOutput.status(f"{operation} completed successfully!", 'complete')
    if results:
        SCLLMOutput.results_summary(results)


# Context manager for capturing and filtering verbose output
@contextlib.contextmanager
def suppress_verbose_output(keep_progress: bool = True):
    """
    Context manager to suppress verbose output while keeping progress bars.
    
    Args:
        keep_progress: If True, keeps tqdm progress bars visible
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    if keep_progress:
        # Create a custom stdout that filters out non-progress messages
        stdout_buffer = StringIO()
        sys.stdout = stdout_buffer
        # Keep stderr for progress bars (tqdm uses stderr by default)
    else:
        # Suppress everything
        sys.stdout = StringIO()
        sys.stderr = StringIO()
    
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr