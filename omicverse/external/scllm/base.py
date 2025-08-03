"""
Base classes for scLLM models.
Provides a unified interface for different single-cell language models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import torch
import numpy as np
from anndata import AnnData


class SCLLMBase(ABC):
    """
    Base class for single-cell language models.
    
    This provides a unified interface for different models like scGPT, scBERT, etc.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.vocab = None
        self.preprocessor = None
        self.is_loaded = False
        
    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """Setup the device for model execution."""
        if device == "auto" or device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model directory or file
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess the data for model input.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed AnnData object
        """
        pass
    
    @abstractmethod
    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using the model.
        
        Args:
            adata: Input AnnData object
            task: Task type ('annotation', 'integration', 'generation', etc.)
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        pass
    
    @abstractmethod
    def fine_tune(self, 
                  train_adata: AnnData,
                  valid_adata: Optional[AnnData] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the model on new data.
        
        Args:
            train_adata: Training data
            valid_adata: Validation data (optional)
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Get cell embeddings from the model.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        pass
    
    def save_model(self, save_path: Union[str, Path], **kwargs) -> None:
        """
        Save the current model state.
        
        Args:
            save_path: Path to save the model
            **kwargs: Additional save parameters
        """
        if not self.is_loaded:
            raise ValueError("No model loaded to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # This will be implemented by subclasses
        self._save_model_specific(save_path, **kwargs)
    
    @abstractmethod
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """Model-specific save implementation."""
        pass
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(model_name={self.model_name}, device={self.device}, status={status})"


class ModelConfig:
    """Configuration class for model parameters."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class TaskConfig:
    """Configuration for specific tasks."""
    
    ANNOTATION_CONFIG = {
        'mask_ratio': 0.0,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-4,
        'CLS': True,
        'MLM': False,
        'MVC': False,
        'ADV': False,
        'CCE': False,
        'ECS': False,
        'DAB': False,
    }
    
    INTEGRATION_CONFIG = {
        'mask_ratio': 0.15,
        'epochs': 20,
        'batch_size': 64,
        'lr': 1e-4,
        'CLS': False,
        'MLM': True,
        'MVC': True,
        'ADV': True,
        'CCE': True,
        'ECS': True,
        'DAB': True,
    }
    
    GENERATION_CONFIG = {
        'mask_ratio': 0.15,
        'epochs': 15,
        'batch_size': 32,
        'lr': 5e-5,
        'CLS': False,
        'MLM': True,
        'MVC': True,
        'ADV': False,
        'CCE': False,
        'ECS': False,
        'DAB': False,
    }
    
    @classmethod
    def get_task_config(cls, task: str) -> Dict[str, Any]:
        """Get configuration for a specific task."""
        task_configs = {
            'annotation': cls.ANNOTATION_CONFIG,
            'integration': cls.INTEGRATION_CONFIG,
            'generation': cls.GENERATION_CONFIG,
        }
        
        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(task_configs.keys())}")
        
        return task_configs[task].copy()