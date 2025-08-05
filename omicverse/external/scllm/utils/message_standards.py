"""
Unified messaging standards for SCLLM models.

This module defines consistent, concise messages that should be used across
all SCLLM models (scGPT, scFoundation, Geneformer, CellPLM).
"""

class MessageStandards:
    """Standard message templates for consistent output across all models."""
    
    # Model Loading Messages
    LOADING_MODEL = "Loading model..."
    MODEL_LOADED = "Model loaded successfully"
    LOADING_VOCAB = "Loading vocabulary..."
    VOCAB_LOADED = "Vocabulary loaded: {count:,} genes"
    LOADING_CONFIG = "Loading configuration..."
    CONFIG_LOADED = "Configuration loaded"
    
    # Preprocessing Messages
    PREPROCESSING_START = "Preprocessing data..."
    PREPROCESSING_COMPLETE = "Preprocessing complete"
    TOKENIZING = "Tokenizing genes..."
    NORMALIZING = "Normalizing expression data..."
    FILTERING_GENES = "Filtering genes by vocabulary..."
    BINNING_DATA = "Binning expression values..."
    
    # Training Messages
    TRAINING_START = "Starting training..."
    TRAINING_COMPLETE = "Training completed"
    FINE_TUNING_START = "Starting fine-tuning..."
    FINE_TUNING_COMPLETE = "Fine-tuning completed"
    BEST_MODEL_SAVED = "New best model saved"
    MODEL_RESTORED = "Best model restored"
    
    # Prediction Messages
    PREDICTING_START = "Running inference..."
    PREDICTING_COMPLETE = "Prediction complete"
    EMBEDDING_START = "Extracting embeddings..."
    EMBEDDING_COMPLETE = "Embedding extraction complete"
    INTEGRATION_START = "Performing batch integration..."
    INTEGRATION_COMPLETE = "Batch integration complete"
    
    # Results Messages
    RESULTS_SUMMARY = "Results summary:"
    CELL_DISTRIBUTION = "Cell type distribution:"
    BATCH_INFO = "Batch information:"
    
    # Warning Messages
    GENES_NOT_FOUND = "Some genes not found in vocabulary"
    USING_DEFAULT = "Using default configuration"
    MODEL_NEEDS_TRAINING = "Model requires fine-tuning for this task"
    
    # Success Messages
    OPERATION_SUCCESS = "Operation completed successfully"
    
    @staticmethod
    def format_data_info(n_obs: int, n_vars: int) -> str:
        """Standard format for data dimensions."""
        return f"Data: {n_obs:,} cells × {n_vars:,} genes"
    
    @staticmethod
    def format_embedding_info(shape: tuple) -> str:
        """Standard format for embedding dimensions."""
        return f"Embeddings: {shape[0]:,} cells × {shape[1]} dimensions"
    
    @staticmethod
    def format_batch_info(batch_counts: dict) -> str:
        """Standard format for batch information."""
        total_batches = len(batch_counts)
        return f"Batches: {total_batches} detected"
    
    @staticmethod
    def format_celltype_info(n_types: int) -> str:
        """Standard format for cell type information."""
        return f"Cell types: {n_types} classes"
    
    @staticmethod
    def format_training_epoch(epoch: int, total: int) -> str:
        """Standard format for training epoch."""
        return f"Epoch {epoch}/{total}"
    
    @staticmethod
    def format_accuracy(accuracy: float) -> str:
        """Standard format for accuracy display."""
        return f"Accuracy: {accuracy:.4f}"
    
    @staticmethod
    def format_loss(loss: float) -> str:
        """Standard format for loss display.""" 
        return f"Loss: {loss:.4f}"


class DebugFilter:
    """Filter for determining which debug messages to keep/remove."""
    
    # Messages that should be removed (too verbose)
    REMOVE_PATTERNS = [
        "shape:",
        "range:",
        "dtype:",
        "device:",
        "tensor info",
        "debug info",
        "checking",
        "found keys:",
        "processing batch",
    ]
    
    # Messages that should be simplified  
    SIMPLIFY_PATTERNS = {
        "Successfully loaded all model weights": "Model weights loaded",
        "Loaded compatible weights": "Compatible weights loaded", 
        "Processing cells...": "Processing data...",
        "Running prediction loop...": "Running inference...",
        "Starting prediction loop...": "Running inference...",
        "Generated embeddings shape": "Embeddings extracted",
        "Created dataloader with": "Data loader ready",
        "Starting integration inference...": "Running integration...",
        "Prediction Analysis:": "Results:",
    }
    
    @staticmethod
    def should_remove(message: str) -> bool:
        """Check if a debug message should be removed."""
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in DebugFilter.REMOVE_PATTERNS)
    
    @staticmethod
    def simplify_message(message: str) -> str:
        """Simplify a verbose message."""
        for verbose, simple in DebugFilter.SIMPLIFY_PATTERNS.items():
            if verbose in message:
                return simple
        return message


class ModelTerminology:
    """Ensure consistent terminology across all models."""
    
    # Standardized terms
    TERMS = {
        # Instead of: "cell type annotation", "celltype prediction", "cell classification"
        'cell_annotation': 'cell type annotation',
        
        # Instead of: "batch correction", "batch effect removal", "integration"  
        'batch_integration': 'batch integration',
        
        # Instead of: "cell embeddings", "representations", "latent vectors"
        'embeddings': 'embeddings',
        
        # Instead of: "fine-tuning", "training", "adaptation"
        'fine_tuning': 'fine-tuning',
        
        # Instead of: "inference", "prediction", "evaluation"
        'inference': 'inference',
        
        # Instead of: "preprocessing", "data preparation", "normalization"
        'preprocessing': 'preprocessing',
    }
    
    @staticmethod
    def standardize_term(term: str) -> str:
        """Convert various terms to standardized versions."""
        term_lower = term.lower()
        
        # Cell type related
        if any(x in term_lower for x in ['cell type', 'celltype', 'cell classification']):
            return 'cell type annotation'
        
        # Batch related  
        if any(x in term_lower for x in ['batch correction', 'batch effect', 'integration']):
            return 'batch integration'
            
        # Embedding related
        if any(x in term_lower for x in ['representation', 'latent', 'encoding']):
            return 'embeddings'
            
        return term