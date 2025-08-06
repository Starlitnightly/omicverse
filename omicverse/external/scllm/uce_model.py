"""
UCE model implementation for omicverse.

This module provides a wrapper for the UCE (Universal Cell Embeddings) model,
following the exact logic of the original eval_single_anndata.py script.
"""

import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from anndata import AnnData
from scipy.sparse import issparse

try:
    from .base import SCLLMBase, ModelConfig, TaskConfig
    from .utils.output_utils import SCLLMOutput, ModelProgressManager
except ImportError:
    from base import SCLLMBase, ModelConfig, TaskConfig
    from utils.output_utils import SCLLMOutput, ModelProgressManager


class UCEModel(SCLLMBase):
    """
    UCE (Universal Cell Embeddings) model wrapper.
    
    This implementation follows the exact logic of eval_single_anndata.py
    to ensure identical results.
    """
    
    def __init__(self, device: Optional[str] = None, **kwargs):
        """
        Initialize UCE model.
        
        Args:
            device: Device to run the model on ('cpu' or 'cuda')
            **kwargs: Additional UCE-specific parameters
        """
        super().__init__(model_name="UCE", device=device)
        
        # UCE configuration parameters (matching eval_single_anndata.py defaults)
        self.config = {
            'nlayers': kwargs.get('nlayers', 4),
            'output_dim': kwargs.get('output_dim', 1280),
            'd_hid': kwargs.get('d_hid', 5120),
            'token_dim': kwargs.get('token_dim', 5120),
            'nhead': 20,  # Fixed in UCE
            'dropout': 0.05,  # Fixed in UCE
            'pad_length': kwargs.get('pad_length', 1536),
            'sample_size': kwargs.get('sample_size', 1024),
            'batch_size': kwargs.get('batch_size', 25),
            'species': kwargs.get('species', 'human'),
            'multi_gpu': kwargs.get('multi_gpu', False),
            # Token indices (UCE defaults)
            'pad_token_idx': 0,
            'chrom_token_left_idx': 1,
            'chrom_token_right_idx': 2,
            'cls_token_idx': 3,
            'CHROM_TOKEN_OFFSET': 143574
        }
        
        # File paths (all external, no hardcoding)
        self.model_path = None
        self.token_file = None
        self.protein_embeddings_dir = None
        self.spec_chrom_csv_path = None
        self.offset_pkl_path = None
        
        SCLLMOutput.status("UCE model initialized", 'loaded')
    
    def load_model(self, model_path: Union[str, Path], 
                   token_file: Union[str, Path],
                   protein_embeddings_dir: Union[str, Path],
                   spec_chrom_csv_path: Union[str, Path],
                   offset_pkl_path: Union[str, Path],
                   **kwargs) -> None:
        """
        Load UCE model and required assets.
        
        Args:
            model_path: Path to UCE model weights (.torch file)
            token_file: Path to token embeddings file
            protein_embeddings_dir: Path to protein embeddings directory
            spec_chrom_csv_path: Path to species chromosome CSV file
            offset_pkl_path: Path to species offsets pickle file
            **kwargs: Additional parameters
        """
        SCLLMOutput.status("Loading UCE model and assets", 'loading')
        
        # Store all required file paths
        self.model_path = Path(model_path)
        self.token_file = Path(token_file)
        self.protein_embeddings_dir = Path(protein_embeddings_dir)
        self.spec_chrom_csv_path = Path(spec_chrom_csv_path)
        self.offset_pkl_path = Path(offset_pkl_path)
        
        # Validate and report all required files
        required_files = [
            (self.model_path, "Model weights"),
            (self.token_file, "Token embeddings"),
            (self.spec_chrom_csv_path, "Species chromosome mapping"),
            (self.offset_pkl_path, "Species offsets")
        ]
        
        SCLLMOutput.status("=== UCE Asset Files Validation ===", 'loading')
        for file_path, description in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"{description} not found: {file_path}")
            else:
                SCLLMOutput.status(f"✓ {description}: {file_path}", 'loaded')
        
        if not self.protein_embeddings_dir.exists():
            raise FileNotFoundError(f"Protein embeddings directory not found: {self.protein_embeddings_dir}")
        else:
            SCLLMOutput.status(f"✓ Protein embeddings directory: {self.protein_embeddings_dir}", 'loaded')
            
        # Check and report key protein embedding files
        key_protein_files = [
            "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"
        ]
        
        for filename in key_protein_files:
            file_path = self.protein_embeddings_dir / filename
            if file_path.exists():
                SCLLMOutput.status(f"✓ Found protein embedding: {filename}", 'loaded')
            else:
                SCLLMOutput.status(f"⚠ Missing protein embedding: {filename}", 'warning')
        
        # Report configuration
        SCLLMOutput.status("=== UCE Configuration ===", 'loading')
        config_items = [
            ('Species', self.config['species']),
            ('Batch size', self.config['batch_size']),
            ('Model layers', self.config['nlayers']),
            ('Output dimension', self.config['output_dim']),
            ('Token dimension', self.config['token_dim']),
            ('Hidden dimension', self.config['d_hid'])
        ]
        
        for name, value in config_items:
            SCLLMOutput.status(f"• {name}: {value}", 'loaded')
        
        # Update config with any provided parameters
        for key in ['nlayers', 'output_dim', 'd_hid', 'token_dim', 'batch_size', 'species']:
            if key in kwargs:
                old_value = self.config[key]
                self.config[key] = kwargs[key]
                if old_value != kwargs[key]:
                    SCLLMOutput.status(f"• Updated {key}: {old_value} → {kwargs[key]}", 'loaded')
        
        # Set global protein embeddings directory for UCE modules
        self._set_global_protein_embeddings_dir()
        
        self.is_loaded = True
        SCLLMOutput.status("UCE model loaded successfully", 'loaded')
    
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Extract cell embeddings using UCE model directly from memory.
        
        This method processes AnnData directly in memory without file I/O:
        1. Process adata in memory
        2. Generate protein embeddings and chromosome mappings
        3. Run UCE model inference
        4. Return embeddings
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        SCLLMOutput.status("Extracting cell embeddings using UCE", 'embedding')
        
        # Process adata directly in memory
        try:
            # Run the direct UCE workflow (no file I/O)
            embeddings = self._run_uce_workflow_direct(adata, **kwargs)
            
            SCLLMOutput.status(f"Extracted embeddings: {embeddings.shape}", 'embedding')
            return embeddings
            
        except Exception as e:
            SCLLMOutput.status(f"UCE embedding extraction failed: {e}", 'error')
            raise RuntimeError(f"UCE embedding extraction failed: {e}")
    
    def _run_uce_workflow(self, adata_path: str, output_dir: str, **kwargs) -> np.ndarray:
        """
        Run the exact UCE workflow from eval_single_anndata.py.
        
        Args:
            adata_path: Path to temporary adata file
            output_dir: Output directory for UCE processing
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        from accelerate import Accelerator
        
        # Import UCE components
        try:
            from .UCE.evaluate import AnndataProcessor, run_eval, get_ESM2_embeddings
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            # Fallback for direct execution
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from evaluate import AnndataProcessor, run_eval, get_ESM2_embeddings
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Create args object exactly like eval_single_anndata.py
        args = self._create_uce_args(adata_path, output_dir, **kwargs)
        
        # Create accelerator exactly like eval_single_anndata.py
        accelerator = Accelerator(project_dir=output_dir)
        
        # Step 1: Create processor and preprocess data
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        
        # Step 2: Load shapes_dict (needed for run_eval)
        with open(processor.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        
        # Step 3: Run evaluation with custom implementation that returns embeddings
        embeddings = self._run_eval_custom(processor.adata, processor.name, 
                                         processor.pe_idx_path, processor.chroms_path,
                                         processor.starts_path, shapes_dict, 
                                         accelerator, args)
        
        return embeddings
    
    def _run_uce_workflow_direct(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Run UCE workflow directly on memory adata object without file I/O.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        from accelerate import Accelerator
        
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            # Fallback for direct execution
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Create accelerator
        accelerator = Accelerator()
        
        # Step 1: Process adata directly in memory
        SCLLMOutput.status("Processing data in memory", 'embedding')
        processed_adata, pe_row_idxs, dataset_chroms, dataset_pos = self._process_adata_direct(adata, **kwargs)
        
        # Step 2: Create shapes dict
        shapes_dict = {"memory_adata": (processed_adata.n_obs, processed_adata.n_vars)}
        
        # Step 3: Run evaluation with direct data
        embeddings = self._run_eval_direct(
            processed_adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, accelerator
        )
        
        return embeddings
    
    def _process_adata_direct(self, adata: AnnData, **kwargs) -> Tuple[AnnData, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Process AnnData directly in memory to extract gene embeddings and chromosome information.
        
        Args:
            adata: Input AnnData object
            
        Returns:
            Tuple of (processed_adata, pe_row_idxs, dataset_chroms, dataset_pos)
        """
        # Make a copy and ensure dense matrix
        adata_processed = adata.copy()
        if issparse(adata_processed.X):
            adata_processed.X = adata_processed.X.toarray()
        
        # Apply basic filtering if requested
        filter_cells = kwargs.get('filter_cells', False)
        filter_genes = kwargs.get('filter_genes', False)
        
        if filter_cells:
            import scanpy as sc
            min_genes = kwargs.get('min_genes', 25)
            sc.pp.filter_cells(adata_processed, min_genes=min_genes)
        
        if filter_genes:
            import scanpy as sc
            min_cells = kwargs.get('min_cells', 10)
            sc.pp.filter_genes(adata_processed, min_cells=min_cells)
        
        # Load gene embeddings using our inline function
        try:
            from .UCE.data_proc.data_utils import load_gene_embeddings_adata_inline
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from data_proc.data_utils import load_gene_embeddings_adata_inline
        
        # Process gene embeddings
        adata_processed, protein_embeddings = load_gene_embeddings_adata_inline(
            adata_processed, 
            species=[self.config['species']],
            embedding_model="ESM2",
            protein_embeddings_dir=str(self.protein_embeddings_dir)
        )
        
        # Generate protein embedding indices and chromosome information
        pe_row_idxs, dataset_chroms, dataset_pos = self._generate_indices_direct(adata_processed)
        
        return adata_processed, pe_row_idxs, dataset_chroms, dataset_pos
    
    def _generate_indices_direct(self, adata_processed: AnnData) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Generate protein embedding indices and chromosome information directly from adata.
        
        Args:
            adata_processed: Processed AnnData object
            
        Returns:
            Tuple of (pe_row_idxs, dataset_chroms, dataset_pos)
        """
        # Import UCE data processing functions
        try:
            from .UCE.data_proc.data_utils import get_species_to_pe, get_spec_chrom_csv
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from data_proc.data_utils import get_species_to_pe, get_spec_chrom_csv
        
        # Load species protein embeddings
        species_to_pe = get_species_to_pe(str(self.protein_embeddings_dir))
        
        # Load species offsets
        with open(self.offset_pkl_path, "rb") as f:
            species_to_offsets = pickle.load(f)
        
        # Load chromosome position information
        gene_to_chrom_pos = get_spec_chrom_csv(str(self.spec_chrom_csv_path))
        
        # Generate indices
        dataset_species = self.config['species']
        spec_pe_genes = list(species_to_pe[dataset_species].keys())
        offset = species_to_offsets[dataset_species]
        
        # Create protein embedding row indices
        pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata_processed.var_names]).long()
        
        # Create chromosome mappings
        spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")
        gene_chrom = spec_chrom.loc[[k.upper() for k in adata_processed.var_names]]
        dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # chromosome codes
        dataset_pos = gene_chrom["start"].values  # genomic positions
        
        return pe_row_idxs, dataset_chroms, dataset_pos
    
    def _run_eval_direct(self, adata: AnnData, pe_row_idxs: torch.Tensor, 
                        dataset_chroms: np.ndarray, dataset_pos: np.ndarray,
                        shapes_dict: dict, accelerator) -> np.ndarray:
        """
        Run UCE model evaluation directly with in-memory data.
        
        Args:
            adata: Processed AnnData object
            pe_row_idxs: Protein embedding indices
            dataset_chroms: Chromosome codes
            dataset_pos: Genomic positions
            shapes_dict: Shape information dictionary
            accelerator: Accelerator object
            
        Returns:
            Cell embeddings as numpy array
        """
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        # Set random seeds for reproducibility (crucial for identical results)
        torch.manual_seed(23)
        np.random.seed(23)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #### Set up the model exactly like evaluate.py ####
        token_dim = self.config['token_dim']
        emsize = 1280  # embedding dimension (fixed in UCE)
        d_hid = self.config['d_hid']
        nlayers = self.config['nlayers']
        nhead = 20  # fixed in UCE
        dropout = 0.05  # fixed in UCE
        
        # Create model exactly like UCE
        model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                               d_hid=d_hid, nlayers=nlayers, dropout=dropout,
                               output_dim=self.config['output_dim'])
        
        # Initialize empty protein embeddings
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        
        # Load model weights
        model.load_state_dict(torch.load(str(self.model_path), map_location="cpu"), strict=True)
        
        # Load real token embeddings using UCE's function
        all_pe = self._get_ESM2_embeddings_direct()
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        SCLLMOutput.status("UCE model loaded and ready for inference", 'embedding')
        model = model.eval()
        
        # Let accelerator handle the device placement
        model = accelerator.prepare(model)
        
        # After accelerator.prepare, ensure embedding weights are on the same device as model
        if hasattr(model, 'pe_embedding') or (hasattr(model, 'module') and hasattr(model.module, 'pe_embedding')):
            pe_embedding = model.pe_embedding if hasattr(model, 'pe_embedding') else model.module.pe_embedding
            model_device = next(model.parameters()).device
            if pe_embedding.weight.device != model_device:
                pe_embedding.weight.data = pe_embedding.weight.data.to(model_device)
        
        #### Create dataset directly from memory data ####
        # Instead of using file-based dataset, create data directly
        dataset_embeds = []
        
        # Convert adata to the format expected by UCE
        X_data = adata.X.astype(np.int64)  # UCE expects integer counts
        
        # Process data in batches
        batch_size = self.config['batch_size']
        n_cells = X_data.shape[0]
        
        # Create a proper UCE-style dataset
        SCLLMOutput.status("Creating UCE dataset in memory", 'embedding')
        dataset = self._create_uce_dataset_direct(
            adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict
        )
        
        # Create data loader with UCE's collator
        try:
            from .UCE.eval_data import MultiDatasetSentenceCollator
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentenceCollator
        
        # Create args for the collator and multi-gpu check
        args = self._create_uce_args_direct()
        
        collator = MultiDatasetSentenceCollator(args)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collator, num_workers=0)
        dataloader = accelerator.prepare(dataloader)
        
        # Run inference using UCE's exact format
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="UCE inference")
        
        with torch.no_grad():
            for batch in pbar:
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
                
                # Move tensors to correct device
                device = next(model.parameters()).device
                batch_sentences = batch_sentences.to(device)
                mask = mask.to(device)
                
                # Transpose to UCE's expected format: (seq_len, batch_size)
                batch_sentences = batch_sentences.permute(1, 0)
                
                # Apply protein embeddings
                if args.multi_gpu:
                    batch_sentences = model.module.pe_embedding(batch_sentences.long())
                else:
                    batch_sentences = model.pe_embedding(batch_sentences.long())
                    
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                
                # Forward pass with proper mask
                _, embedding = model.forward(batch_sentences, mask=mask)
                
                # Gather embeddings
                accelerator.wait_for_everyone()
                embeddings = accelerator.gather_for_metrics(embedding)
                if accelerator.is_main_process:
                    dataset_embeds.append(embeddings.detach().cpu().numpy())
        
        # Combine embeddings
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_embeddings = np.vstack(dataset_embeds)
            return final_embeddings
        else:
            return np.array([])  # Non-main process returns empty
    
    def _get_ESM2_embeddings_direct(self):
        """
        Load ESM2 embeddings directly (exact copy of UCE's get_ESM2_embeddings function).
        """
        # Load ESM2 embeddings and special tokens
        all_pe = torch.load(str(self.token_file))
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, self.config['token_dim']))
            # 1895 is the total number of chromosome choices, hardcoded in UCE
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))
            all_pe.requires_grad = False
        
        return all_pe
    
    def _create_uce_args_direct(self) -> Any:
        """Create args object for direct processing (no file paths needed)."""
        class Args:
            pass
        
        args = Args()
        
        # UCE configuration (no file paths)
        args.pad_length = self.config['pad_length']
        args.sample_size = self.config['sample_size']
        args.cls_token_idx = self.config['cls_token_idx']
        args.chrom_token_left_idx = self.config['chrom_token_left_idx'] 
        args.chrom_token_right_idx = self.config['chrom_token_right_idx']
        args.pad_token_idx = self.config['pad_token_idx']
        args.CHROM_TOKEN_OFFSET = self.config['CHROM_TOKEN_OFFSET']
        args.multi_gpu = self.config['multi_gpu']
        
        return args
    
    def _create_uce_dataset_direct(self, adata: AnnData, pe_row_idxs: torch.Tensor,
                                  dataset_chroms: np.ndarray, dataset_pos: np.ndarray,
                                  shapes_dict: dict):
        """Create a UCE-style dataset from memory data."""
        
        class MemoryUCEDataset:
            def __init__(self, adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, args):
                self.adata = adata
                self.pe_row_idxs = pe_row_idxs
                self.dataset_chroms = dataset_chroms  
                self.dataset_pos = dataset_pos
                self.shapes_dict = shapes_dict
                self.args = args
                self.n_obs = adata.n_obs
                self.X_data = adata.X.astype(np.int64)  # UCE expects integer counts
                
                # Setup exactly like MultiDatasetSentences
                self.dataset_name = "memory_adata"
                self.num_cells = {self.dataset_name: self.n_obs}
                self.num_genes = {self.dataset_name: adata.n_vars}
                self.total_num_cells = self.n_obs
                
                # Create the mappings that UCE expects
                self.dataset_to_protein_embeddings = {self.dataset_name: pe_row_idxs}
                self.dataset_to_chroms = {self.dataset_name: dataset_chroms}
                self.dataset_to_starts = {self.dataset_name: dataset_pos}
            
            def __len__(self):
                return self.n_obs
                
            def __getitem__(self, idx):
                if isinstance(idx, int):
                    # Get the count data for this cell
                    counts = torch.tensor(self.X_data[idx]).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    
                    # Use UCE's sample_cell_sentences function
                    try:
                        from .UCE.eval_data import sample_cell_sentences
                    except ImportError:
                        import sys
                        uce_path = Path(__file__).parent / "UCE"  # Fixed path
                        if str(uce_path) not in sys.path:
                            sys.path.insert(0, str(uce_path))
                        from eval_data import sample_cell_sentences
                    
                    batch_sentences, mask, seq_len, cell_sentences = \
                        sample_cell_sentences(counts, weights, self.dataset_name, self.args,
                                            dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                                            dataset_to_chroms=self.dataset_to_chroms,
                                            dataset_to_starts=self.dataset_to_starts)
                    
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    raise NotImplementedError
        
        args = self._create_uce_args_direct()
        return MemoryUCEDataset(adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, args)
    
    def _create_uce_args(self, adata_path: str, output_dir: str, **kwargs):
        """Create args object matching eval_single_anndata.py argument parser."""
        class Args:
            pass
        
        args = Args()
        
        # Required arguments
        args.adata_path = adata_path
        args.dir = output_dir
        args.model_loc = str(self.model_path)
        args.species = self.config['species']
        
        # UCE asset paths
        args.token_file = str(self.token_file)
        args.protein_embeddings_dir = str(self.protein_embeddings_dir)
        args.spec_chrom_csv_path = str(self.spec_chrom_csv_path)
        args.offset_pkl_path = str(self.offset_pkl_path)
        
        # Model parameters
        args.nlayers = self.config['nlayers']
        args.output_dim = self.config['output_dim']
        args.d_hid = self.config['d_hid']
        args.token_dim = self.config['token_dim']
        args.batch_size = self.config['batch_size']
        args.pad_length = self.config['pad_length']
        args.sample_size = self.config['sample_size']
        args.multi_gpu = self.config['multi_gpu']
        
        # Token indices
        args.pad_token_idx = self.config['pad_token_idx']
        args.chrom_token_left_idx = self.config['chrom_token_left_idx']
        args.chrom_token_right_idx = self.config['chrom_token_right_idx']
        args.cls_token_idx = self.config['cls_token_idx']
        args.CHROM_TOKEN_OFFSET = self.config['CHROM_TOKEN_OFFSET']
        
        # Other parameters
        args.filter = kwargs.get('filter', False)
        args.skip = kwargs.get('skip', False)
        args.CXG = kwargs.get('CXG', False)
        
        return args
    
    def _run_eval_custom(self, adata, name, pe_idx_path, chroms_path, starts_path, 
                        shapes_dict, accelerator, args) -> np.ndarray:
        """
        Custom implementation of run_eval that returns embeddings instead of writing files.
        
        This follows the exact logic of UCE's run_eval function.
        """
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Set random seeds for reproducibility (crucial for identical results)
        torch.manual_seed(23)
        np.random.seed(23)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #### Set up the model exactly like evaluate.py ####
        token_dim = args.token_dim
        emsize = 1280  # embedding dimension (fixed in UCE)
        d_hid = args.d_hid
        nlayers = args.nlayers
        nhead = 20  # fixed in UCE
        dropout = 0.05  # fixed in UCE
        
        # Create model exactly like UCE
        model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                               d_hid=d_hid, nlayers=nlayers, dropout=dropout,
                               output_dim=args.output_dim)
        
        # Initialize empty protein embeddings
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_loc, map_location="cpu"), strict=True)
        
        # Load real token embeddings using UCE's function
        all_pe = self._get_ESM2_embeddings(args)
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        SCLLMOutput.status(f"Model loaded from: {args.model_loc}", 'loaded')
        model = model.eval()
        model = accelerator.prepare(model)
        
        #### Run the model exactly like evaluate.py ####
        # Create dataset
        dataset = MultiDatasetSentences(sorted_dataset_names=[name],
                                      shapes_dict=shapes_dict,
                                      args=args, npzs_dir=args.dir,
                                      dataset_to_protein_embeddings_path=pe_idx_path,
                                      datasets_to_chroms_path=chroms_path,
                                      datasets_to_starts_path=starts_path)
        
        # Create data loader
        multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=multi_dataset_sentence_collator, num_workers=0)
        dataloader = accelerator.prepare(dataloader)
        
        # Run inference
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        dataset_embeds = []
        
        with torch.no_grad():
            for batch in pbar:
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
                batch_sentences = batch_sentences.permute(1, 0)
                
                # Apply protein embeddings
                if args.multi_gpu:
                    batch_sentences = model.module.pe_embedding(batch_sentences.long())
                else:
                    batch_sentences = model.pe_embedding(batch_sentences.long())
                    
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                _, embedding = model.forward(batch_sentences, mask=mask)
                
                # Gather embeddings (exactly like UCE)
                accelerator.wait_for_everyone()
                embeddings = accelerator.gather_for_metrics((embedding))
                if accelerator.is_main_process:
                    dataset_embeds.append(embeddings.detach().cpu().numpy())
        
        # Combine embeddings
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_embeddings = np.vstack(dataset_embeds)
            return final_embeddings
        else:
            return np.array([])  # Non-main process returns empty
    
    def _get_ESM2_embeddings(self, args):
        """
        Exact copy of UCE's get_ESM2_embeddings function.
        """
        # Load ESM2 embeddings and special tokens
        all_pe = torch.load(args.token_file)
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
            # 1895 is the total number of chromosome choices, hardcoded in UCE
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))
            all_pe.requires_grad = False
        
        return all_pe
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess AnnData for UCE model.
        
        UCE handles preprocessing internally during the workflow,
        so this method performs minimal preprocessing.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed AnnData object
        """
        SCLLMOutput.status("Preprocessing data for UCE", 'preprocessing')
        
        adata_processed = adata.copy()
        
        # Convert sparse matrix to dense if needed
        if issparse(adata_processed.X):
            adata_processed.X = adata_processed.X.toarray()
        
        # Basic filtering (optional, UCE can handle filtering internally)
        filter_cells = kwargs.get('filter_cells', False)
        filter_genes = kwargs.get('filter_genes', False)
        
        if filter_cells:
            min_genes = kwargs.get('min_genes', 200)
            import scanpy as sc
            sc.pp.filter_cells(adata_processed, min_genes=min_genes)
            SCLLMOutput.status(f"Filtered cells with < {min_genes} genes", 'preprocessing')
        
        if filter_genes:
            min_cells = kwargs.get('min_cells', 3)
            import scanpy as sc
            sc.pp.filter_genes(adata_processed, min_cells=min_cells)
            SCLLMOutput.status(f"Filtered genes expressed in < {min_cells} cells", 'preprocessing')
        
        SCLLMOutput.status(f"Preprocessed data: {adata_processed.n_obs} cells × {adata_processed.n_vars} genes", 'preprocessing')
        
        return adata_processed
    
    def _set_global_protein_embeddings_dir(self):
        """
        Create new_species_protein_embeddings.csv file if needed.
        The protein_embeddings_dir is now passed directly via parameters.
        """
        try:
            SCLLMOutput.status("=== UCE Configuration ===", 'loading')
            SCLLMOutput.status(f"✓ Protein embeddings directory: {self.protein_embeddings_dir}", 'loaded')
            
            # Create new_species_protein_embeddings.csv file if needed
            self._create_new_species_csv()
            
            SCLLMOutput.status("✓ UCE configuration completed successfully", 'loaded')
            
        except Exception as e:
            SCLLMOutput.status(f"✗ Could not configure UCE: {e}", 'warning')
            raise
    
    def _create_new_species_csv(self):
        """
        Create new_species_protein_embeddings.csv file in the output directory.
        This prevents errors from hardcoded CSV path in data_utils.py.
        """
        # The CSV file is used in data_utils.py line 258
        # We create an empty one to prevent file not found errors
        import pandas as pd
        
        # Create empty CSV with required columns
        empty_df = pd.DataFrame(columns=['species', 'path'])
        
        # We need to create this file in multiple potential locations
        # since different UCE functions might look for it in different places
        potential_dirs = [
            Path.cwd(),  # Current working directory
            self.protein_embeddings_dir.parent,  # Parent of protein embeddings dir
        ]
        
        for directory in potential_dirs:
            csv_path = directory / "new_species_protein_embeddings.csv"
            try:
                if not csv_path.exists():
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    empty_df.to_csv(csv_path, index=False)
                    SCLLMOutput.status(f"✓ Created new_species CSV: {csv_path}", 'loaded')
                else:
                    SCLLMOutput.status(f"✓ Found existing new_species CSV: {csv_path}", 'loaded')
            except Exception as e:
                SCLLMOutput.status(f"⚠ Could not create CSV in {directory}: {e}", 'warning')
                continue  # Ignore errors for individual paths
    
    def predict(self, adata: AnnData, task: str = "embedding", **kwargs) -> Dict[str, Any]:
        """Make predictions using UCE model."""
        if task == "embedding":
            embeddings = self.get_embeddings(adata, **kwargs)
            return {
                'embeddings': embeddings,
                'task': task,
                'model_name': self.model_name
            }
        else:
            raise ValueError(f"Task '{task}' not supported by UCE model")
    
    def fine_tune(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None,
                  task: str = "embedding", **kwargs) -> Dict[str, Any]:
        """Fine-tuning not implemented for UCE."""
        SCLLMOutput.status("Fine-tuning not implemented for UCE", 'warning')
        return {
            'status': 'not_implemented',
            'message': 'UCE fine-tuning not currently supported',
            'model_name': self.model_name
        }
    
    def integrate(self, adata: AnnData, batch_key: str = "batch", 
                  correction_method: str = "mnn", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using UCE embeddings.
        
        Args:
            adata: AnnData object with batch information
            batch_key: Column name for batch labels in adata.obs
            correction_method: Batch correction method ('harmony', 'mnn', 'center_scale', 'none')
            **kwargs: Additional parameters for embedding generation
            
        Returns:
            Dictionary with integration results including corrected embeddings
        """
        SCLLMOutput.status(f"🔗 Performing batch integration for {adata.n_obs} cells", 'integration')
        
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        
        # Extract embeddings for all cells
        SCLLMOutput.status("Extracting embeddings for integration", 'integration')
        embeddings = self.get_embeddings(adata, **kwargs)
        
        # Apply batch correction
        try:
            from tqdm.auto import tqdm
            with tqdm(total=2, desc="Batch integration", ncols=100) as pbar:
                pbar.set_description(f"Applying {correction_method} correction...")
                integrated_embeddings = self._apply_batch_correction(
                    embeddings, adata.obs[batch_key], correction_method
                )
                pbar.update(1)
                pbar.set_description("Finalizing integration...")
                pbar.update(1)
            
            SCLLMOutput.status(f"Integration completed with {correction_method} correction", 'integration')
            
            return {
                'embeddings': integrated_embeddings,
                'original_embeddings': embeddings,
                'batch_key': batch_key,
                'correction_method': correction_method,
                'model_name': self.model_name,
                'task': 'integration'
            }
            
        except Exception as e:
            SCLLMOutput.status(f"Integration failed: {e}", 'error')
            raise RuntimeError(f"Batch integration failed: {e}")
    
    def _apply_batch_correction(self, embeddings: np.ndarray, batch_labels: pd.Series, method: str) -> np.ndarray:
        """Apply batch correction to embeddings."""
        import warnings
        
        if method == 'harmony':
            try:
                import harmonypy as hm
                SCLLMOutput.status("Applying Harmony correction", 'integration')
                harmony_out = hm.run_harmony(embeddings.T, batch_labels, max_iter_harmony=20)  
                return harmony_out.Z_corr.T
            except ImportError:
                warnings.warn("harmonypy not available, using MNN correction")
                method = 'mnn'
        
        if method == 'mnn':
            return self._apply_mnn_correction(embeddings, batch_labels)
        
        elif method == 'center_scale':
            return self._apply_center_scale_correction(embeddings, batch_labels)
        
        elif method == 'none':
            return embeddings
        
        else:
            warnings.warn(f"Unknown correction method {method}, using MNN correction")
            return self._apply_mnn_correction(embeddings, batch_labels)
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply MNN (Mutual Nearest Neighbors) correction - same as CellPLM implementation."""
        SCLLMOutput.status("Applying MNN correction", 'integration')
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = batch_labels.unique()
            
            if len(unique_batches) < 2:
                SCLLMOutput.status("Only one batch found, no correction needed", 'integration')
                return embeddings
            
            # Simple MNN-style correction between consecutive batches (same as CellPLM)
            for i in range(len(unique_batches) - 1):
                batch1_name = unique_batches[i]
                batch2_name = unique_batches[i + 1] if i + 1 < len(unique_batches) else unique_batches[0]
                
                batch1_mask = batch_labels == batch1_name
                batch2_mask = batch_labels == batch2_name
                
                batch1_data = corrected[batch1_mask]
                batch2_data = corrected[batch2_mask]
                
                if batch1_data.shape[0] > 5 and batch2_data.shape[0] > 5:
                    # Find mutual nearest neighbors (same algorithm as CellPLM)
                    k = min(5, min(batch1_data.shape[0], batch2_data.shape[0]) // 2)
                    
                    # Find nearest neighbors from batch2 to batch1
                    nn1 = NearestNeighbors(n_neighbors=k).fit(batch1_data)
                    distances1, indices1 = nn1.kneighbors(batch2_data)
                    
                    # Find nearest neighbors from batch1 to batch2  
                    nn2 = NearestNeighbors(n_neighbors=k).fit(batch2_data)
                    distances2, indices2 = nn2.kneighbors(batch1_data)
                    
                    # Apply simple correction by moving batches closer (same as CellPLM)
                    batch1_centroid = batch1_data.mean(axis=0)
                    batch2_centroid = batch2_data.mean(axis=0)
                    correction_vector = (batch1_centroid - batch2_centroid) * 0.5
                    
                    corrected[batch2_mask] += correction_vector
            
            SCLLMOutput.status(f"MNN correction applied to {len(unique_batches)} batches", 'integration')
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"MNN correction failed: {e}, using center_scale correction", 'warning')
            return self._apply_center_scale_correction(embeddings, batch_labels)
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply center and scale batch correction - same as CellPLM implementation."""
        SCLLMOutput.status("Applying center and scale correction", 'integration')
        
        try:
            corrected = embeddings.copy()
            
            # Calculate global statistics (same as CellPLM)
            global_mean = corrected.mean(axis=0)
            global_std = corrected.std(axis=0) + 1e-8
            
            # Correct each batch (same as CellPLM)
            for batch_name in batch_labels.unique():
                batch_mask = batch_labels == batch_name
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:
                    # Calculate batch statistics
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8
                    
                    # Center and scale to global statistics
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std * global_std + global_mean
            
            SCLLMOutput.status(f"Center-scale correction applied to {len(batch_labels.unique())} batches", 'integration')
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"Center-scale correction failed: {e}, returning original embeddings", 'warning')
            return embeddings
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """UCE model saving (not implemented)."""
        SCLLMOutput.status("Model saving not implemented for UCE", 'warning')
    
    def __repr__(self):
        return f"UCEModel(device={self.device}, loaded={self.is_loaded})"