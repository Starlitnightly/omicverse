#!/usr/bin/env python
"""
scFoundation Usage Examples for omicverse
==========================================

This file demonstrates how to use the newly integrated scFoundation model
within the omicverse.external.scllm framework.
"""

import omicverse as ov
import numpy as np
import pandas as pd
import scanpy as sc


def example_basic_usage():
    """
    Basic usage example of scFoundation with SCLLMManager.
    """
    print("=== Basic scFoundation Usage ===")
    
    # Load your data
    # adata = sc.read_h5ad('your_data.h5ad')
    
    # Method 1: Using SCLLMManager (Recommended)
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/your/scfoundation/model.ckpt",
        device="cuda"  # or "cpu"
    )
    
    # Get cell embeddings (default: cell-level embeddings with 'all' pooling)
    embeddings = manager.get_embeddings(adata)
    print(f"Cell embeddings shape: {embeddings.shape}")
    
    # The embeddings are automatically added to adata.obsm['X_scfoundation']
    # You can use them for downstream analysis like clustering, UMAP, etc.


def example_different_embedding_types():
    """
    Example showing different types of embeddings available in scFoundation.
    """
    print("=== Different Embedding Types ===")
    
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # 1. Cell embeddings with different pooling strategies
    cell_emb_all = manager.get_embeddings(adata, output_type="cell", pool_type="all")
    cell_emb_max = manager.get_embeddings(adata, output_type="cell", pool_type="max")
    
    # 2. Gene embeddings (for each cell individually)
    gene_embeddings = manager.get_embeddings(adata, output_type="gene")
    
    # 3. Gene embeddings (batch processing - more efficient for large datasets)
    gene_emb_batch = manager.get_embeddings(adata, output_type="gene_batch")
    
    print(f"Cell embeddings (all pooling): {cell_emb_all.shape}")
    print(f"Cell embeddings (max pooling): {cell_emb_max.shape}")
    print(f"Gene embeddings: {gene_embeddings.shape}")
    print(f"Gene embeddings (batch): {gene_emb_batch.shape}")


def example_convenience_functions():
    """
    Example using convenience functions for quick operations.
    """
    print("=== Using Convenience Functions ===")
    
    # Quick embedding extraction
    embeddings = ov.external.scllm.get_embeddings_with_scfoundation(
        adata=adata,
        model_path="/path/to/model.ckpt",
        output_type="cell",
        pool_type="all",
        device="cuda"
    )
    
    # End-to-end workflow with automatic saving
    results = ov.external.scllm.end_to_end_scfoundation_embedding(
        adata=adata,
        model_path="/path/to/model.ckpt",
        save_embeddings_path="./scfoundation_embeddings.npy",
        output_type="cell",
        pool_type="all",
        device="cuda"
    )
    
    print("Embeddings extracted and saved automatically!")


def example_preprocessing_options():
    """
    Example showing different preprocessing options.
    """
    print("=== Preprocessing Options ===")
    
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # For raw count data (default)
    embeddings_raw = manager.get_embeddings(
        adata, 
        pre_normalized="F",  # False - will apply normalization
        input_type="singlecell"
    )
    
    # For already normalized data
    embeddings_norm = manager.get_embeddings(
        adata,
        pre_normalized="T",  # True - data already normalized
        input_type="singlecell"
    )
    
    # For bulk RNA-seq data
    embeddings_bulk = manager.get_embeddings(
        adata,
        pre_normalized="F",
        input_type="bulk"
    )
    
    print("Different preprocessing modes handled!")


def example_resolution_control():
    """
    Example showing resolution control for scFoundation.
    """
    print("=== Resolution Control ===")
    
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # Different resolution settings
    # Target resolution (t4 means resolution = 4)
    emb_t4 = manager.get_embeddings(adata, tgthighres="t4")
    
    # Fold change of resolution (f2 means 2x the original resolution)
    emb_f2 = manager.get_embeddings(adata, tgthighres="f2")
    
    # Addition to resolution (a1 means +1 to log resolution)
    emb_a1 = manager.get_embeddings(adata, tgthighres="a1")
    
    print("Different resolution settings applied!")


def example_downstream_analysis():
    """
    Example showing how to use scFoundation embeddings for downstream analysis.
    """
    print("=== Downstream Analysis ===")
    
    # Get scFoundation embeddings
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    embeddings = manager.get_embeddings(adata)
    
    # Add embeddings to adata
    adata.obsm['X_scfoundation'] = embeddings
    
    # Use embeddings for clustering
    sc.pp.neighbors(adata, use_rep='X_scfoundation')
    sc.tl.leiden(adata, resolution=0.5)
    
    # Use embeddings for UMAP
    sc.tl.umap(adata)
    
    # Visualize
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data')
    
    print("Downstream analysis completed using scFoundation embeddings!")


def example_model_comparison():
    """
    Example comparing scGPT and scFoundation embeddings.
    """
    print("=== Model Comparison ===")
    
    # Get scGPT embeddings
    scgpt_manager = ov.external.scllm.SCLLMManager(
        model_type="scgpt",
        model_path="/path/to/scgpt/model"
    )
    scgpt_embeddings = scgpt_manager.get_embeddings(adata)
    
    # Get scFoundation embeddings  
    scfoundation_manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/scfoundation/model.ckpt"
    )
    scfoundation_embeddings = scfoundation_manager.get_embeddings(adata)
    
    # Add both to adata for comparison
    adata.obsm['X_scgpt'] = scgpt_embeddings
    adata.obsm['X_scfoundation'] = scfoundation_embeddings
    
    # Compare clustering results
    sc.pp.neighbors(adata, use_rep='X_scgpt', key_added='scgpt')
    sc.pp.neighbors(adata, use_rep='X_scfoundation', key_added='scfoundation')
    
    sc.tl.leiden(adata, neighbors_key='scgpt', key_added='leiden_scgpt')
    sc.tl.leiden(adata, neighbors_key='scfoundation', key_added='leiden_scfoundation')
    
    print("Model comparison setup complete!")


# Example usage patterns
USAGE_PATTERNS = {
    "basic": """
# Basic usage
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation",
    model_path="model.ckpt"
)
embeddings = manager.get_embeddings(adata)
""",
    
    "convenience": """
# Using convenience functions
embeddings = ov.external.scllm.get_embeddings_with_scfoundation(
    adata, "model.ckpt", output_type="cell"
)
""",
    
    "end_to_end": """
# End-to-end workflow
results = ov.external.scllm.end_to_end_scfoundation_embedding(
    adata=adata,
    model_path="model.ckpt", 
    save_embeddings_path="embeddings.npy"
)
""",
    
    "different_outputs": """
# Different output types
cell_emb = manager.get_embeddings(adata, output_type="cell")
gene_emb = manager.get_embeddings(adata, output_type="gene") 
gene_batch_emb = manager.get_embeddings(adata, output_type="gene_batch")
""",
    
    "preprocessing": """
# Preprocessing control
embeddings = manager.get_embeddings(
    adata,
    pre_normalized="F",  # Raw counts
    input_type="singlecell",
    tgthighres="t4"  # Resolution control
)
"""
}


if __name__ == "__main__":
    print("scFoundation Integration Examples")
    print("=" * 40)
    print()
    print("This file contains examples of how to use scFoundation")
    print("with the omicverse.external.scllm framework.")
    print()
    print("Key features added:")
    print("- ScFoundationModel class")
    print("- Integration with SCLLMManager")
    print("- Convenience functions")
    print("- Multiple embedding types (cell, gene, gene_batch)")
    print("- Preprocessing options")
    print("- Resolution control")
    print()
    
    for pattern_name, pattern_code in USAGE_PATTERNS.items():
        print(f"=== {pattern_name.upper()} USAGE ===")
        print(pattern_code)
        print()
    
    print("For more details, see the individual example functions above!")


def example_fine_tuning():
    """
    Example showing how to fine-tune scFoundation for cell type annotation.
    """
    print("=== scFoundation Fine-tuning ===")
    
    # Load pretrained model
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/pretrained/model.ckpt"
    )
    
    # Fine-tune on reference data (must have 'celltype' in .obs)
    fine_tune_results = manager.fine_tune(
        train_adata=reference_adata,
        valid_adata=validation_adata,  # Optional
        epochs=10,
        batch_size=32,
        lr=1e-3,
        frozen_more=True  # Freeze token/position embeddings
    )
    
    print(f"Best validation accuracy: {fine_tune_results['best_accuracy']:.3f}")
    
    # Predict on query data using fine-tuned model
    predictions = manager.predict(query_adata, task="annotation")
    
    print("Fine-tuning and prediction completed!")


def example_fine_tuning_convenience():
    """
    Example using convenience functions for fine-tuning workflow.
    """
    print("=== Fine-tuning Convenience Functions ===")
    
    # Method 1: Step-by-step workflow
    # Step 1: Fine-tune
    fine_tune_results = ov.external.scllm.fine_tune_scfoundation(
        train_adata=reference_adata,
        model_path="/path/to/pretrained/model.ckpt",
        valid_adata=validation_adata,
        save_path="/path/to/save/finetuned_model",
        epochs=15,
        lr=5e-4
    )
    
    # Step 2: Predict
    predictions = ov.external.scllm.predict_celltypes_with_scfoundation(
        query_adata=query_adata,
        finetuned_model_path="/path/to/save/finetuned_model",
        save_predictions=True
    )
    
    # Method 2: End-to-end workflow
    results = ov.external.scllm.end_to_end_scfoundation_annotation(
        reference_adata=reference_adata,
        query_adata=query_adata,
        pretrained_model_path="/path/to/pretrained/model.ckpt",
        save_finetuned_path="/path/to/finetuned_model",
        validation_split=0.2,
        epochs=10
    )
    
    print("End-to-end annotation workflow completed!")


def example_fine_tuning_parameters():
    """
    Example showing different fine-tuning parameters.
    """
    print("=== Fine-tuning Parameters ===")
    
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # Conservative fine-tuning (freeze more layers)
    conservative_results = manager.fine_tune(
        train_adata=reference_adata,
        epochs=5,
        batch_size=16,
        lr=1e-4,
        frozen_more=True  # Freeze token and position embeddings
    )
    
    # Aggressive fine-tuning (unfreeze more layers)
    aggressive_results = manager.fine_tune(
        train_adata=reference_adata,
        epochs=20,
        batch_size=64,
        lr=1e-3,
        frozen_more=False  # Allow token/position embeddings to update
    )
    
    print("Different fine-tuning strategies completed!")


# Update usage patterns to include fine-tuning
USAGE_PATTERNS.update({
    "fine_tuning_basic": """
# Basic fine-tuning
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation", 
    model_path="pretrained.ckpt"
)
results = manager.fine_tune(
    train_adata=reference_data,  # Must have 'celltype' in .obs
    epochs=10,
    batch_size=32
)
predictions = manager.predict(query_data, task="annotation")
""",
    
    "fine_tuning_convenience": """
# End-to-end fine-tuning workflow
results = ov.external.scllm.end_to_end_scfoundation_annotation(
    reference_adata=reference_data,
    query_adata=query_data,
    pretrained_model_path="pretrained.ckpt",
    save_finetuned_path="finetuned.ckpt",
    validation_split=0.2,
    epochs=15
)
""",
    
    "fine_tuning_custom": """
# Custom fine-tuning parameters
results = manager.fine_tune(
    train_adata=reference_data,
    valid_adata=validation_data,
    epochs=20,
    batch_size=64,
    lr=5e-4,
    frozen_more=False,  # Allow more layers to update
    n_classes=8  # Override auto-detected class count
)
"""
})


def example_batch_integration():
    """
    Example showing how to perform batch integration with scFoundation.
    """
    print("=== scFoundation Batch Integration ===")
    
    # Basic integration
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # Integrate batches using Harmony (default)
    integration_results = manager.predict(
        adata,  # Must have batch information in .obs['batch']
        task="integration",
        batch_key="batch",
        correction_method="harmony"
    )
    
    # Results contain both original and integrated embeddings
    integrated_embeddings = integration_results["embeddings"]
    original_embeddings = integration_results["original_embeddings"]
    
    print("Batch integration completed!")


def example_integration_methods():
    """
    Example showing different integration methods.
    """
    print("=== Different Integration Methods ===")
    
    manager = ov.external.scllm.SCLLMManager(
        model_type="scfoundation",
        model_path="/path/to/model.ckpt"
    )
    
    # Method 1: Harmony (fast and effective)
    harmony_results = manager.predict(
        adata, task="integration", 
        correction_method="harmony",
        max_iter_harmony=10
    )
    
    # Method 2: ComBat (classic batch correction)
    combat_results = manager.predict(
        adata, task="integration",
        correction_method="combat"
    )
    
    # Method 3: Scanorama (mutual nearest neighbors)
    scanorama_results = manager.predict(
        adata, task="integration",
        correction_method="scanorama"
    )
    
    # Method 4: MNN (scanpy implementation)
    mnn_results = manager.predict(
        adata, task="integration",
        correction_method="mnn"
    )
    
    print("Tested all integration methods!")


def example_integration_convenience():
    """
    Example using convenience functions for integration workflow.
    """
    print("=== Integration Convenience Functions ===")
    
    # Method 1: Single integration method
    results = ov.external.scllm.integrate_with_scfoundation(
        query_adata=adata,
        model_path="/path/to/model.ckpt",
        batch_key="batch",
        correction_method="harmony",
        save_embeddings=True  # Automatically saves to adata.obsm
    )
    
    # Method 2: Test multiple methods
    multi_results = ov.external.scllm.end_to_end_scfoundation_integration(
        query_adata=adata,
        model_path="/path/to/model.ckpt",
        batch_key="batch",
        correction_methods=["harmony", "combat", "scanorama"],
        save_all_methods=True,  # Save all methods for comparison
        max_iter_harmony=15
    )
    
    print("Integration workflows completed!")


def example_integration_evaluation():
    """
    Example showing how to evaluate integration results.
    """
    print("=== Integration Evaluation ===")
    
    # Perform integration
    results = ov.external.scllm.integrate_with_scfoundation(
        query_adata=adata,
        model_path="/path/to/model.ckpt",
        correction_method="harmony"
    )
    
    # Add embeddings to adata for evaluation
    adata.obsm['X_integrated'] = results["embeddings"]
    adata.obsm['X_original'] = results["original_embeddings"]
    
    # Evaluate integration using scanpy
    sc.pp.neighbors(adata, use_rep='X_integrated')
    sc.tl.umap(adata)
    
    # Visualize results
    sc.pl.umap(adata, color='batch', title='After Integration')
    
    # Compare with original embeddings
    sc.pp.neighbors(adata, use_rep='X_original')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='batch', title='Before Integration')
    
    print("Integration evaluation completed!")


# Update usage patterns to include integration
USAGE_PATTERNS.update({
    "integration_basic": """
# Basic batch integration
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation", 
    model_path="model.ckpt"
)
results = manager.predict(
    adata,  # Must have 'batch' in .obs
    task="integration",
    correction_method="harmony"
)
""",
    
    "integration_convenience": """
# Using convenience functions
results = ov.external.scllm.integrate_with_scfoundation(
    query_adata=adata,
    model_path="model.ckpt",
    batch_key="batch",
    correction_method="harmony"
)
""",
    
    "integration_multi_method": """
# Test multiple integration methods
results = ov.external.scllm.end_to_end_scfoundation_integration(
    query_adata=adata,
    model_path="model.ckpt", 
    correction_methods=["harmony", "combat", "scanorama"],
    save_all_methods=True
)
""",
    
    "integration_custom": """
# Integration with custom parameters
results = manager.predict(
    adata,
    task="integration",
    correction_method="harmony",
    max_iter_harmony=20,
    random_state=42
)
"""
})

    print("For more details, see the individual example functions above!")