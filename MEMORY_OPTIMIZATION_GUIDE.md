# Memory Optimization Guide for OmicVerse Single-Cell Analysis

This guide addresses memory issues when processing large single-cell datasets with OmicVerse, particularly during `lazy` function execution.

## Problem Description

The original `ov.single.lazy()` function processes all steps in memory, causing crashes on large datasets (typically >100k cells) during memory-intensive operations like:

1. **MDE computation** (`pymde.preserve_neighbors`) - High memory usage for neighbor graph construction
2. **Harmony batch correction** - Creates full data copies for batch integration  
3. **SCCAF clustering** - Iterative clustering with RandomForest on full feature matrices
4. **scVI processing** - Deep learning model training requires substantial GPU/CPU memory

## Solutions

### 1. Checkpointing Version: `lazy_checkpoint()`

Automatically saves intermediate results at each step, allowing resume on crash.

```python
import omicverse as ov
import scanpy as sc

# Load your data
adata = sc.read_h5ad('large_dataset.h5ad')

# Run with checkpointing
adata = ov.single.lazy_checkpoint(
    adata,
    species='human',
    sample_key='batch',
    checkpoint_dir='./checkpoints',
    save_intermediate=True,
    # Optional: customize parameters
    harmony_kwargs={'n_pcs': 50},
    scvi_kwargs={'n_latent': 30}
)
```

**Checkpoint Management:**
```python
# List available checkpoints
ov.single.list_checkpoints('./checkpoints')

# Resume from specific step
adata = ov.single.resume_from_checkpoint('./checkpoints', 'harmony')

# Clean up intermediate files (keep final result)
ov.single.cleanup_checkpoints('./checkpoints', keep_final=True)
```

### 2. Step-by-Step Processing

Run each step individually with manual memory management:

```python
import omicverse as ov
import scanpy as sc

# Print the step-by-step guide
ov.single.lazy_step_by_step_guide()

# Load your data
adata = sc.read_h5ad('large_dataset.h5ad')

# Step 1: Quality Control
adata = ov.single.lazy_step_qc(adata, sample_key='batch', 
                               output_path='step1_qc.h5ad')

# Step 2: Preprocessing  
adata = ov.single.lazy_step_preprocess(adata, 
                                      output_path='step2_preprocess.h5ad')

# Step 3: Scaling
adata = ov.single.lazy_step_scale(adata, 
                                 output_path='step3_scale.h5ad')

# Step 4: PCA
adata = ov.single.lazy_step_pca(adata, 
                               output_path='step4_pca.h5ad')

# Step 5: Cell cycle scoring
adata = ov.single.lazy_step_cell_cycle(adata, species='human',
                                      output_path='step5_cell_cycle.h5ad')

# Step 6a: Harmony batch correction (MEMORY INTENSIVE!)
adata = ov.single.lazy_step_harmony(adata, sample_key='batch',
                                   output_path='step6a_harmony.h5ad')

# Step 6b: scVI batch correction (OPTIONAL - VERY MEMORY INTENSIVE!)
# Skip this step if memory is limited
# adata = ov.single.lazy_step_scvi(adata, sample_key='batch',
#                                 output_path='step6b_scvi.h5ad')

# Step 7: Select best batch correction method
adata = ov.single.lazy_step_select_best_method(adata,
                                              output_path='step7_best_method.h5ad')

# Step 8: MDE embedding (MEMORY INTENSIVE!)
adata = ov.single.lazy_step_mde(adata, 
                               output_path='step8_mde.h5ad')

# Step 9: Clustering (MOST MEMORY INTENSIVE!)
adata = ov.single.lazy_step_clustering(adata, 
                                      output_path='step9_clustering.h5ad',
                                      max_iterations=5)  # Reduce iterations if needed

# Step 10: Final embeddings
adata = ov.single.lazy_step_final_embeddings(adata,
                                            output_path='step10_final.h5ad')
```

## Memory Optimization Strategies

### 1. Pre-processing Optimization

- **Filter cells/genes early**: Remove low-quality cells before processing
- **Subset highly variable genes**: Use only HVG subset for memory-intensive steps
- **Reduce n_PCs**: Use fewer principal components (e.g., 30 instead of 50)

### 2. Hardware Recommendations

**Minimum Requirements for Large Datasets (>100k cells):**
- RAM: 32GB+ (64GB+ recommended for >500k cells)
- Storage: 100GB+ free space for checkpoints
- CPU: 8+ cores for parallel processing

**Memory-constrained Systems:**
- Skip scVI step (use Harmony only)
- Reduce SCCAF iterations (`max_iterations=3`)
- Process in smaller batches
- Use SSD for faster I/O during checkpointing

### 3. Runtime Optimization

```python
# Between memory-intensive steps, restart Python to free memory
import gc
import os

def restart_kernel():
    os.execv(sys.executable, ['python'] + sys.argv)

# Or manually clean up
del adata_hvg  # Remove intermediate objects
gc.collect()   # Force garbage collection
```

## Memory-Intensive Steps Warning System

The new functions include warnings for memory-intensive operations:

- ⚠️ **Step 6a (Harmony)**: Moderate memory usage - processes HVG subset
- ⚠️ **Step 6b (scVI)**: High memory usage - deep learning model training  
- ⚠️ **Step 8 (MDE)**: High memory usage - neighbor graph computation
- ⚠️ **Step 9 (Clustering)**: Highest memory usage - SCCAF iterative clustering

## Troubleshooting

### Common Issues

1. **Jupyter Kernel Crashes**: 
   - Solution: Use checkpointing version or step-by-step processing
   - Monitor memory with `htop` or Task Manager

2. **"Out of Memory" Errors**:
   - Reduce dataset size or increase RAM
   - Skip optional scVI step
   - Reduce SCCAF iterations

3. **Long Processing Times**:
   - Use SSD storage for checkpoints
   - Reduce `n_neighbors` parameter
   - Process smaller subsets

### Recovery from Crashes

```python
# If processing crashes, resume from last checkpoint
adata = ov.single.resume_from_checkpoint('./checkpoints', 'harmony')

# Continue from where you left off
adata = ov.single.lazy_step_mde(adata, output_path='step8_mde.h5ad')
# ... continue with remaining steps
```

## Performance Comparison

| Method | Memory Usage | Resume on Crash | Flexibility | Recommended For |
|--------|-------------|-----------------|-------------|----------------|
| `lazy()` | High | ❌ | Low | Small datasets (<50k cells) |
| `lazy_checkpoint()` | High | ✅ | Medium | Medium datasets (50k-200k cells) |
| Step-by-step | Variable | ✅ | High | Large datasets (>200k cells) |

## Best Practices

1. **Always save intermediate results** for large datasets
2. **Monitor system memory** during processing
3. **Test on subsets first** to estimate memory requirements
4. **Use appropriate hardware** for your dataset size
5. **Plan for storage space** - checkpoints can be large
6. **Clean up checkpoints** after successful completion

This optimization guide should help you successfully process large single-cell datasets without memory crashes.