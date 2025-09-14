#!/usr/bin/env python3
"""
Sample Data Generator for OmicVerse Single Cell Analysis Platform

This script generates sample single-cell data for testing the web platform.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os
from pathlib import Path

def create_sample_data(n_cells=2000, n_genes=3000, n_clusters=8, output_dir=None):
    """
    Create sample single-cell data with realistic structure
    
    Parameters:
    -----------
    n_cells : int
        Number of cells to generate
    n_genes : int  
        Number of genes to generate
    n_clusters : int
        Number of cell clusters
    output_dir : str or None
        Output directory for the h5ad file
    """
    
    print(f"🧬 生成示例单细胞数据...")
    print(f"   - 细胞数量: {n_cells}")
    print(f"   - 基因数量: {n_genes}")
    print(f"   - 聚类数量: {n_clusters}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate gene names (mix of real-like gene names)
    gene_prefixes = ['ENSG', 'ENSM', 'LOC', 'MIR', 'LINC']
    common_genes = [
        'GAPDH', 'ACTB', 'TUBB', 'RPL13A', 'RPS18',
        'CD3D', 'CD4', 'CD8A', 'CD19', 'CD14',
        'IL2', 'IL4', 'IL6', 'IL10', 'TNF',
        'IFNG', 'TGF', 'VEGFA', 'MYC', 'TP53',
        'FOXP3', 'GATA3', 'TBX21', 'RORC', 'BCL6'
    ]
    
    # Generate gene names
    gene_names = common_genes.copy()
    while len(gene_names) < n_genes:
        prefix = np.random.choice(gene_prefixes)
        number = np.random.randint(10000, 99999)
        gene_names.append(f"{prefix}{number}")
    
    gene_names = gene_names[:n_genes]
    
    # Generate cell barcodes
    cell_barcodes = [f"CELL_{i:06d}" for i in range(n_cells)]
    
    # Generate cluster assignments
    cluster_labels = np.random.choice(range(n_clusters), size=n_cells)
    
    # Generate expression matrix with cluster structure
    print("📊 生成表达矩阵...")
    
    # Base expression levels
    base_expression = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Add cluster-specific expression patterns
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        n_cluster_cells = np.sum(cluster_mask)
        
        # Select marker genes for this cluster (10% of genes)
        n_markers = n_genes // 10
        marker_genes = np.random.choice(n_genes, n_markers, replace=False)
        
        # Increase expression of marker genes in this cluster
        base_expression[cluster_mask][:, marker_genes] *= np.random.uniform(2, 5, size=(n_cluster_cells, n_markers))
    
    # Create AnnData object
    print("🔧 创建AnnData对象...")
    adata = sc.AnnData(X=base_expression)
    adata.obs_names = cell_barcodes
    adata.var_names = gene_names
    
    # Add metadata
    adata.obs['cluster'] = cluster_labels.astype(str)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')
    
    # Add some continuous variables
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1)
    adata.obs['total_counts'] = np.sum(adata.X, axis=1)
    adata.obs['mt_frac'] = np.random.beta(2, 20, n_cells)  # Mitochondrial fraction
    adata.obs['ribo_frac'] = np.random.beta(5, 10, n_cells)  # Ribosomal fraction
    
    # Add some categorical variables
    cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'DC', 'Neutrophil', 'Eosinophil', 'Other']
    adata.obs['cell_type'] = np.random.choice(cell_types, n_cells)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    # Add batch information
    adata.obs['batch'] = np.random.choice(['Batch_1', 'Batch_2', 'Batch_3'], n_cells)
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    
    # Add gene metadata
    adata.var['highly_variable'] = False
    hvg_indices = np.random.choice(n_genes, n_genes//4, replace=False)
    adata.var.iloc[hvg_indices, adata.var.columns.get_loc('highly_variable')] = True
    
    # Add some gene statistics
    adata.var['n_cells'] = np.sum(adata.X > 0, axis=0)
    adata.var['mean_expression'] = np.mean(adata.X, axis=0)
    adata.var['std_expression'] = np.std(adata.X, axis=0)
    
    # Perform basic preprocessing and dimensionality reduction
    print("🔄 执行预处理和降维...")
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Store raw data
    adata.raw = adata
    
    # Feature selection
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_hvg = adata[:, adata.var.highly_variable]
    
    # Scale data
    sc.pp.scale(adata_hvg, max_value=10)
    
    # PCA
    sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=40)
    
    # UMAP
    sc.tl.umap(adata_hvg)
    
    # t-SNE
    sc.tl.tsne(adata_hvg, perplexity=30)
    
    # Leiden clustering
    sc.tl.leiden(adata_hvg, resolution=0.5)
    
    # Copy embeddings and clustering back to original object
    adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
    adata.obsm['X_tsne'] = adata_hvg.obsm['X_tsne']
    adata.obs['leiden'] = adata_hvg.obs['leiden']
    adata.uns = adata_hvg.uns
    
    # Save the data
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_data.h5ad"
    
    print(f"💾 保存数据到: {output_path}")
    adata.write_h5ad(output_path)
    
    print("✅ 示例数据生成完成!")
    print(f"   - 文件路径: {output_path}")
    print(f"   - 文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   - 包含降维结果: PCA, UMAP, t-SNE")
    print(f"   - 包含聚类结果: Leiden clustering")
    
    return str(output_path)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample single-cell data')
    parser.add_argument('--cells', type=int, default=2000, help='Number of cells (default: 2000)')
    parser.add_argument('--genes', type=int, default=3000, help='Number of genes (default: 3000)')
    parser.add_argument('--clusters', type=int, default=8, help='Number of clusters (default: 8)')
    parser.add_argument('--output', type=str, help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    print("🧪 OmicVerse 示例数据生成器")
    print("=" * 40)
    
    try:
        output_path = create_sample_data(
            n_cells=args.cells,
            n_genes=args.genes, 
            n_clusters=args.clusters,
            output_dir=args.output
        )
        
        print("\n🎯 使用说明:")
        print("1. 启动网页服务器:")
        print("   python start_server.py")
        print("2. 在浏览器中上传生成的文件:")
        print(f"   {output_path}")
        print("3. 开始分析!")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
