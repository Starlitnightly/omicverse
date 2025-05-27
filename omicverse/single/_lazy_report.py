import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from ..pl import *


def generate_scRNA_report(adata, output_path="scRNA_analysis_report.html", 
                         species='human', sample_key=None):
    """
    Generate MultiQC-style HTML report for single-cell RNA-seq analysis
    
    Parameters:
    -----------
    adata : AnnData object
        The analyzed single-cell data object from lazy function
    output_path : str
        Path to save the HTML report
    species : str
        Species information for the analysis
    sample_key : str
        Key for batch/sample information
    """
    
    # Set scanpy settings for clean plots
    sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(8, 6))
    plt.style.use('default')

    if sample_key is None:
        sample_key = 'batch_none'
        adata.obs[sample_key] = 'sample1'
    
    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_str
    
    # Extract analysis status and parameters
    status = adata.uns.get('status', {})
    status_args = adata.uns.get('status_args', {})
    
    # Basic data information
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # Calculate additional statistics
    median_genes = np.median(adata.obs['n_genes_by_counts']) if 'n_genes_by_counts' in adata.obs else 0
    median_umis = np.median(adata.obs['total_counts']) if 'total_counts' in adata.obs else 0
    mito_genes = adata.var_names[adata.var_names.str.startswith(('MT-', 'mt-', 'Mt-'))]
    n_hvgs = sum(adata.var['highly_variable']) if 'highly_variable' in adata.var else 0
    
    # Generate visualizations with white background
    plots = {}
    
    # 1. QC metrics violin plot
    if all(col in adata.obs for col in ['detected_genes', 'nUMIs', 'mito_perc']):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        for i,key in enumerate(['detected_genes', 'nUMIs', 'mito_perc']):
            violin_box(
                adata,
                key,
                groupby=sample_key,
                ax=axes[i],
            )
        
        plt.suptitle('Quality Control Metrics Distribution', fontsize=16, y=1.02)
        plots['qc_violin'] = fig_to_base64(fig)
    
    # 2. PCA plot
    if 'X_pca' in adata.obsm:
        fig = plt.figure(figsize=(12, 5), facecolor='white')
        
        # PCA variance ratio
        plt.subplot(1, 2, 1)
        from ..utils._plot import plot_pca_variance_ratio1
        plot_pca_variance_ratio1(adata)
        plt.title('PCA Variance Explained', fontsize=12)
        
        # PCA scatter plot
        plt.subplot(1, 2, 2)
        if sample_key and sample_key in adata.obs:
            sc.pl.pca(adata, color=sample_key, show=False, frameon=False)
        else:
            sc.pl.pca(adata, show=False, frameon=False)
        plt.title('PCA Visualization', fontsize=12)
        
        plt.tight_layout()
        plots['pca_plot'] = fig_to_base64(fig)
    
    # 3. Batch correction comparison with UMAP
    if 'X_umap_harmony' in adata.obsm and 'X_umap_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
        
        # Original PCA
        sc.pl.pca(adata, color=sample_key, ax=axes[0,0], show=False, frameon=False, title='Original PCA')
        
        # Harmony UMAP
        embedding(adata, basis='X_umap_harmony', color=sample_key, ax=axes[0,1], 
                       show=False, frameon='small', title='Harmony UMAP')
        
        # scVI UMAP
        embedding(adata, basis='X_umap_scVI', color=sample_key, ax=axes[0,2], 
                       show=False, frameon='small', title='scVI UMAP')
        
        # Cell cycle visualization if available
        if 'phase' in adata.obs:
            sc.pl.pca(adata, color='phase', ax=axes[1,0], show=False, frameon=False, title='Cell Cycle - PCA')
            embedding(adata, basis='X_umap_harmony', color='phase', ax=axes[1,1], 
                           show=False, frameon='small', title='Cell Cycle - Harmony UMAP')
            embedding(adata, basis='X_umap_scVI', color='phase', ax=axes[1,2], 
                           show=False, frameon='small', title='Cell Cycle - scVI UMAP')
        
        plt.tight_layout()
        plots['batch_correction'] = fig_to_base64(fig)
    
    # Fallback: if only X_harmony and X_scVI are available (backward compatibility)
    elif 'X_harmony' in adata.obsm and 'X_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
        
        # Original PCA
        sc.pl.pca(adata, color=sample_key, ax=axes[0,0], show=False, frameon=False, title='Original PCA')
        
        # Harmony
        embedding(adata, basis='X_harmony', color=sample_key, ax=axes[0,1], 
                       show=False, frameon='small', title='Harmony Integration')
        
        # scVI
        embedding(adata, basis='X_scVI', color=sample_key, ax=axes[0,2], 
                       show=False, frameon='small', title='scVI Integration')
        
        # Cell cycle visualization if available
        if 'phase' in adata.obs:
            sc.pl.pca(adata, color='phase', ax=axes[1,0], show=False, frameon=False, title='Cell Cycle - PCA')
            embedding(adata, basis='X_harmony', color='phase', ax=axes[1,1], 
                           show=False, frameon='small', title='Cell Cycle - Harmony')
            embedding(adata, basis='X_scVI', color='phase', ax=axes[1,2], 
                           show=False, frameon='small', title='Cell Cycle - scVI')
        
        plt.tight_layout()
        plots['batch_correction'] = fig_to_base64(fig)
    
    # 4. Clustering visualization with multiple methods
    if 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
        # Create a larger figure for multiple clustering results
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor='white')
        
        # Best clusters (MDE)
        embedding(adata, basis='X_mde', color='best_clusters', ax=axes[0,0],
                       show=False, frameon='small', title='Best Clusters (SCCAF)')
        
        # Additional clustering methods if available
        if 'leiden_clusters_L1' in adata.obs:
            embedding(adata, basis='X_mde', color='leiden_clusters_L1', ax=axes[0,1],
                           show=False, frameon='small', title='Leiden Clusters (L1)')
        else:
            axes[0,1].text(0.5, 0.5, 'Leiden L1\nNot Available', ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_xticks([])
            axes[0,1].set_yticks([])
        
        if 'louvain_clusters_L1' in adata.obs:
            embedding(adata, basis='X_mde', color='louvain_clusters_L1', ax=axes[0,2],
                           show=False, frameon='small', title='Louvain Clusters (L1)')
        else:
            axes[0,2].text(0.5, 0.5, 'Louvain L1\nNot Available', ha='center', va='center', transform=axes[0,2].transAxes)
            axes[0,2].set_xticks([])
            axes[0,2].set_yticks([])
        
        # Sample distribution if available
        if sample_key and sample_key in adata.obs:
            embedding(adata, basis='X_mde', color=sample_key, ax=axes[1,0],
                           show=False, frameon='small', title='Sample Distribution')
        else:
            # Alternative: show total UMI counts
            if 'total_counts' in adata.obs:
                embedding(adata, basis='X_mde', color='total_counts', ax=axes[1,0],
                               show=False, frameon='small', title='Total UMI Counts')
            else:
                axes[1,0].text(0.5, 0.5, 'Sample Info\nNot Available', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_xticks([])
                axes[1,0].set_yticks([])
        
        # Additional clustering methods (L2)
        if 'leiden_clusters_L2' in adata.obs:
            embedding(adata, basis='X_mde', color='leiden_clusters_L2', ax=axes[1,1],
                           show=False, frameon='small', title='Leiden Clusters (L2)')
        else:
            axes[1,1].text(0.5, 0.5, 'Leiden L2\nNot Available', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])
        
        if 'louvain_clusters_L2' in adata.obs:
            embedding(adata, basis='X_mde', color='louvain_clusters_L2', ax=axes[1,2],
                           show=False, frameon='small', title='Louvain Clusters (L2)')
        else:
            axes[1,2].text(0.5, 0.5, 'Louvain L2\nNot Available', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_xticks([])
            axes[1,2].set_yticks([])
        
        plt.tight_layout()
        plots['clustering'] = fig_to_base64(fig)
    
    # Fallback for simpler clustering visualization
    elif 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
        
        # Clusters
        embedding(adata, basis='X_mde', color='best_clusters', ax=axes[0],
                       show=False, frameon='small', title='Final Clusters')
        
        # Sample distribution if available
        if sample_key and sample_key in adata.obs:
            embedding(adata, basis='X_mde', color=sample_key, ax=axes[1],
                           show=False, frameon='small', title='Sample Distribution')
        else:
            # Alternative: show total UMI counts
            if 'total_counts' in adata.obs:
                embedding(adata, basis='X_mde', color='total_counts', ax=axes[1],
                               show=False, frameon='small', title='Total UMI Counts')
        
        plt.tight_layout()
        plots['clustering'] = fig_to_base64(fig)
    
    # 5. Cell cycle distribution
    if 'phase' in adata.obs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        # Phase distribution
        phase_counts = adata.obs['phase'].value_counts()
        axes[0].pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%',
                   colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_title('Cell Cycle Phase Distribution', fontsize=14)
        
        # Phase by sample if available
        if sample_key and sample_key in adata.obs:
            phase_sample = pd.crosstab(adata.obs[sample_key], adata.obs['phase'], normalize='index') * 100
            phase_sample.plot(kind='bar', ax=axes[1], rot=45)
            axes[1].set_title('Cell Cycle Phase by Sample (%)', fontsize=14)
            axes[1].legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Alternative: show cell cycle scores
            if all(col in adata.obs for col in ['S_score', 'G2M_score']):
                axes[1].scatter(adata.obs['S_score'], adata.obs['G2M_score'], 
                              c=adata.obs['phase'].astype('category').cat.codes, 
                              alpha=0.6, s=10)
                axes[1].set_xlabel('S Score')
                axes[1].set_ylabel('G2M Score')
                axes[1].set_title('Cell Cycle Scores', fontsize=14)
        
        plt.tight_layout()
        plots['cell_cycle'] = fig_to_base64(fig)
    
    # 6. QC metrics correlation heatmap
    qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    if 'doublet_score' in adata.obs:
        qc_cols.append('doublet_score')
    if 'S_score' in adata.obs:
        qc_cols.extend(['S_score', 'G2M_score'])
    
    available_qc_cols = [col for col in qc_cols if col in adata.obs]
    if len(available_qc_cols) > 2:
        fig = plt.figure(figsize=(8, 6), facecolor='white')
        corr_matrix = adata.obs[available_qc_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('QC Metrics Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plots['qc_correlation'] = fig_to_base64(fig)
    
    # 7. Gene expression overview
    if 'highly_variable' in adata.var:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        # Mean vs variance of genes
        axes[0].scatter(adata.var['means'], adata.var['variances'], 
                       c=adata.var['highly_variable'], alpha=0.6, s=10)
        axes[0].set_xlabel('Mean Expression')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('Gene Expression: Mean vs Variance')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # Distribution of highly variable genes
        hvg_stats = adata.var.groupby('highly_variable').size()
        axes[1].bar(['Non-HVG', 'HVG'], hvg_stats.values, 
                   color=['#E0E0E0', '#4ECDC4'], alpha=0.8)
        axes[1].set_ylabel('Number of Genes')
        axes[1].set_title('Highly Variable Genes Selection')
        
        # Add percentage labels
        total_genes = sum(hvg_stats.values)
        for i, v in enumerate(hvg_stats.values):
            axes[1].text(i, v + total_genes*0.01, f'{v}\n({v/total_genes*100:.1f}%)', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plots['gene_expression'] = fig_to_base64(fig)
    
    # Generate MultiQC-style HTML content with logo and dark mode
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OmicVerse scRNA-seq Analysis Report</title>
        <link rel="icon" href="https://raw.githubusercontent.com/Starlitnightly/omicverse/master/README.assets/logo.png" type="image/png">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                line-height: 1.6;
                color: #333;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            
            /* Header */
            .header {{
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 1000;
            }}
            
            .header-content {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            
            .header-left {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            
            .logo {{
                width: 40px;
                height: 40px;
                border-radius: 8px;
                background: white;
                padding: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 300;
                color: white;
            }}
            
            .header-right {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            
            .theme-toggle {{
                background: rgba(255,255,255,0.2);
                border: 1px solid rgba(255,255,255,0.3);
                color: white;
                padding: 8px 12px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.9em;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            
            .theme-toggle:hover {{
                background: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }}
            
            .header-info {{
                font-size: 0.85em;
                opacity: 0.9;
                text-align: right;
            }}
            
            /* Layout */
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                background: white;
                min-height: calc(100vh - 70px);
                box-shadow: 0 0 20px rgba(0,0,0,0.05);
            }}
            
            /* Sidebar Navigation */
            .sidebar {{
                width: 250px;
                background: white;
                border-right: 1px solid #e0e0e0;
                padding: 20px 0;
                position: sticky;
                top: 70px;
                height: calc(100vh - 70px);
                overflow-y: auto;
            }}
            
            .nav-header {{
                padding: 0 20px 20px 20px;
                border-bottom: 1px solid #e0e0e0;
                margin-bottom: 20px;
            }}
            
            .nav-header h3 {{
                font-size: 0.9em;
                color: #6c757d;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .nav-menu {{
                list-style: none;
            }}
            
            .nav-menu li {{
                margin-bottom: 2px;
            }}
            
            .nav-menu a {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 12px 20px;
                color: #333;
                text-decoration: none;
                font-size: 0.9em;
                font-weight: 500;
                border-left: 3px solid transparent;
                transition: all 0.3s ease;
            }}
            
            .nav-menu a:hover {{
                background: #f8f9fa;
                border-left-color: #28a745;
                color: #28a745;
            }}
            
            .nav-menu a.active {{
                background: #28a745;
                color: white;
                border-left-color: #20c997;
            }}
            
            .nav-icon {{
                font-size: 1.1em;
                width: 20px;
                text-align: center;
            }}
            
            /* Main Content */
            .main-content {{
                flex: 1;
                padding: 30px;
                background: #f8f9fa;
                overflow-y: auto;
            }}
            
            /* Modules/Sections */
            .module {{
                background: white;
                border-radius: 8px;
                margin-bottom: 25px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            
            .module:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transform: translateY(-2px);
            }}
            
            .module-header {{
                background: #f8f9fa;
                padding: 20px 25px;
                border-bottom: 1px solid #e9ecef;
                border-radius: 8px 8px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .module-header h2 {{
                color: #28a745;
                font-size: 24px;
                font-weight: 600;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .module-icon {{
                font-size: 1.2em;
                color: #28a745;
            }}
            
            .badge {{
                background: #28a745;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.75em;
                margin-left: 10px;
                font-weight: normal;
            }}
            
            .badge.success {{ background: #28a745; }}
            .badge.warning {{ background: #ffc107; }}
            .badge.danger {{ background: #dc3545; }}
            .badge.info {{ background: #20c997; }}
            
            .module-content {{
                padding: 25px;
            }}
            
            /* Stats Grid */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .stat-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #28a745, #20c997);
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.1);
                border-color: #28a745;
            }}
            
            .stat-value {{
                font-size: 2em;
                font-weight: 300;
                color: #28a745;
                display: block;
                margin-bottom: 5px;
                line-height: 1;
            }}
            
            .stat-label {{
                color: #6c757d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            /* Tables */
            .table-container {{
                overflow-x: auto;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                font-size: 0.9em;
            }}
            
            .data-table th {{
                background: #28a745;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #20c997;
                position: sticky;
                top: 0;
            }}
            
            .data-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #e9ecef;
                color: #333;
            }}
            
            .data-table tr:hover {{
                background-color: #f8fff9;
            }}
            
            .data-table tr:nth-child(even) {{
                background-color: #fafafa;
            }}
            
            .data-table tr:nth-child(even):hover {{
                background-color: #f0f8f0;
            }}
            
            /* Status indicators */
            .status-pass {{
                background: #d4edda !important;
                color: #155724 !important;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
                display: inline-block;
            }}
            
            .status-warn {{
                background: #fff3cd !important;
                color: #856404 !important;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
                display: inline-block;
            }}
            
            .status-fail {{
                background: #f8d7da !important;
                color: #721c24 !important;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
                display: inline-block;
            }}
            
            /* Alert boxes */
            .alert {{
                padding: 15px;
                border-radius: 4px;
                margin: 20px 0;
                border-left: 4px solid;
            }}
            
            .alert-info {{
                background-color: #cce7f0;
                border-color: #20c997;
                color: #0c5460;
            }}
            
            .alert-success {{
                background-color: #d4edda;
                border-color: #28a745;
                color: #155724;
            }}
            
            .alert-warning {{
                background-color: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }}
            
            /* Plot containers */
            .plot-container {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 25px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
                text-align: center;
            }}
            
            .plot-container:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .plot-title {{
                font-size: 16px;
                font-weight: 600;
                color: #28a745;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e9f2ec;
            }}
            
            /* Parameters display */
            .parameters {{
                background: #f8fff9;
                border: 1px solid #c3e6cb;
                border-radius: 6px;
                padding: 15px;
                margin: 15px 0;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 0.9em;
                line-height: 1.6;
            }}
            
            .parameters-title {{
                font-weight: 600;
                color: #155724;
                margin-bottom: 8px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 1em;
            }}
            
            /* Metric highlights */
            .metric {{
                display: inline-block;
                background: #d4edda;
                color: #155724;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: 500;
                font-size: 0.9em;
            }}
            
            /* Progress bars */
            .progress {{
                height: 24px;
                background: #e9ecef;
                border-radius: 12px;
                overflow: hidden;
                margin: 15px 0;
                border: 1px solid #dee2e6;
            }}
            
            .progress-bar {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                border-radius: 12px;
                transition: width 0.8s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.8em;
                font-weight: 600;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }}
            
            /* Responsive design */
            @media (max-width: 1024px) {{
                .container {{
                    flex-direction: column;
                }}
                
                .sidebar {{
                    width: 100%;
                    height: auto;
                    position: static;
                }}
                
                .stats-grid {{
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                }}
            }}
            
            @media (max-width: 768px) {{
                .header-content {{
                    flex-direction: column;
                    gap: 10px;
                    text-align: center;
                }}
                
                .header-left {{
                    justify-content: center;
                }}
                
                .main-content {{
                    padding: 20px;
                }}
                
                .stats-grid {{
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                }}
            }}
            
            /* Smooth scrolling */
            html {{
                scroll-behavior: smooth;
            }}
            
            /* Loading animation */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .module {{
                animation: fadeIn 0.6s ease-out;
            }}
            
            /* Print styles */
            @media print {{
                .sidebar, .theme-toggle {{
                    display: none;
                }}
                .main-content {{
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                }}
            }}

            /* Dark mode support - simplified from original */
            [data-theme="dark"] {{
                background-color: #343a40;
                color: #f8f9fa;
            }}
            
            [data-theme="dark"] body {{
                background-color: #343a40;
                color: #f8f9fa;
            }}
            
            [data-theme="dark"] .header {{
                background: linear-gradient(135deg, #218838 0%, #20c997 100%);
            }}
            
            [data-theme="dark"] .container,
            [data-theme="dark"] .sidebar,
            [data-theme="dark"] .module,
            [data-theme="dark"] .stat-card,
            [data-theme="dark"] .data-table,
            [data-theme="dark"] .plot-container {{
                background: #343a40;
                border-color: #495057;
            }}
            
            [data-theme="dark"] .main-content {{
                background-color: #2c3136;
            }}
            
            [data-theme="dark"] .module-header {{
                background: #2c3136;
                border-color: #495057;
            }}
            
            [data-theme="dark"] .nav-menu a {{
                color: #f8f9fa;
            }}
            
            [data-theme="dark"] .nav-menu a:hover {{
                background: #2c3136;
                color: #28a745;
            }}
            
            [data-theme="dark"] .stat-value {{
                color: #20c997;
            }}
            
            [data-theme="dark"] .module-header h2,
            [data-theme="dark"] .plot-title {{
                color: #20c997;
            }}
            
            [data-theme="dark"] .alert-success {{
                background-color: rgba(40, 167, 69, 0.2);
                color: #a3cfbb;
            }}
            
            [data-theme="dark"] .alert-info {{
                background-color: rgba(32, 201, 151, 0.2);
                color: #a3d7e2;
            }}
            
            [data-theme="dark"] .alert-warning {{
                background-color: rgba(255, 193, 7, 0.2);
                color: #e0d3a3;
            }}
            
            [data-theme="dark"] .parameters {{
                background: rgba(40, 167, 69, 0.1);
                border-color: #218838;
                color: #d4edda;
            }}
            
            [data-theme="dark"] .parameters-title {{
                color: #a3cfbb;
            }}
            
            [data-theme="dark"] .data-table td {{
                color: #f8f9fa;
                border-color: #495057;
            }}
            
            [data-theme="dark"] .data-table tr:hover {{
                background-color: #2c3136;
            }}
            
            [data-theme="dark"] .data-table tr:nth-child(even) {{
                background-color: #3c4147;
            }}
            
            [data-theme="dark"] .data-table tr:nth-child(even):hover {{
                background-color: #2c3136;
            }}
            
            [data-theme="dark"] .stat-label {{
                color: #adb5bd;
            }}
            
            [data-theme="dark"] .progress {{
                background: #495057;
                border-color: #6c757d;
            }}
            
            [data-theme="dark"] .metric {{
                background: rgba(40, 167, 69, 0.2);
                color: #a3cfbb;
            }}
        </style>
        <script>
            // Theme management
            const themeToggle = document.createElement('button');
            const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const savedTheme = localStorage.getItem('theme');
            
            // Set initial theme
            const initialTheme = savedTheme || (prefersDarkMode ? 'dark' : 'light');
            document.documentElement.setAttribute('data-theme', initialTheme);
            
            document.addEventListener('DOMContentLoaded', function() {{
                // Theme toggle functionality
                const themeToggleBtn = document.querySelector('.theme-toggle');
                if (themeToggleBtn) {{
                    updateThemeToggleText();
                    
                    themeToggleBtn.addEventListener('click', function() {{
                        const currentTheme = document.documentElement.getAttribute('data-theme');
                        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                        
                        document.documentElement.setAttribute('data-theme', newTheme);
                        localStorage.setItem('theme', newTheme);
                        updateThemeToggleText();
                    }});
                }}
                
                // Navigation functionality
                const navLinks = document.querySelectorAll('.nav-menu a');
                navLinks.forEach(link => {{
                    link.addEventListener('click', function(e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        const targetElement = document.getElementById(targetId);
                        if (targetElement) {{
                            targetElement.scrollIntoView({{ 
                                behavior: 'smooth',
                                block: 'start',
                                inline: 'nearest'
                            }});
                            
                            // Update active link
                            navLinks.forEach(l => l.classList.remove('active'));
                            this.classList.add('active');
                        }}
                    }});
                }});
                
                // Highlight current section on scroll
                const observer = new IntersectionObserver(entries => {{
                    entries.forEach(entry => {{
                        if (entry.isIntersecting) {{
                            const sectionId = entry.target.id;
                            navLinks.forEach(l => l.classList.remove('active'));
                            const activeLink = document.querySelector(`a[href="#${{sectionId}}"]`);
                            if (activeLink) activeLink.classList.add('active');
                        }}
                    }});
                }}, {{
                    threshold: 0.3,
                    rootMargin: '-10% 0px -10% 0px'
                }});
                
                // Observe all sections
                document.querySelectorAll('.module').forEach(section => {{
                    observer.observe(section);
                }});
                
                // Add loading animation delay
                document.querySelectorAll('.module').forEach((module, index) => {{
                    module.style.animationDelay = `${{index * 0.1}}s`;
                }});
            }});
            
            function updateThemeToggleText() {{
                const themeToggleBtn = document.querySelector('.theme-toggle');
                const currentTheme = document.documentElement.getAttribute('data-theme');
                if (themeToggleBtn) {{
                    themeToggleBtn.innerHTML = currentTheme === 'dark' 
                        ? '<span>☀️</span> Light Mode' 
                        : '<span>🌙</span> Dark Mode';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div class="header-left">
                    <img src="https://raw.githubusercontent.com/Starlitnightly/omicverse/master/README.assets/logo.png" 
                         alt="OmicVerse Logo" class="logo" onerror="this.style.display='none'">
                    <h1>OmicVerse scRNA-seq Analysis Report</h1>
                </div>
                <div class="header-right">
                    <button class="theme-toggle">🌙 Dark Mode</button>
                    <div class="header-info">
                        <div><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                        <div><strong>Species:</strong> {species.title()} • <strong>Pipeline:</strong> OmicVerse</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="container">
            <nav class="sidebar">
                <div class="nav-header">
                    <h3>Analysis Sections</h3>
                </div>
                <ul class="nav-menu">
                    <li><a href="#overview" class="active"><span class="nav-icon">📊</span>General Statistics</a></li>
                    <li><a href="#quality-control"><span class="nav-icon">🔍</span>Quality Control</a></li>
                    <li><a href="#gene-expression"><span class="nav-icon">🧬</span>Gene Expression</a></li>
                    <li><a href="#pca-analysis"><span class="nav-icon">📈</span>PCA Analysis</a></li>
                    <li><a href="#batch-correction"><span class="nav-icon">🔄</span>Batch Correction</a></li>
                    <li><a href="#clustering"><span class="nav-icon">🎯</span>Clustering</a></li>
                    <li><a href="#cell-cycle"><span class="nav-icon">⏰</span>Cell Cycle</a></li>
                    <li><a href="#pipeline-status"><span class="nav-icon">⚙️</span>Pipeline Status</a></li>
                    <li><a href="#integration-benchmark"><span class="nav-icon">🏆</span>Benchmarking</a></li>
                </ul>
            </nav>
            
            <main class="main-content">
                <!-- General Statistics Module -->
                <div id="overview" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">📊</span>General Statistics</h2>
                        <span class="badge">Overview</span>
                    </div>
                    <div class="module-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <span class="stat-value">{n_cells:,}</span>
                                <div class="stat-label">Total Cells</div>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value">{n_genes:,}</span>
                                <div class="stat-label">Total Genes</div>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value">{n_hvgs:,}</span>
                                <div class="stat-label">Highly Variable Genes</div>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value">{int(median_genes):,}</span>
                                <div class="stat-label">Median Genes/Cell</div>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value">{int(median_umis):,}</span>
                                <div class="stat-label">Median UMIs/Cell</div>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value">{len([k for k, v in status.items() if v])}/{len(status)}</span>
                                <div class="stat-label">Analysis Steps</div>
                            </div>
                        </div>
                        
                        <div class="alert alert-info">
                            <strong>📋 Dataset Summary:</strong> This single-cell RNA-seq dataset contains 
                            <span class="metric">{n_cells:,} cells</span> and <span class="metric">{n_genes:,} genes</span>. 
                            After quality control and feature selection, <span class="metric">{n_hvgs:,} highly variable genes</span> 
                            ({n_hvgs/n_genes*100:.1f}% of total) were identified for downstream analysis.
                        </div>
                        
                        <div class="progress">
                            <div class="progress-bar" style="width: {len([k for k, v in status.items() if v])/len(status)*100:.0f}%">
                                {len([k for k, v in status.items() if v])}/{len(status)} Steps Completed ({len([k for k, v in status.items() if v])/len(status)*100:.0f}%)
                            </div>
                        </div>
                    </div>
                </div>
    """
    
    # Quality Control Module
    if 'qc_violin' in plots:
        html_content += f"""
                <!-- Quality Control Module -->
                <div id="quality-control" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🔍</span>Quality Control</h2>
                        <span class="badge info">QC Metrics</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">Distribution of Quality Control Metrics</div>
                            <img src="data:image/png;base64,{plots['qc_violin']}" alt="QC Violin Plot">
                        </div>
                        
                        <div class="parameters">
                            <div class="parameters-title">🔧 QC Filtering Parameters</div>
                            • <strong>Mitochondrial gene threshold:</strong> {status_args.get('qc', {}).get('mito_perc', status_args.get('qc', {}).get('tresh', {}).get('mito_perc', 'N/A'))}%<br>
                            • <strong>Minimum UMI counts:</strong> {status_args.get('qc', {}).get('nUMIs', status_args.get('qc', {}).get('tresh', {}).get('nUMIs', 'N/A'))}<br>
                            • <strong>Minimum detected genes:</strong> {status_args.get('qc', {}).get('detected_genes', status_args.get('qc', {}).get('tresh', {}).get('detected_genes', 'N/A'))}<br>
                            • <strong>Doublet detection method:</strong> {status_args.get('qc', {}).get('doublets_method', 'N/A')}<br>
                            • <strong>Batch key:</strong> {status_args.get('qc', {}).get('batch_key', 'N/A')}
                        </div>
        """
        
        if 'qc_correlation' in plots:
            html_content += f"""
                        <div class="plot-container">
                            <div class="plot-title">QC Metrics Correlation Matrix</div>
                            <img src="data:image/png;base64,{plots['qc_correlation']}" alt="QC Correlation">
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
        """
    
    # Gene Expression Module
    if 'gene_expression' in plots:
        html_content += f"""
                <!-- Gene Expression Module -->
                <div id="gene-expression" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🧬</span>Gene Expression Analysis</h2>
                        <span class="badge success">Feature Selection</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">Gene Expression Overview and HVG Selection</div>
                            <img src="data:image/png;base64,{plots['gene_expression']}" alt="Gene Expression">
                        </div>
                        
                        <div class="alert alert-success">
                            <strong>✅ Feature Selection Results:</strong> {n_hvgs:,} highly variable genes were selected 
                            from {n_genes:,} total genes ({n_hvgs/n_genes*100:.1f}%). These features will be used 
                            for dimensionality reduction and downstream analyses.
                        </div>
                    </div>
                </div>
        """
    
    # PCA Analysis Module
    if 'pca_plot' in plots:
        html_content += f"""
                <!-- PCA Analysis Module -->
                <div id="pca-analysis" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">📈</span>Principal Component Analysis</h2>
                        <span class="badge info">Dimensionality Reduction</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">PCA Results and Variance Explained</div>
                            <img src="data:image/png;base64,{plots['pca_plot']}" alt="PCA Analysis">
                        </div>
                        
                        <div class="parameters">
                            <div class="parameters-title">🔧 PCA Parameters</div>
                            • <strong>Number of components:</strong> {status_args.get('pca', {}).get('n_pcs', 'N/A')}<br>
                            • <strong>Data layer:</strong> {status_args.get('pca', {}).get('layer', 'X (default)')}<br>
                            • <strong>Use highly variable genes:</strong> {status_args.get('pca', {}).get('use_highly_variable', 'True')}
                        </div>
                    </div>
                </div>
        """
    
    # Batch Correction Module
    if 'batch_correction' in plots:
        html_content += f"""
                <!-- Batch Correction Module -->
                <div id="batch-correction" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🔄</span>Batch Effect Correction</h2>
                        <span class="badge">Integration</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">Batch Correction Comparison: Before and After Integration</div>
                            <img src="data:image/png;base64,{plots['batch_correction']}" alt="Batch Correction">
                        </div>
                        
                        <div class="alert alert-info">
                            <strong>🔄 Integration Methods Applied:</strong> Multiple batch correction methods were evaluated. 
                            <span class="metric">{adata.uns.get('bench_best_res', 'Unknown')}</span> was selected as the 
                            optimal integration method based on benchmarking metrics.
                        </div>
                        
                        <div class="parameters">
                            <div class="parameters-title">🔧 Integration Parameters</div>
                            • <strong>Harmony PCs:</strong> {status_args.get('harmony', {}).get('n_pcs', 'N/A')}<br>
                            • <strong>scVI latent dimensions:</strong> {status_args.get('scVI', {}).get('n_latent', 'N/A')}<br>
                            • <strong>scVI layers:</strong> {status_args.get('scVI', {}).get('n_layers', 'N/A')}<br>
                            • <strong>Best method:</strong> {adata.uns.get('bench_best_res', 'Not determined')}
                        </div>
                    </div>
                </div>
        """
    
    # Clustering Module
    if 'clustering' in plots:
        n_clusters = len(adata.obs['best_clusters'].unique()) if 'best_clusters' in adata.obs else 0
        html_content += f"""
                <!-- Clustering Module -->
                <div id="clustering" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🎯</span>Cell Clustering</h2>
                        <span class="badge success">{n_clusters} Clusters</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">Final Clustering Results</div>
                            <img src="data:image/png;base64,{plots['clustering']}" alt="Clustering Results">
                        </div>
                        
                        <div class="alert alert-success">
                            <strong>🎯 Clustering Summary:</strong> Automated clustering identified 
                            <span class="metric">{n_clusters} distinct cell clusters</span> using the SCCAF algorithm 
                            with Leiden clustering. Results are visualized using MDE (Minimum Distortion Embedding).
                        </div>
                    </div>
                </div>
        """
    
    # Cell Cycle Module
    if 'cell_cycle' in plots:
        html_content += f"""
                <!-- Cell Cycle Module -->
                <div id="cell-cycle" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">⏰</span>Cell Cycle Analysis</h2>
                        <span class="badge info">Phase Distribution</span>
                    </div>
                    <div class="module-content">
                        <div class="plot-container">
                            <div class="plot-title">Cell Cycle Phase Distribution and Scores</div>
                            <img src="data:image/png;base64,{plots['cell_cycle']}" alt="Cell Cycle Analysis">
                        </div>
        """
        
        # Add cell cycle statistics table
        if 'phase' in adata.obs:
            phase_counts = adata.obs['phase'].value_counts()
            html_content += """
                        <div class="table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>Cell Cycle Phase</th>
                                        <th>Cell Count</th>
                                        <th>Percentage</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
            """
            
            for phase, count in phase_counts.items():
                percentage = (count / n_cells) * 100
                if percentage > 20:
                    status_class = "status-pass"
                    status_text = "✅ Normal"
                elif percentage > 10:
                    status_class = "status-warn"
                    status_text = "⚠️ Low"
                else:
                    status_class = "status-fail"
                    status_text = "❌ Very Low"
                    
                html_content += f"""
                                    <tr>
                                        <td><strong>{phase}</strong></td>
                                        <td>{count:,}</td>
                                        <td>{percentage:.1f}%</td>
                                        <td><span class="{status_class}">{status_text}</span></td>
                                    </tr>
                """
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            """
    
    # Integration Benchmark Module
    if 'bench_res' in adata.uns:
        html_content += f"""
                <!-- Integration Benchmark Module -->
                <div id="integration-benchmark" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🏆</span>Integration Method Benchmark</h2>
                        <span class="badge success">Performance Metrics</span>
                    </div>
                    <div class="module-content">
                        <div class="alert alert-success">
                            <strong>🏆 Winner:</strong> <span class="metric">{adata.uns['bench_best_res']}</span> 
                            was selected as the best performing integration method based on comprehensive benchmarking.
                        </div>
                        
                        <div class="table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>Integration Method</th>
                                        <th>Batch Correction Score</th>
                                        <th>Bio Conservation Score</th>
                                        <th>Overall Score</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
        """
        
        bench_res = adata.uns['bench_res']
        best_method = adata.uns['bench_best_res']
        
        for method in bench_res.index:
            batch_score = f"{bench_res.loc[method, 'Batch correction']:.3f}" if 'Batch correction' in bench_res.columns else 'N/A'
            bio_score = f"{bench_res.loc[method, 'Bio conservation']:.3f}" if 'Bio conservation' in bench_res.columns else 'N/A'
            overall_score = f"{bench_res.loc[method, 'Overall']:.3f}" if 'Overall' in bench_res.columns else 'N/A'
            
            if method == best_method:
                status_class = "status-pass"
                status_text = "🏆 Best"
            else:
                status_class = "status-warn"
                status_text = "⚠️ Alternative"
            
            html_content += f"""
                                    <tr>
                                        <td><strong>{method}</strong></td>
                                        <td>{batch_score}</td>
                                        <td>{bio_score}</td>
                                        <td>{overall_score}</td>
                                        <td><span class="{status_class}">{status_text}</span></td>
                                    </tr>
            """
        
        html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
        """
    else:
        # Show a message when benchmarking results are not available
        best_method = adata.uns.get('bench_best_res', 'X_scVI')  # Default fallback
        html_content += f"""
                <!-- Integration Benchmark Module -->
                <div id="integration-benchmark" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">🏆</span>Integration Method Benchmark</h2>
                        <span class="badge warning">Auto-Selected</span>
                    </div>
                    <div class="module-content">
                        <div class="alert alert-info">
                            <strong>🏆 Best Method:</strong> <span class="metric">{best_method}</span> 
                            was automatically selected as the integration method. Detailed benchmarking metrics are not available.
                        </div>
                        
                        <div class="parameters">
                            <div class="parameters-title">🔧 Available Integration Methods</div>
                            • <strong>Harmony:</strong> {'✅ Available' if 'X_umap_harmony' in adata.obsm or 'X_harmony' in adata.obsm else '❌ Not available'}<br>
                            • <strong>scVI:</strong> {'✅ Available' if 'X_umap_scVI' in adata.obsm or 'X_scVI' in adata.obsm else '❌ Not available'}<br>
                            • <strong>Selected:</strong> {best_method}
                        </div>
                    </div>
                </div>
        """
    
    # Pipeline Status Module
    html_content += f"""
                <!-- Pipeline Status Module -->
                <div id="pipeline-status" class="module">
                    <div class="module-header">
                        <h2><span class="module-icon">⚙️</span>Analysis Pipeline Status</h2>
                        <span class="badge">Workflow</span>
                    </div>
                    <div class="module-content">
                        <div class="table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>Analysis Step</th>
                                        <th>Status</th>
                                        <th>Parameters</th>
                                    </tr>
                                </thead>
                                <tbody>
    """
    
    # Pipeline steps with their status (updated for new lazy function)
    steps = [
        ('qc', 'Quality Control & Filtering', '🔍'),
        ('preprocess', 'Preprocessing & Normalization', '⚙️'),
        ('scaled', 'Data Scaling', '📏'),
        ('pca', 'Principal Component Analysis', '📈'),
        ('cell_cycle', 'Cell Cycle Scoring', '🔄'),
        ('harmony', 'Harmony Integration', '🎵'),  # Updated to lowercase
        ('scVI', 'scVI Integration', '🧬'),
        ('eval_bench', 'Method Benchmarking', '📊'),
        ('eval_clusters', 'SCCAF Clustering Analysis', '🎯')
    ]
    
    # Check for backward compatibility with old key names
    if 'Harmony' in status and 'harmony' not in status:
        status['harmony'] = status['Harmony']
        if 'status_args' in adata.uns and 'Harmony' in adata.uns['status_args']:
            adata.uns['status_args']['harmony'] = adata.uns['status_args']['Harmony']
    
    for step_key, step_name, emoji in steps:
        step_status = status.get(step_key, False)
        if step_status:
            status_class = "status-pass"
            status_text = "✅ Completed"
        else:
            status_class = "status-fail"
            status_text = "❌ Not Completed"
        
        # Get parameters for this step
        step_params = status_args.get(step_key, {})
        params_text = ""
        if step_params:
            params_list = [f"{k}: {v}" for k, v in list(step_params.items())[:3]]  # Show first 3 params
            params_text = "; ".join(params_list)
            if len(step_params) > 3:
                params_text += f" (+ {len(step_params)-3} more)"
        else:
            params_text = "Default parameters"
        
        html_content += f"""
                                    <tr>
                                        <td><strong>{emoji} {step_name}</strong></td>
                                        <td><span class="{status_class}">{status_text}</span></td>
                                        <td style="font-family: monospace; font-size: 0.8em;">{params_text}</td>
                                    </tr>
        """
    
    # Close HTML
    html_content += f"""
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="alert alert-info">
                            <strong>📋 Pipeline Summary:</strong> This analysis was completed using the OmicVerse lazy function pipeline. 
                            The pipeline automatically performed quality control, normalization, batch correction, clustering, and benchmarking 
                            to provide comprehensive single-cell RNA-seq analysis results.
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ MultiQC-style report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
    print(f"📊 Visualizations included: {len(plots)} plots")
    print(f"🎨 Features: White theme, Dark mode toggle, OmicVerse logo")
    print(f"🎯 Clustering methods detected: {len([col for col in ['best_clusters', 'leiden_clusters_L1', 'louvain_clusters_L1', 'leiden_clusters_L2', 'louvain_clusters_L2'] if col in adata.obs])}")
    print(f"🧬 Integration methods: {'Harmony ✓' if 'X_umap_harmony' in adata.obsm or 'X_harmony' in adata.obsm else 'Harmony ✗'}, {'scVI ✓' if 'X_umap_scVI' in adata.obsm or 'X_scVI' in adata.obsm else 'scVI ✗'}")
    print(f"🏆 Best integration method: {adata.uns.get('bench_best_res', 'Not determined')}")
    return html_content

# Usage example for OmicVerse:
# html_report = generate_scRNA_report(adata, species='human', sample_key='sample_id')
# This creates a professional MultiQC-style HTML report with:
# - Clean white theme with dark mode toggle
# - OmicVerse logo integration
# - Responsive design and smooth animations
# - Enhanced navigation and modern UI components
# - Support for all OmicVerse lazy() function features