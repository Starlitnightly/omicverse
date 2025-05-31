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

    # 定义日间/夜间模式颜色方案 - 现代简约风格
    color_schemes = {
        "day": {
            "primary": "#1f6912",       # DeepSeek 蓝色
            "primary-light": "#e8f0fe", # 浅蓝背景
            "secondary": "#6c757d",     # 次要色
            "background": "#f8f9fa",    # 背景色
            "card": "#ffffff",          # 卡片背景
            "text": "#212529",          # 正文文字
            "text-light": "#5c6670",    # 浅色文字
            "border": "#dee2e6",        # 边框色
            "success": "#198754",       # 成功色
            "warning": "#ffc107",       # 警告色
            "danger": "#dc3545",        # 危险色
            "plot_bg": "white"          # 绘图背景
        },
        "night": {
            "primary": "#1c9519",       # 夜间主色调
            "primary-light": "#1e293b", # 深蓝背景
            "secondary": "#94a3b8",     # 次要色
            "background": "#0f172a",    # 背景色
            "card": "#1e293b",          # 卡片背景
            "text": "#e2e8f0",          # 正文文字
            "text-light": "#94a3b8",    # 浅色文字
            "border": "#334155",        # 边框色
            "success": "#22c55e",       # 成功色
            "warning": "#f59e0b",       # 警告色
            "danger": "#ef4444",        # 危险色
            "plot_bg": "#1e293b"        # 绘图背景
        }
    }

    # 生成两种模式的图像
    plots = {"day": {}, "night": {}}
    
    # 修改图像生成函数支持夜间模式
    def fig_to_base64(fig, mode="day"):
        """Convert matplotlib figure to base64 string with theme support"""
        facecolor = color_schemes[mode]["plot_bg"]
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor=facecolor, edgecolor='none')
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
    
    # 生成两种模式的图像
    plots = {"day": {}, "night": {}}
    
    for mode in ["day", "night"]:
        # 设置绘图主题
        if mode == "night":
            
            sc.settings.set_figure_params(dpi=100)
            plt.style.use('dark_background')
        else:
            
            sc.settings.set_figure_params(dpi=100)
            plt.style.use('default')
        
        # 1. QC metrics violin plot
        if all(col in adata.obs for col in ['detected_genes', 'nUMIs', 'mito_perc']):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), 
                                    facecolor=color_schemes[mode]["plot_bg"])
            for i,key in enumerate(['detected_genes', 'nUMIs', 'mito_perc']):
                violin_box(
                    adata,
                    key,
                    groupby=sample_key,
                    ax=axes[i],
                )
            
            plt.suptitle('Quality Control Metrics Distribution', fontsize=16, y=1.02)
            plots[mode]['qc_violin'] = fig_to_base64(fig, mode)
        
        # 2. PCA plot
        if 'X_pca' in adata.obsm:
            fig = plt.figure(figsize=(12, 5), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['pca_plot'] = fig_to_base64(fig, mode)
        
        # 3. Batch correction comparison with UMAP
        if 'X_umap_harmony' in adata.obsm and 'X_umap_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['batch_correction'] = fig_to_base64(fig, mode)
        
        # Fallback: if only X_harmony and X_scVI are available (backward compatibility)
        elif 'X_harmony' in adata.obsm and 'X_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['batch_correction'] = fig_to_base64(fig, mode)
        
        # 4. Clustering visualization with multiple methods
        if 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
            # Create a larger figure for multiple clustering results
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['clustering'] = fig_to_base64(fig, mode)
        
        # Fallback for simpler clustering visualization
        elif 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['clustering'] = fig_to_base64(fig, mode)
        
        # 5. Cell cycle distribution
        if 'phase' in adata.obs:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['cell_cycle'] = fig_to_base64(fig, mode)
        
        # 6. QC metrics correlation heatmap
        qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        if 'doublet_score' in adata.obs:
            qc_cols.append('doublet_score')
        if 'S_score' in adata.obs:
            qc_cols.extend(['S_score', 'G2M_score'])
        
        available_qc_cols = [col for col in qc_cols if col in adata.obs]
        if len(available_qc_cols) > 2:
            fig = plt.figure(figsize=(8, 6), facecolor=color_schemes[mode]["plot_bg"])
            corr_matrix = adata.obs[available_qc_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            plt.title('QC Metrics Correlation Matrix', fontsize=14)
            plt.tight_layout()
            plots[mode]['qc_correlation'] = fig_to_base64(fig, mode)
        
        # 7. Gene expression overview
        if 'highly_variable' in adata.var:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=color_schemes[mode]["plot_bg"])
            
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
            plots[mode]['gene_expression'] = fig_to_base64(fig, mode)
    
    # Generate MultiQC-style HTML content with logo and dark mode
    # 提取分析状态和参数
    status = adata.uns.get('status', {})
    status_args = adata.uns.get('status_args', {})
    
    # 基本数据信息
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # 计算额外统计信息
    median_genes = np.median(adata.obs['n_genes_by_counts']) if 'n_genes_by_counts' in adata.obs else 0
    median_umis = np.median(adata.obs['total_counts']) if 'total_counts' in adata.obs else 0
    n_hvgs = sum(adata.var['highly_variable']) if 'highly_variable' in adata.var else 0
    n_clusters = len(adata.obs['best_clusters'].unique()) if 'best_clusters' in adata.obs else 0

    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en" class="day-mode">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>scRNA-seq Analysis Report | DeepSeek Style</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                /* Day mode variables */
                --primary: {color_schemes['day']['primary']};
                --primary-light: {color_schemes['day']['primary-light']};
                --secondary: {color_schemes['day']['secondary']};
                --background: {color_schemes['day']['background']};
                --card: {color_schemes['day']['card']};
                --text: {color_schemes['day']['text']};
                --text-light: {color_schemes['day']['text-light']};
                --border: {color_schemes['day']['border']};
                --success: {color_schemes['day']['success']};
                --warning: {color_schemes['day']['warning']};
                --danger: {color_schemes['day']['danger']};
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                --transition: all 0.3s ease;
            }}
            
            .night-mode {{
                /* Night mode variables */
                --primary: {color_schemes['night']['primary']};
                --primary-light: {color_schemes['night']['primary-light']};
                --secondary: {color_schemes['night']['secondary']};
                --background: {color_schemes['night']['background']};
                --card: {color_schemes['night']['card']};
                --text: {color_schemes['night']['text']};
                --text-light: {color_schemes['night']['text-light']};
                --border: {color_schemes['night']['border']};
                --success: {color_schemes['night']['success']};
                --warning: {color_schemes['night']['warning']};
                --danger: {color_schemes['night']['danger']};
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: var(--background);
                color: var(--text);
                line-height: 1.6;
                font-size: 14px;
                transition: var(--transition);
                -webkit-font-smoothing: antialiased;
                display: flex;
                min-height: 100vh;
            }}
            
            /* Sidebar */
            .sidebar {{
                width: 240px;
                background: var(--card);
                border-right: 1px solid var(--border);
                padding: 20px 0;
                height: 100vh;
                position: sticky;
                top: 0;
                overflow-y: auto;
                transition: var(--transition);
                box-shadow: var(--shadow);
                z-index: 100;
            }}
            
            .logo-container {{
                padding: 0 20px 20px;
                border-bottom: 1px solid var(--border);
                margin-bottom: 20px;
            }}
            
            .logo {{
                font-size: 20px;
                font-weight: 700;
                color: var(--primary);
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .logo-icon {{
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                overflow: hidden;
                background: var(--primary-light);
            }}
            
            .logo-icon img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }}
            
            .nav-section {{
                margin-bottom: 25px;
            }}
            
            .nav-title {{
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-light);
                padding: 0 20px 10px;
                margin-bottom: 10px;
                border-bottom: 1px solid var(--border);
            }}
            
            .nav-menu {{
                list-style: none;
            }}
            
            .nav-item {{
                margin-bottom: 2px;
            }}
            
            .nav-link {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                color: var(--text);
                text-decoration: none;
                font-weight: 500;
                border-left: 3px solid transparent;
                transition: var(--transition);
            }}
            
            .nav-link:hover, .nav-link.active {{
                background: var(--primary-light);
                border-left-color: var(--primary);
                color: var(--primary);
            }}
            
            .nav-icon {{
                width: 20px;
                text-align: center;
            }}
            
            /* Main content */
            .main-content {{
                flex: 1;
                padding: 30px;
                overflow-y: auto;
            }}
            
            .header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 25px;
            }}
            
            .page-title {{
                font-size: 22px;
                font-weight: 600;
                color: var(--text);
            }}
            
            .header-actions {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            
            .theme-toggle {{
                background: var(--card);
                border: 1px solid var(--border);
                color: var(--text);
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 13px;
                font-weight: 500;
                transition: var(--transition);
            }}
            
            .theme-toggle:hover {{
                border-color: var(--primary);
                color: var(--primary);
            }}
            
            /* Dashboard */
            .dashboard {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            /* Cards */
            .card {{
                background: var(--card);
                border-radius: 10px;
                padding: 20px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
                transition: var(--transition);
            }}
            
            .card:hover {{
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
            }}
            
            .card-header {{
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            
            .card-title {{
                font-size: 16px;
                font-weight: 600;
                color: var(--text);
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .card-icon {{
                color: var(--primary);
            }}
            
            .card-badge {{
                background: var(--primary-light);
                color: var(--primary);
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            
            /* Stats */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .stat-item {{
                text-align: center;
                padding: 15px;
                background: var(--primary-light);
                border-radius: 8px;
            }}
            
            .stat-value {{
                font-size: 22px;
                font-weight: 700;
                color: var(--primary);
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                font-size: 13px;
                color: var(--text-light);
            }}
            
            /* Plots */
            .plot-container {{
                text-align: center;
                margin-top: 15px;
            }}
            
            .plot-img {{
                max-width: 100%;
                max-height: 400px;
                border-radius: 8px;
                border: 1px solid var(--border);
            }}
            
            /* Info boxes */
            .info-box {{
                background: var(--primary-light);
                border-left: 3px solid var(--primary);
                padding: 15px;
                border-radius: 0 6px 6px 0;
                margin: 20px 0;
                font-size: 14px;
            }}
            
            /* Progress bar */
            .progress-container {{
                margin: 20px 0;
            }}
            
            .progress-bar {{
                height: 8px;
                background: rgba(var(--primary-rgb), 0.1);
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 8px;
            }}
            
            .progress-fill {{
                height: 100%;
                background: var(--primary);
                border-radius: 4px;
                transition: width 0.8s ease;
            }}
            
            .progress-text {{
                font-size: 13px;
                color: var(--text-light);
                display: flex;
                justify-content: space-between;
            }}
            
            /* Tables */
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
                margin: 15px 0;
            }}
            
            .data-table th {{
                background: var(--primary-light);
                color: var(--text);
                text-align: left;
                padding: 10px 15px;
                font-weight: 600;
                border-bottom: 1px solid var(--border);
            }}
            
            .data-table td {{
                padding: 10px 15px;
                border-bottom: 1px solid var(--border);
                color: var(--text);
            }}
            
            .data-table tr:last-child td {{
                border-bottom: none;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            
            .status-success {{
                background: rgba(var(--success-rgb), 0.1);
                color: var(--success);
            }}
            
            .status-warning {{
                background: rgba(var(--warning-rgb), 0.1);
                color: var(--warning);
            }}
            
            .status-danger {{
                background: rgba(var(--danger-rgb), 0.1);
                color: var(--danger);
            }}
            
            /* Footer */
            .footer {{
                text-align: center;
                padding: 20px 0;
                color: var(--text-light);
                font-size: 13px;
                margin-top: 40px;
                border-top: 1px solid var(--border);
            }}
            
            /* Responsive design */
            @media (max-width: 1024px) {{
                .sidebar {{
                    width: 200px;
                }}
            }}
            
            @media (max-width: 768px) {{
                body {{
                    flex-direction: column;
                }}
                
                .sidebar {{
                    width: 100%;
                    height: auto;
                    position: static;
                }}
                
                .main-content {{
                    padding: 20px;
                }}
                
                .dashboard {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            /* Animation */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .fade-in {{
                animation: fadeIn 0.4s ease-out forwards;
            }}
        </style>
        <script>
            // Convert hex to RGB
            function hexToRgb(hex) {{
                var result = /^#?([a-f\d]{{2}})([a-f\d]{{2}})([a-f\d]{{2}})$/i.exec(hex);
                return result ? {{
                    r: parseInt(result[1], 16),
                    g: parseInt(result[2], 16),
                    b: parseInt(result[3], 16)
                }} : null;
            }}
            
            // Set CSS variables for RGB colors
            function setColorVariables() {{
                const root = document.documentElement;
                const primaryRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--primary').trim());
                root.style.setProperty('--primary-rgb', `${{primaryRgb.r}}, ${{primaryRgb.g}}, ${{primaryRgb.b}}`);
                
                const successRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--success').trim());
                root.style.setProperty('--success-rgb', `${{successRgb.r}}, ${{successRgb.g}}, ${{successRgb.b}}`);
                
                const warningRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--warning').trim());
                root.style.setProperty('--warning-rgb', `${{warningRgb.r}}, ${{warningRgb.g}}, ${{warningRgb.b}}`);
            }}
            
            // Theme management
            document.addEventListener('DOMContentLoaded', function() {{
                setColorVariables();
                
                // Theme toggle functionality
                const themeToggleBtn = document.querySelector('.theme-toggle');
                if (themeToggleBtn) {{
                    // Set initial theme based on system preference
                    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    const savedTheme = localStorage.getItem('theme');
                    const initialTheme = savedTheme || (prefersDarkMode ? 'night' : 'day');
                    
                    if (initialTheme === 'night') {{
                        document.documentElement.classList.add('night-mode');
                        themeToggleBtn.innerHTML = '<i>☀️</i> Light Mode';
                    }} else {{
                        themeToggleBtn.innerHTML = '<i>🌙</i> Dark Mode';
                    }}
                    
                    themeToggleBtn.addEventListener('click', function() {{
                        const isNightMode = document.documentElement.classList.toggle('night-mode');
                        localStorage.setItem('theme', isNightMode ? 'night' : 'day');
                        themeToggleBtn.innerHTML = isNightMode ? '<i>☀️</i> Light Mode' : '<i>🌙</i> Dark Mode';
                        
                        // Switch plot images
                        document.querySelectorAll('.plot-img.day-mode').forEach(img => {{
                            img.style.display = isNightMode ? 'none' : 'block';
                        }});
                        document.querySelectorAll('.plot-img.night-mode').forEach(img => {{
                            img.style.display = isNightMode ? 'block' : 'none';
                        }});
                    }});
                }}
                
                // Navigation functionality
                const navLinks = document.querySelectorAll('.nav-link');
                navLinks.forEach(link => {{
                    link.addEventListener('click', function(e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        const targetElement = document.getElementById(targetId);
                        if (targetElement) {{
                            targetElement.scrollIntoView({{ 
                                behavior: 'smooth',
                                block: 'start'
                            }});
                            
                            // Update active link
                            navLinks.forEach(l => l.classList.remove('active'));
                            this.classList.add('active');
                        }}
                    }});
                }});
                
                // Apply fade-in animations
                document.querySelectorAll('.card').forEach((el, index) => {{
                    el.style.animationDelay = `${{index * 0.1}}s`;
                    el.classList.add('fade-in');
                }});
            }});
        </script>
    </head>
    <body>
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="logo-container">
                <div class="logo">
                    <div class="logo-icon">
                        <img src="https://omicverse.readthedocs.io/en/latest/img/logo.png" 
                            alt="OmicVerse Logo" class="logo">
                    </div>
                    <span>OmicVerse Report</span>
                </div>
            </div>
            
            <div class="nav-section">
                <div class="nav-title">Analysis Sections</div>
                <ul class="nav-menu">
                    <li class="nav-item"><a href="#overview" class="nav-link active"><span class="nav-icon">📊</span>Overview</a></li>
                    <li class="nav-item"><a href="#quality" class="nav-link"><span class="nav-icon">🔍</span>Quality Control</a></li>
                    <li class="nav-item"><a href="#expression" class="nav-link"><span class="nav-icon">🧬</span>Gene Expression</a></li>
                    <li class="nav-item"><a href="#pca" class="nav-link"><span class="nav-icon">📈</span>Dimensionality</a></li>
                    <li class="nav-item"><a href="#batch" class="nav-link"><span class="nav-icon">🔄</span>Batch Correction</a></li>
                    <li class="nav-item"><a href="#clustering" class="nav-link"><span class="nav-icon">🎯</span>Clustering</a></li>
                    <li class="nav-item"><a href="#cellcycle" class="nav-link"><span class="nav-icon">⏰</span>Cell Cycle</a></li>
                    <li class="nav-item"><a href="#benchmark" class="nav-link"><span class="nav-icon">🏆</span>Benchmark</a></li>
                    <li class="nav-item"><a href="#pipeline" class="nav-link"><span class="nav-icon">⚙️</span>Pipeline</a></li>
                </ul>
            </div>
            
            <div class="nav-section">
                <div class="nav-title">Report Info</div>
                <ul class="nav-menu">
                    <li class="nav-item"><div class="nav-link"><span class="nav-icon">📅</span>Date: {datetime.now().strftime('%Y-%m-%d')}</div></li>
                    <li class="nav-item"><div class="nav-link"><span class="nav-icon">🧬</span>Species: {species.title()}</div></li>
                    <li class="nav-item"><div class="nav-link"><span class="nav-icon">🔬</span>Cells: {n_cells:,}</div></li>
                    <li class="nav-item"><div class="nav-link"><span class="nav-icon">🧪</span>Genes: {n_genes:,}</div></li>
                </ul>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="header">
                <h1 class="page-title">scRNA-seq Analysis Report</h1>
                <div class="header-actions">
                    <button class="theme-toggle">
                        <i>🌙</i> Dark Mode
                    </button>
                </div>
            </div>
            
            <!-- Overview Section -->
            <div id="overview" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">📊</i> Dataset Overview</h2>
                    <span class="card-badge">Summary</span>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{n_cells:,}</div>
                        <div class="stat-label">Total Cells</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-value">{n_genes:,}</div>
                        <div class="stat-label">Total Genes</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-value">{n_hvgs:,}</div>
                        <div class="stat-label">HVGs</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-value">{int(median_genes):,}</div>
                        <div class="stat-label">Med. Genes/Cell</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-value">{int(median_umis):,}</div>
                        <div class="stat-label">Med. UMIs/Cell</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-value">{n_clusters}</div>
                        <div class="stat-label">Clusters</div>
                    </div>
                </div>
                
                <div class="info-box">
                    This single-cell RNA-seq dataset contains <strong>{n_cells:,} cells</strong> and 
                    <strong>{n_genes:,} genes</strong>. After quality control, <strong>{n_hvgs:,} highly variable genes</strong> 
                    ({n_hvgs/n_genes*100:.1f}% of total) were used for downstream analysis.
                </div>
                
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {len([k for k, v in status.items() if v])/len(status)*100:.0f}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>Analysis Progress</span>
                        <span>{len([k for k, v in status.items() if v])}/{len(status)} Steps</span>
                    </div>
                </div>
            </div>
    """
    
    # Quality Control Section
    if 'qc_violin' in plots['day']:
        html_content += f"""
            <!-- Quality Control Section -->
            <div id="quality" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">🔍</i> Quality Control</h2>
                    <span class="card-badge">Metrics</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['qc_violin']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['qc_violin']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
                
                <div class="mt-3">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mitochondrial threshold</td>
                                <td>{status_args.get('qc', {}).get('mito_perc', status_args.get('qc', {}).get('tresh', {}).get('mito_perc', 'N/A'))}%</td>
                            </tr>
                            <tr>
                                <td>Minimum UMIs</td>
                                <td>{status_args.get('qc', {}).get('nUMIs', status_args.get('qc', {}).get('tresh', {}).get('nUMIs', 'N/A'))}</td>
                            </tr>
                            <tr>
                                <td>Minimum genes</td>
                                <td>{status_args.get('qc', {}).get('detected_genes', status_args.get('qc', {}).get('tresh', {}).get('detected_genes', 'N/A'))}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        """
    
    # Gene Expression Section
    if 'gene_expression' in plots['day']:
        html_content += f"""
            <!-- Gene Expression Section -->
            <div id="expression" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">🧬</i> Gene Expression</h2>
                    <span class="card-badge">HVGs</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['gene_expression']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['gene_expression']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
                
                <div class="info-box">
                    <strong>Highly Variable Genes:</strong> {n_hvgs:,} genes selected 
                    ({n_hvgs/n_genes*100:.1f}% of total) for downstream analysis.
                </div>
            </div>
        """
    
    # PCA Section
    if 'pca_plot' in plots['day']:
        html_content += f"""
            <!-- PCA Section -->
            <div id="pca" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">📈</i> Dimensionality Reduction</h2>
                    <span class="card-badge">PCA</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['pca_plot']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['pca_plot']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
                
                <div class="mt-3">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Number of components</td>
                                <td>{status_args.get('pca', {}).get('n_pcs', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Data layer</td>
                                <td>{status_args.get('pca', {}).get('layer', 'X (default)')}</td>
                            </tr>
                            <tr>
                                <td>Use HVGs</td>
                                <td>{status_args.get('pca', {}).get('use_highly_variable', 'True')}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        """
    
    # Batch Correction Section
    if 'batch_correction' in plots['day']:
        html_content += f"""
            <!-- Batch Correction Section -->
            <div id="batch" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">🔄</i> Batch Correction</h2>
                    <span class="card-badge">Integration</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['batch_correction']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['batch_correction']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
                
                <div class="info-box">
                    <strong>Best Method:</strong> <span class="status-badge status-success">{adata.uns.get('bench_best_res', 'Unknown')}</span> 
                    selected as the optimal integration method.
                </div>
            </div>
        """
    
    # Clustering Section
    if 'clustering' in plots['day']:
        html_content += f"""
            <!-- Clustering Section -->
            <div id="clustering" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">🎯</i> Cell Clustering</h2>
                    <span class="card-badge">{n_clusters} Clusters</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['clustering']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['clustering']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
                
                <div class="info-box">
                    Identified <strong>{n_clusters} distinct cell clusters</strong> using SCCAF algorithm.
                </div>
            </div>
        """
    
    # Cell Cycle Section
    if 'cell_cycle' in plots['day']:
        html_content += f"""
            <!-- Cell Cycle Section -->
            <div id="cellcycle" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">⏰</i> Cell Cycle Analysis</h2>
                    <span class="card-badge">Distribution</span>
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['day']['cell_cycle']}" 
                         class="plot-img day-mode">
                    <img src="data:image/png;base64,{plots['night']['cell_cycle']}" 
                         class="plot-img night-mode" style="display:none;">
                </div>
            </div>
        """
    
    # Benchmark Section
    if 'bench_res' in adata.uns:
        html_content += f"""
            <!-- Benchmark Section -->
            <div id="benchmark" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">🏆</i> Integration Benchmark</h2>
                    <span class="card-badge">Comparison</span>
                </div>
                
                <div class="mt-3">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Batch Correction</th>
                                <th>Bio Conservation</th>
                                <th>Overall</th>
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
                status_class = "status-badge status-success"
                status_text = "Best"
            else:
                status_class = "status-badge"
                status_text = "Alternative"
            
            html_content += f"""
                            <tr>
                                <td>{method}</td>
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
        """
    
    # Pipeline Section
    html_content += f"""
            <!-- Pipeline Section -->
            <div id="pipeline" class="card fade-in">
                <div class="card-header">
                    <h2 class="card-title"><i class="card-icon">⚙️</i> Analysis Pipeline</h2>
                    <span class="card-badge">Status</span>
                </div>
                
                <div class="mt-3">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>Status</th>
                                <th>Parameters</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Pipeline steps
    steps = [
        ('qc', 'Quality Control', '🔍'),
        ('preprocess', 'Preprocessing', '⚙️'),
        ('scaled', 'Data Scaling', '📏'),
        ('pca', 'PCA', '📈'),
        ('cell_cycle', 'Cell Cycle', '🔄'),
        ('harmony', 'Harmony', '🎵'),
        ('scVI', 'scVI', '🧬'),
        ('eval_bench', 'Benchmarking', '📊'),
        ('eval_clusters', 'Clustering', '🎯')
    ]
    
    # Check for backward compatibility
    if 'Harmony' in status and 'harmony' not in status:
        status['harmony'] = status['Harmony']
    
    for step_key, step_name, emoji in steps:
        step_status = status.get(step_key, False)
        if step_status:
            status_class = "status-badge status-success"
            status_text = "Completed"
        else:
            status_class = "status-badge status-danger"
            status_text = "Not Completed"
        
        # Get parameters for this step
        step_params = status_args.get(step_key, {})
        params_text = ""
        if step_params:
            params_list = [f"{k}: {v}" for k, v in list(step_params.items())[:2]]
            params_text = "; ".join(params_list)
            if len(step_params) > 2:
                params_text += f" (+{len(step_params)-2})"
        else:
            params_text = "Default"
        
        html_content += f"""
                            <tr>
                                <td>{emoji} {step_name}</td>
                                <td><span class="{status_class}">{status_text}</span></td>
                                <td>{params_text}</td>
                            </tr>
        """
    
    # Close HTML
    html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by OmicVerse with DeepSeek-style UI • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ DeepSeek-style report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
    #print(f"📊 Visualizations: {len(plots['day']} plots")
    print(f"🎨 Features: Left sidebar, Professional layout, Moderate image sizes")
    return html_content

# Usage example for OmicVerse:
# html_report = generate_scRNA_report(adata, species='human', sample_key='sample_id')
# This creates a professional MultiQC-style HTML report with:
# - Clean white theme with dark mode toggle
# - OmicVerse logo integration
# - Responsive design and smooth animations
# - Enhanced navigation and modern UI components
# - Support for all OmicVerse lazy() function features