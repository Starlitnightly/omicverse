```python
# Line 1:  -- import omicverse as ov
# Line 1: Imports the omicverse library and assigns it the alias 'ov'.
# Line 2:  -- print(f'omicverse version:{ov.__version__}')
# Line 2: Prints the version of the omicverse library.
# Line 3:  -- import scanpy as sc
# Line 3: Imports the scanpy library and assigns it the alias 'sc'.
# Line 4:  -- print(f'scanpy version:{sc.__version__}')
# Line 4: Prints the version of the scanpy library.
# Line 5:  -- ov.ov_plot_set()
# Line 5: Sets the plotting style for omicverse.
# Line 10:  -- adata = sc.read_10x_mtx(
# Line 10: Reads 10x Genomics data into an AnnData object.
# Line 11:  --     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
# Line 11: Specifies the directory containing the matrix files.
# Line 12:  --     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
# Line 12: Sets gene symbols as variable names.
# Line 13:  --     cache=True)                              # write a cache file for faster subsequent reading
# Line 13: Enables caching for faster reading in subsequent executions.
# Line 17:  -- adata=ov.pp.qc(adata,
# Line 17: Applies quality control filtering on the AnnData object.
# Line 18:  --               tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
# Line 18: Sets thresholds for mitochondrial percentage, number of UMIs, and detected genes for QC.
# Line 20:  -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 20: Preprocesses the AnnData object, including normalization and HVG selection.
# Line 23:  -- adata.raw = adata
# Line 23: Stores a copy of the raw data in the `.raw` attribute of the AnnData object.
# Line 24:  -- adata = adata[:, adata.var.highly_variable_features]
# Line 24: Filters the AnnData object to keep only highly variable genes.
# Line 27:  -- ov.pp.scale(adata)
# Line 27: Scales the gene expression data in the AnnData object.
# Line 30:  -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 30: Performs Principal Component Analysis (PCA) on the scaled data.
# Line 33:  -- sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 33: Constructs a neighborhood graph for the AnnData object.
# Line 34:  --                use_rep='scaled|original|X_pca')
# Line 34: Specifies the representations used for neighbor graph construction.
# Line 37:  -- sc.tl.leiden(adata)
# Line 37: Performs Leiden clustering on the AnnData object.
# Line 40:  -- sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca')
# Line 40: Computes a dendrogram based on the Leiden clusters.
# Line 41:  -- sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca',
# Line 41: Ranks genes based on differential expression between leiden clusters.
# Line 42:  --                         method='wilcoxon',use_raw=False,)
# Line 42: Uses the Wilcoxon test for gene ranking and does not use the raw data.
# Line 45:  -- adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
# Line 45: Computes and stores a Manifold Diffusion Embedding (MDE) in the obsm attribute.
# Line 46:  -- adata
# Line 46: Displays the AnnData object.
# Line 48:  -- ov.pl.embedding(adata,
# Line 48: Creates an embedding plot.
# Line 49:  --                    basis='X_mde',
# Line 49: Specifies the embedding basis for plotting.
# Line 50:  --                    color=['leiden'], 
# Line 50: Sets the colors based on the 'leiden' clustering.
# Line 51:  --                    legend_loc='on data', 
# Line 51: Places the legend on the plot.
# Line 52:  --                    frameon='small',
# Line 52: Sets a smaller frame size for plot.
# Line 53:  --                    legend_fontoutline=2,
# Line 53: Sets the outline width of legend text.
# Line 54:  --                    palette=ov.utils.palette()[14:],
# Line 54: Sets the color palette for plotting.
# Line 56:  -- import os
# Line 56: Imports the os library for environment variable manipulation.
# Line 57:  -- all_markers={'cluster1':['CD3D','CD3E'],
# Line 57: Defines a dictionary of marker genes for specific clusters.
# Line 58:  --             'cluster2':['MS4A1']}
# Line 58: Defines additional marker genes for a specific cluster.
# Line 60:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 60: Sets an environment variable for API key.
# Line 61:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 61: Uses a function to predict cell types using a large language model based on the markers.
# Line 62:  --                       model='qwen-plus', provider='qwen',
# Line 62: Specifies the large language model and provider.
# Line 63:  --                       topgenenumber=5)
# Line 63: Specifies the top number of genes to consider.
# Line 64:  -- result
# Line 64: Displays the result.
# Line 66:  -- all_markers=ov.single.get_celltype_marker(adata,clustertype='leiden',rank=True,
# Line 66: Gets marker genes based on the Leiden clusters.
# Line 67:  --                                           key='rank_genes_groups',
# Line 67: Specifies the key for retrieving the ranked genes.
# Line 68:  --                                           foldchange=2,topgenenumber=5)
# Line 68: Sets the fold change threshold and number of top genes.
# Line 69:  -- all_markers
# Line 69: Displays all markers.
# Line 71:  -- import os
# Line 71: Imports the os library for environment variable manipulation.
# Line 72:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 72: Sets an environment variable for API key.
# Line 73:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 73: Predicts cell types based on the identified markers using a large language model.
# Line 74:  --                       model='qwen-plus', provider='qwen',
# Line 74: Specifies the large language model and provider.
# Line 75:  --                       topgenenumber=5)
# Line 75: Specifies the top number of genes to consider.
# Line 76:  -- result
# Line 76: Displays the result.
# Line 78:  -- new_result={}
# Line 78: Initializes an empty dictionary.
# Line 79:  -- for key in result.keys():
# Line 79: Iterates through the keys in the result dictionary.
# Line 80:  --     new_result[key]=result[key].split(': ')[-1].split(' (')[0].split('. ')[1]
# Line 80: Processes the result strings to extract the cell type name.
# Line 81:  -- new_result
# Line 81: Displays the processed results.
# Line 83:  -- adata.obs['gpt_celltype'] = adata.obs['leiden'].map(new_result).astype('category')
# Line 83: Maps the cell types from new_result to the obs attribute of the AnnData object.
# Line 85:  -- ov.pl.embedding(adata,
# Line 85: Creates an embedding plot with cell type annotation.
# Line 86:  --                    basis='X_mde',
# Line 86: Specifies the embedding basis for plotting.
# Line 87:  --                    color=['leiden','gpt_celltype'], 
# Line 87: Specifies the colors for the embedding plot.
# Line 88:  --                    legend_loc='on data', 
# Line 88: Sets the location of legend on the data points.
# Line 89:  --                    frameon='small',
# Line 89: Sets the frame to small size.
# Line 90:  --                    legend_fontoutline=2,
# Line 90: Sets the font outline width for legend text.
# Line 91:  --                    palette=ov.utils.palette()[14:],
# Line 91: Sets the color palette for the plot.
# Line 93:  -- all_markers={'cluster1':['CD3D','CD3E'],
# Line 93: Defines a dictionary of marker genes for specific clusters.
# Line 94:  --             'cluster2':['MS4A1']}
# Line 94: Defines additional marker genes for a specific cluster.
# Line 96:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 96: Sets an environment variable for API key.
# Line 97:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 97: Predicts cell types using a large language model, specifically gpt-4o.
# Line 98:  --                       model='gpt-4o', provider='openai',
# Line 98: Specifies the large language model and provider.
# Line 99:  --                       topgenenumber=5)
# Line 99: Specifies the top number of genes to consider.
# Line 100:  -- result
# Line 100: Displays the result.
# Line 102:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 102: Sets an environment variable for API key.
# Line 103:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 103: Predicts cell types using a large language model, specifically qwen-plus.
# Line 104:  --                       model='qwen-plus', provider='qwen',
# Line 104: Specifies the large language model and provider.
# Line 105:  --                       topgenenumber=5)
# Line 105: Specifies the top number of genes to consider.
# Line 106:  -- result
# Line 106: Displays the result.
# Line 108:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 108: Sets an environment variable for API key.
# Line 109:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 109: Predicts cell types using a large language model, specifically moonshot-v1-8k.
# Line 110:  --                       model='moonshot-v1-8k', provider='kimi',
# Line 110: Specifies the large language model and provider.
# Line 111:  --                       topgenenumber=5)
# Line 111: Specifies the top number of genes to consider.
# Line 112:  -- result
# Line 112: Displays the result.
# Line 114:  -- os.environ['AGI_API_KEY'] = 'sk-**'  # Replace with your actual API key
# Line 114: Sets an environment variable for API key.
# Line 115:  -- result = ov.single.gptcelltype(all_markers, tissuename='PBMC', speciename='human',
# Line 115: Predicts cell types using a large language model, specifically moonshot-v1-8k, with a base URL.
# Line 116:  --                       model='moonshot-v1-8k', base_url="https://api.moonshot.cn/v1",
# Line 116: Specifies the large language model, provider, and base URL.
# Line 117:  --                       topgenenumber=5)
# Line 117: Specifies the top number of genes to consider.
# Line 118:  -- result
# Line 118: Displays the result.
# Line 120:  -- anno_model = 'path/to/your/local/LLM'  # '~/models/Qwen2-7B-Instruct'
# Line 120: Defines a variable for a local LLM model path.
# Line 122:  -- result = ov.single.gptcelltype_local(all_markers, tissuename='PBMC', speciename='human', 
# Line 122: Predicts cell types using a locally hosted large language model.
# Line 123:  --                      model_name=anno_model, topgenenumber=5)
# Line 123: Specifies the local model and top number of genes.
# Line 124:  -- result
# Line 124: Displays the result.
```