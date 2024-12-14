```python
# Line 1:  Import the scanpy library for single-cell analysis -- import scanpy as sc
# Line 2:  Import the omicverse library, likely for multi-omics analysis and visualization -- import omicverse as ov
# Line 3:  Set plotting style using omicverse's plot_set function -- ov.plot_set()
# Line 4:  -- 
# Line 5:  Import the scvelo library for RNA velocity analysis -- import scvelo as scv
# Line 6:  Load the dentategyrus dataset from scvelo -- adata=scv.datasets.dentategyrus()
# Line 7:  -- 
# Line 8:  Time the preprocessing step using ipython's magic command -- %%time
# Line 9:  Preprocess the adata object using omicverse with shiftlog normalization and pearson scaling, keeping 2000 highly variable genes -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 10: Display the preprocessed adata object -- adata
# Line 11:  -- 
# Line 12: Scale the data using omicverse -- ov.pp.scale(adata)
# Line 13: Perform principal component analysis using omicverse -- ov.pp.pca(adata)
# Line 14:  -- 
# Line 15: Import the matplotlib.pyplot library for plotting -- import matplotlib.pyplot as plt
# Line 16: Import patheffects from matplotlib for adding outline to objects -- from matplotlib import patheffects
# Line 17: Create a matplotlib figure and axes object with a specified size -- fig, ax = plt.subplots(figsize=(4,4))
# Line 18: Generate an embedding plot using omicverse, with specified color, frame, title, legend, and other aesthetic settings -- ov.pl.embedding(
# Line 19:  --     adata,
# Line 20:  --     basis="X_umap",
# Line 21:  --     color=['clusters'],
# Line 22:  --     frameon='small',
# Line 23:  --     title="Celltypes",
# Line 24:  --     #legend_loc='on data',
# Line 25:  --     legend_fontsize=14,
# Line 26:  --     legend_fontoutline=2,
# Line 27:  --     #size=10,
# Line 28:  --     ax=ax,
# Line 29:  --     #legend_loc=True, 
# Line 30:  --     add_outline=False, 
# Line 31:  --     #add_outline=True,
# Line 32:  --     outline_color='black',
# Line 33:  --     outline_width=1,
# Line 34:  --     show=False,
# Line 35:  -- )
# Line 36:  -- 
# Line 37: Import the numpy library for numerical operations -- import numpy as np
# Line 38: Initialize a cNMF object with specified components, iterations, seed, high variance genes, output directory and name -- cnmf_obj = ov.single.cNMF(adata,components=np.arange(5,11), n_iter=20, seed=14, num_highvar_genes=2000,
# Line 39:  --                           output_dir='example_dg/cNMF', name='dg_cNMF')
# Line 40:  -- 
# Line 41: Run the factorization for the cNMF object with a specified worker and number of workers -- cnmf_obj.factorize(worker_i=0, total_workers=2)
# Line 42:  -- 
# Line 43: Combine the results of the cNMF factorization, skipping missing files -- cnmf_obj.combine(skip_missing_files=True)
# Line 44:  -- 
# Line 45: Create a plot for k selection of the cNMF object -- cnmf_obj.k_selection_plot(close_fig=False)
# Line 46:  -- 
# Line 47: Set the selected K value for consensus clustering -- selected_K = 7
# Line 48: Set the density threshold for filtering -- density_threshold = 2.00
# Line 49:  -- 
# Line 50: Run consensus clustering with the specified k and density threshold, showing the clustering results -- cnmf_obj.consensus(k=selected_K, 
# Line 51:  --                    density_threshold=density_threshold, 
# Line 52:  --                    show_clustering=True, 
# Line 53:  --                    close_clustergram_fig=False)
# Line 54:  -- 
# Line 55: Set the density threshold for filtering -- density_threshold = 0.10
# Line 56:  -- 
# Line 57: Run consensus clustering with the specified k and density threshold, showing the clustering results -- cnmf_obj.consensus(k=selected_K, 
# Line 58:  --                    density_threshold=density_threshold, 
# Line 59:  --                    show_clustering=True, 
# Line 60:  --                    close_clustergram_fig=False)
# Line 61:  -- 
# Line 62: Import the seaborn library for statistical data visualization -- import seaborn as sns
# Line 63: Import the matplotlib.pyplot library for plotting -- import matplotlib.pyplot as plt
# Line 64: Import patheffects from matplotlib for adding outline to objects -- from matplotlib import patheffects
# Line 65:  -- 
# Line 66: Import gridspec for creating more complex figure layouts -- from matplotlib import gridspec
# Line 67: Import matplotlib.pyplot library for plotting -- import matplotlib.pyplot as plt
# Line 68:  -- 
# Line 69: Define the width ratios for the subplots -- width_ratios = [0.2, 4, 0.5, 10, 1]
# Line 70: Define the height ratios for the subplots -- height_ratios = [0.2, 4]
# Line 71: Create a matplotlib figure object with specified size based on ratios -- fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
# Line 72: Create a gridspec object for defining the structure of the subplots -- gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
# Line 73:  --                        0.01, 0.01, 0.98, 0.98,
# Line 74:  --                        height_ratios=height_ratios,
# Line 75:  --                        width_ratios=width_ratios,
# Line 76:  --                        wspace=0, hspace=0)
# Line 77:  --             
# Line 78: Extract topic distances from the cNMF object -- D = cnmf_obj.topic_dist[cnmf_obj.spectra_order, :][:, cnmf_obj.spectra_order]
# Line 79: Add a subplot for the distance matrix with no x or y ticks, labels or frame -- dist_ax = fig.add_subplot(gs[1,1], xscale='linear', yscale='linear',
# Line 80:  --                                     xticks=[], yticks=[],xlabel='', ylabel='',
# Line 81:  --                                     frameon=True)
# Line 82: Display the topic distance matrix using imshow -- dist_im = dist_ax.imshow(D, interpolation='none', cmap='viridis',
# Line 83:  --                          aspect='auto', rasterized=True)
# Line 84:  -- 
# Line 85: Add a subplot for the left cluster labels with no x or y ticks, labels or frame -- left_ax = fig.add_subplot(gs[1,0], xscale='linear', yscale='linear', xticks=[], yticks=[],
# Line 86:  --                 xlabel='', ylabel='', frameon=True)
# Line 87: Display the cluster labels on the left using imshow -- left_ax.imshow(cnmf_obj.kmeans_cluster_labels.values[cnmf_obj.spectra_order].reshape(-1, 1),
# Line 88:  --                             interpolation='none', cmap='Spectral', aspect='auto',
# Line 89:  --                             rasterized=True)
# Line 90:  -- 
# Line 91: Add a subplot for the top cluster labels with no x or y ticks, labels or frame -- top_ax = fig.add_subplot(gs[0,1], xscale='linear', yscale='linear', xticks=[], yticks=[],
# Line 92:  --                 xlabel='', ylabel='', frameon=True)
# Line 93: Display the cluster labels on the top using imshow -- top_ax.imshow(cnmf_obj.kmeans_cluster_labels.values[cnmf_obj.spectra_order].reshape(1, -1),
# Line 94:  --                   interpolation='none', cmap='Spectral', aspect='auto',
# Line 95:  --                     rasterized=True)
# Line 96:  -- 
# Line 97: Create a nested gridspec for the colorbar with no spacing -- cbar_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1, 2],
# Line 98:  --                                    wspace=0, hspace=0)
# Line 99: Add a subplot for the colorbar with no x or y labels, with a title -- cbar_ax = fig.add_subplot(cbar_gs[1,2], xscale='linear', yscale='linear',
# Line 100:  --     xlabel='', ylabel='', frameon=True, title='Euclidean\nDistance')
# Line 101: Set the title of the colorbar subplot -- cbar_ax.set_title('Euclidean\nDistance',fontsize=12)
# Line 102: Find the minimum value in the distance matrix for color scaling -- vmin = D.min().min()
# Line 103: Find the maximum value in the distance matrix for color scaling -- vmax = D.max().max()
# Line 104: Add a colorbar to the figure based on the dist_im object, with specified ticks and formatting -- fig.colorbar(dist_im, cax=cbar_ax,
# Line 105:  --         ticks=np.linspace(vmin, vmax, 3),
# Line 106:  --         )
# Line 107: Set the font size of the y-axis tick labels -- cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(),fontsize=12)
# Line 108:  -- 
# Line 109: Filter local density values based on the specified threshold -- density_filter = cnmf_obj.local_density.iloc[:, 0] < density_threshold
# Line 110: Create a matplotlib figure and axes for histogram -- fig, hist_ax = plt.subplots(figsize=(4,4))
# Line 111:  -- 
# Line 112: Create a histogram of the local density values, with specified bins -- hist_ax.hist(cnmf_obj.local_density.values, bins=np.linspace(0, 1, 50))
# Line 113: Put y-axis ticks on the right side of the plot -- hist_ax.yaxis.tick_right()
# Line 114:  -- 
# Line 115: Get the current x limits of the hist_ax plot -- xlim = hist_ax.get_xlim()
# Line 116: Get the current y limits of the hist_ax plot -- ylim = hist_ax.get_ylim()
# Line 117: If the density threshold is within the x-axis limits, add a vertical line and a text indicating the threshold -- if density_threshold < xlim[1]:
# Line 118:  --     hist_ax.axvline(density_threshold, linestyle='--', color='k')
# Line 119:  --     hist_ax.text(density_threshold  + 0.02, ylim[1] * 0.95, 'filtering\nthreshold\n\n', va='top')
# Line 120: Set the x-axis limits of the hist_ax plot -- hist_ax.set_xlim(xlim)
# Line 121: Set the x-axis label with information about filtering based on density threshold -- hist_ax.set_xlabel('Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering'%(sum(~density_filter), len(density_filter), 100*(~density_filter).mean()))
# Line 122: Set the title of the histogram -- hist_ax.set_title('Local density histogram')
# Line 123:  -- 
# Line 124: Load the cNMF results with the specified K and density threshold -- result_dict = cnmf_obj.load_results(K=selected_K, density_threshold=density_threshold)
# Line 125:  -- 
# Line 126:  -- 
# Line 127: Display the head of the 'usage_norm' dataframe from the results -- result_dict['usage_norm'].head()
# Line 128:  -- 
# Line 129: Display the head of the 'gep_scores' dataframe from the results -- result_dict['gep_scores'].head()
# Line 130:  -- 
# Line 131: Display the head of the 'gep_tpm' dataframe from the results -- result_dict['gep_tpm'].head()
# Line 132:  -- 
# Line 133: Display the head of the 'top_genes' dataframe from the results -- result_dict['top_genes'].head()
# Line 134:  -- 
# Line 135: Add the results of cNMF to the anndata object -- cnmf_obj.get_results(adata,result_dict)
# Line 136:  -- 
# Line 137: Generate an embedding plot of the usage_norm values from cNMF using omicverse -- ov.pl.embedding(adata, basis='X_umap',color=result_dict['usage_norm'].columns,
# Line 138:  --            use_raw=False, ncols=3, vmin=0, vmax=1,frameon='small')
# Line 139:  -- 
# Line 140: Generate an embedding plot using omicverse with specified color, frame, title, legend, and other aesthetic settings -- ov.pl.embedding(
# Line 141:  --     adata,
# Line 142:  --     basis="X_umap",
# Line 143:  --     color=['cNMF_cluster'],
# Line 144:  --     frameon='small',
# Line 145:  --     #title="Celltypes",
# Line 146:  --     #legend_loc='on data',
# Line 147:  --     legend_fontsize=14,
# Line 148:  --     legend_fontoutline=2,
# Line 149:  --     #size=10,
# Line 150:  --     #legend_loc=True, 
# Line 151:  --     add_outline=False, 
# Line 152:  --     #add_outline=True,
# Line 153:  --     outline_color='black',
# Line 154:  --     outline_width=1,
# Line 155:  --     show=False,
# Line 156:  -- )
# Line 157:  -- 
# Line 158: Add the random forest classifier results to the anndata object -- cnmf_obj.get_results_rfc(adata,result_dict,
# Line 159:  --                          use_rep='scaled|original|X_pca',
# Line 160:  --                         cNMF_threshold=0.5)
# Line 161:  -- 
# Line 162: Generate an embedding plot using omicverse with specified color, frame, title, legend, and other aesthetic settings -- ov.pl.embedding(
# Line 163:  --     adata,
# Line 164:  --     basis="X_umap",
# Line 165:  --     color=['cNMF_cluster_rfc','cNMF_cluster_clf'],
# Line 166:  --     frameon='small',
# Line 167:  --     #title="Celltypes",
# Line 168:  --     #legend_loc='on data',
# Line 169:  --     legend_fontsize=14,
# Line 170:  --     legend_fontoutline=2,
# Line 171:  --     #size=10,
# Line 172:  --     #legend_loc=True, 
# Line 173:  --     add_outline=False, 
# Line 174:  --     #add_outline=True,
# Line 175:  --     outline_color='black',
# Line 176:  --     outline_width=1,
# Line 177:  --     show=False,
# Line 178:  -- )
# Line 179:  -- 
# Line 180: Initialize empty list to hold top genes -- plot_genes=[]
# Line 181: Loop through columns of top genes dataframe and add the top 3 genes from each column to plot_genes list -- for i in result_dict['top_genes'].columns:
# Line 182:  --     plot_genes+=result_dict['top_genes'][i][:3].values.reshape(-1).tolist()
# Line 183:  -- 
# Line 184: Create a dotplot of top genes for each cNMF cluster using scanpy -- sc.pl.dotplot(adata,plot_genes,
# Line 185:  --               "cNMF_cluster", dendrogram=False,standard_scale='var',)
```