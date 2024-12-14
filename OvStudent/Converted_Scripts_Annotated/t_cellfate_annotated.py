```
# Line 1:  import omicverse as ov -- import omicverse as ov
# Line 3:  import scanpy as sc -- import scanpy as sc
# Line 5:  import pandas as pd -- import pandas as pd
# Line 6:  from tqdm.auto import tqdm -- from tqdm.auto import tqdm
# Line 7:  ov.plot_set() -- ov.plot_set()
# Line 9:  adata = ov.single.mouse_hsc_nestorowa16() -- adata = ov.single.mouse_hsc_nestorowa16()
# Line 10:  adata -- adata
# Line 12:  prior_network = ov.single.load_human_prior_interaction_network(dataset='nichenet')  -- prior_network = ov.single.load_human_prior_interaction_network(dataset='nichenet')
# Line 15:  prior_network = ov.single.convert_human_to_mouse_network(prior_network,server_name='asia')  -- prior_network = ov.single.convert_human_to_mouse_network(prior_network,server_name='asia')
# Line 16:  prior_network -- prior_network
# Line 18:  prior_network.to_csv('result/combined_network_Mouse.txt.gz',sep='\t') -- prior_network.to_csv('result/combined_network_Mouse.txt.gz',sep='\t')
# Line 20:  prior_network=ov.read('result/combined_network_Mouse.txt.gz',index_col=0) -- prior_network=ov.read('result/combined_network_Mouse.txt.gz',index_col=0)
# Line 22:  CEFCON_obj = ov.single.pyCEFCON(adata, prior_network, repeats=5, solver='GUROBI') -- CEFCON_obj = ov.single.pyCEFCON(adata, prior_network, repeats=5, solver='GUROBI')
# Line 23:  CEFCON_obj -- CEFCON_obj
# Line 25:  CEFCON_obj.preprocess() -- CEFCON_obj.preprocess()
# Line 27:  CEFCON_obj.train() -- CEFCON_obj.train()
# Line 29:  CEFCON_obj.predicted_driver_regulators() -- CEFCON_obj.predicted_driver_regulators()
# Line 31:  CEFCON_obj.cefcon_results_dict['E_pseudotime'].driver_regulator.head() -- CEFCON_obj.cefcon_results_dict['E_pseudotime'].driver_regulator.head()
# Line 33:  CEFCON_obj.predicted_RGM() -- CEFCON_obj.predicted_RGM()
# Line 35:  CEFCON_obj.cefcon_results_dict['E_pseudotime'] -- CEFCON_obj.cefcon_results_dict['E_pseudotime']
# Line 37:  lineage = 'E_pseudotime' -- lineage = 'E_pseudotime'
# Line 38:  result = CEFCON_obj.cefcon_results_dict[lineage] -- result = CEFCON_obj.cefcon_results_dict[lineage]
# Line 40:  gene_ad=sc.AnnData(result.gene_embedding) -- gene_ad=sc.AnnData(result.gene_embedding)
# Line 41:  sc.pp.neighbors(gene_ad, n_neighbors=30, use_rep='X') -- sc.pp.neighbors(gene_ad, n_neighbors=30, use_rep='X')
# Line 43:  sc.tl.leiden(gene_ad, resolution=1) -- sc.tl.leiden(gene_ad, resolution=1)
# Line 44:  sc.tl.umap(gene_ad, n_components=2, min_dist=0.3) -- sc.tl.umap(gene_ad, n_components=2, min_dist=0.3)
# Line 46:  ov.utils.embedding(gene_ad,basis='X_umap',legend_loc='on data', -- ov.utils.embedding(gene_ad,basis='X_umap',legend_loc='on data',
# Line 47:                        legend_fontsize=8, legend_fontoutline=2, --                        legend_fontsize=8, legend_fontoutline=2,
# Line 48:                   color='leiden',frameon='small',title='Leiden clustering using CEFCON\nderived gene embeddings') --                   color='leiden',frameon='small',title='Leiden clustering using CEFCON\nderived gene embeddings')
# Line 50:  import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 51:  import seaborn as sns -- import seaborn as sns
# Line 52:  data_for_plot = result.driver_regulator[result.driver_regulator['is_driver_regulator']] -- data_for_plot = result.driver_regulator[result.driver_regulator['is_driver_regulator']]
# Line 53:  data_for_plot = data_for_plot[0:20] -- data_for_plot = data_for_plot[0:20]
# Line 55:  plt.figure(figsize=(2, 20 * 0.2)) -- plt.figure(figsize=(2, 20 * 0.2))
# Line 56:  sns.set_theme(style='ticks', font_scale=0.5) -- sns.set_theme(style='ticks', font_scale=0.5)
# Line 58:  ax = sns.barplot(x='influence_score', y=data_for_plot.index, data=data_for_plot, orient='h', -- ax = sns.barplot(x='influence_score', y=data_for_plot.index, data=data_for_plot, orient='h',
# Line 59:                  palette=sns.color_palette(f"ch:start=.5,rot=-.5,reverse=1,dark=0.4", n_colors=20)) --                  palette=sns.color_palette(f"ch:start=.5,rot=-.5,reverse=1,dark=0.4", n_colors=20))
# Line 60:  ax.set_title(result.name) -- ax.set_title(result.name)
# Line 61:  ax.set_xlabel('Influence score') -- ax.set_xlabel('Influence score')
# Line 62:  ax.set_ylabel('Driver regulators') -- ax.set_ylabel('Driver regulators')
# Line 64:  ax.spines['left'].set_position(('outward', 10)) -- ax.spines['left'].set_position(('outward', 10))
# Line 65:  ax.spines['bottom'].set_position(('outward', 10)) -- ax.spines['bottom'].set_position(('outward', 10))
# Line 66:  plt.xticks(fontsize=12) -- plt.xticks(fontsize=12)
# Line 67:  plt.yticks(fontsize=12) -- plt.yticks(fontsize=12)
# Line 69:  plt.grid(False) -- plt.grid(False)
# Line 70:  ax.spines['top'].set_visible(False) -- ax.spines['top'].set_visible(False)
# Line 71:  ax.spines['right'].set_visible(False) -- ax.spines['right'].set_visible(False)
# Line 72:  ax.spines['bottom'].set_visible(True) -- ax.spines['bottom'].set_visible(True)
# Line 73:  ax.spines['left'].set_visible(True) -- ax.spines['left'].set_visible(True)
# Line 75:  plt.title('E_pseudotime',fontsize=12) -- plt.title('E_pseudotime',fontsize=12)
# Line 76:  plt.xlabel('Influence score',fontsize=12) -- plt.xlabel('Influence score',fontsize=12)
# Line 77:  plt.ylabel('Driver regulon',fontsize=12) -- plt.ylabel('Driver regulon',fontsize=12)
# Line 79:  sns.despine() -- sns.despine()
# Line 81:  result.plot_driver_genes_Venn() -- result.plot_driver_genes_Venn()
# Line 83:  adata_lineage = adata[adata.obs_names[adata.obs[result.name].notna()],:] -- adata_lineage = adata[adata.obs_names[adata.obs[result.name].notna()],:]
# Line 85:  result.plot_RGM_activity_heatmap(cell_label=adata_lineage.obs['cell_type_finely'], -- result.plot_RGM_activity_heatmap(cell_label=adata_lineage.obs['cell_type_finely'],
# Line 86:                                  type='out',col_cluster=True,bbox_to_anchor=(1.48, 0.25)) --                                  type='out',col_cluster=True,bbox_to_anchor=(1.48, 0.25))
```