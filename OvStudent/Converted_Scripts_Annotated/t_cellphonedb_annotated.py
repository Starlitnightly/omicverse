```
# Line 1: Imports the scanpy library for single-cell analysis. -- import scanpy as sc
# Line 2: Imports the matplotlib.pyplot library for plotting. -- import matplotlib.pyplot as plt
# Line 3: Imports the pandas library for data manipulation. -- import pandas as pd
# Line 4: Imports the numpy library for numerical operations. -- import numpy as np
# Line 5: Imports the omicverse library for omics data analysis. -- import omicverse as ov
# Line 6: Imports the os library for operating system functionalities. -- import os
# Line 8: Sets the plotting style using omicverse. -- ov.plot_set()
# Line 11: Reads an AnnData object from a HDF5 file. -- adata=sc.read('data/cpdb/normalised_log_counts.h5ad')
# Line 12: Filters the AnnData object to include only specified cell labels. -- adata=adata[adata.obs['cell_labels'].isin(['eEVT','iEVT','EVT_1','EVT_2','DC','dNK1','dNK2','dNK3',
# Line 13:                                          'VCT','VCT_CCC','VCT_fusing','VCT_p','GC','SCT'])]
# Line 14: Displays the filtered AnnData object. -- adata
# Line 16: Generates an embedding plot using UMAP, colored by cell labels with custom color palette. -- ov.pl.embedding(adata,
# Line 17:                basis='X_umap',
# Line 18:                color='cell_labels',
# Line 19:                frameon='small',
# Line 20:                palette=ov.pl.red_color+ov.pl.blue_color+ov.pl.green_color+ov.pl.orange_color+ov.pl.purple_color)
# Line 22: Finds the maximum value in the AnnData's expression matrix. -- adata.X.max()
# Line 24: Filters cells based on a minimum gene count. -- sc.pp.filter_cells(adata, min_genes=200)
# Line 25: Filters genes based on a minimum cell count. -- sc.pp.filter_genes(adata, min_cells=3)
# Line 26: Creates a new AnnData object with filtered data, observations, and variables. -- adata1=sc.AnnData(adata.X,obs=pd.DataFrame(index=adata.obs.index),
# Line 27:                           var=pd.DataFrame(index=adata.var.index))
# Line 28: Writes the new AnnData object to an HDF5 file. -- adata1.write_h5ad('data/cpdb/norm_log.h5ad',compression='gzip')
# Line 29: Displays the new AnnData object. -- adata1
# Line 32: Creates a pandas DataFrame for metadata containing cell IDs and labels. -- df_meta = pd.DataFrame(data={'Cell':list(adata[adata1.obs.index].obs.index),
# Line 33:                              'cell_type':[ i for i in adata[adata1.obs.index].obs['cell_labels']]
# Line 34:                            })
# Line 35: Sets the 'Cell' column as the index of the DataFrame. -- df_meta.set_index('Cell', inplace=True)
# Line 36: Saves the metadata DataFrame to a tab-separated file. -- df_meta.to_csv('data/cpdb/meta.tsv', sep = '\t')
# Line 38: Imports the os module. -- import os
# Line 39: Gets the current working directory. -- os.getcwd()
# Line 41: Defines the path to the CellphoneDB database file. -- cpdb_file_path = '/Users/fernandozeng/Desktop/analysis/cellphonedb-data/cellphonedb.zip'
# Line 42: Defines the path to the metadata file. -- meta_file_path = os.getcwd()+'/data/cpdb/meta.tsv'
# Line 43: Defines the path to the normalized count matrix file. -- counts_file_path = os.getcwd()+'/data/cpdb/norm_log.h5ad'
# Line 44: Sets the microenvironment file path to None. -- microenvs_file_path = None
# Line 45: Sets the active transcription factor file path to None. -- active_tf_path = None
# Line 46: Defines the output path for CellphoneDB results. -- out_path =os.getcwd()+'/data/cpdb/test_cellphone'
# Line 48: Imports the CellphoneDB statistical analysis method. -- from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
# Line 50: Executes CellphoneDB statistical analysis with specified parameters. -- cpdb_results = cpdb_statistical_analysis_method.call(
# Line 51:     cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
# Line 52:     meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
# Line 53:     counts_file_path = counts_file_path,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
# Line 54:     counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
# Line 55:     active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
# Line 56:     microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
# Line 57:     score_interactions = True,                       # optional: whether to score interactions or not.
# Line 58:     iterations = 1000,                               # denotes the number of shufflings performed in the analysis.
# Line 59:     threshold = 0.1,                                 # defines the min % of cells expressing a gene for this to be employed in the analysis.
# Line 60:     threads = 10,                                     # number of threads to use in the analysis.
# Line 61:     debug_seed = 42,                                 # debug randome seed. To disable >=0.
# Line 62:     result_precision = 3,                            # Sets the rounding for the mean values in significan_means.
# Line 63:     pvalue = 0.05,                                   # P-value threshold to employ for significance.
# Line 64:     subsampling = False,                             # To enable subsampling the data (geometri sketching).
# Line 65:     subsampling_log = False,                         # (mandatory) enable subsampling log1p for non log-transformed data inputs.
# Line 66:     subsampling_num_pc = 100,                        # Number of componets to subsample via geometric skectching (dafault: 100).
# Line 67:     subsampling_num_cells = 1000,                    # Number of cells to subsample (integer) (default: 1/3 of the dataset).
# Line 68:     separator = '|',                                 # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
# Line 69:     debug = False,                                   # Saves all intermediate tables employed during the analysis in pkl format.
# Line 70:     output_path = out_path,                          # Path to save results.
# Line 71:     output_suffix = None                             # Replaces the timestamp in the output files by a user defined string in the  (default: None).
# Line 73: Saves the CellphoneDB results to a pickle file. -- ov.utils.save(cpdb_results,'data/cpdb/gex_cpdb_test.pkl')
# Line 75: Loads CellphoneDB results from a pickle file. -- cpdb_results=ov.utils.load('data/cpdb/gex_cpdb_test.pkl')
# Line 77: Calculates cell interaction edges for the network visualization. -- interaction=ov.single.cpdb_network_cal(adata = adata,
# Line 78:         pvals = cpdb_results['pvalues'],
# Line 79:         celltype_key = "cell_labels",)
# Line 81: Displays the head of interaction edges dataframe. -- interaction['interaction_edges'].head()
# Line 83: Sets the plotting style using omicverse. -- ov.plot_set()
# Line 85: Creates a figure and axes for a heatmap plot. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 86: Generates a CellphoneDB heatmap using the interaction edges. -- ov.pl.cpdb_heatmap(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 87:                    fontsize=11,
# Line 88:           ax=ax,legend_kws={'fontsize':12,'bbox_to_anchor':(5, -0.9),'loc':'center left',})
# Line 90: Creates a figure and axes for a heatmap plot with specified source cells. -- fig, ax = plt.subplots(figsize=(2,4))
# Line 91: Generates a CellphoneDB heatmap with source cell subset. -- ov.pl.cpdb_heatmap(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 92:                    source_cells=['EVT_1','EVT_2','dNK1','dNK2','dNK3'],
# Line 93:           ax=ax,legend_kws={'fontsize':12,'bbox_to_anchor':(5, -0.9),'loc':'center left',})
# Line 95: Generates a CellphoneDB chord diagram. -- fig=ov.pl.cpdb_chord(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 96:           count_min=60,fontsize=12,padding=50,radius=100,save=None,)
# Line 97: Displays the chord diagram figure. -- fig.show()
# Line 99: Creates a figure and axes for a network plot. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 100: Generates a CellphoneDB network graph with cell labels. -- ov.pl.cpdb_network(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 101:              counts_min=60,
# Line 102:             nodesize_scale=5,
# Line 103:                   ax=ax)
# Line 105: Creates a figure and axes for a network plot with specified source cells. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 106: Generates a CellphoneDB network with a source cell subset. -- ov.pl.cpdb_network(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 107:             counts_min=60,
# Line 108:             nodesize_scale=5,
# Line 109:             source_cells=['EVT_1','EVT_2','dNK1','dNK2','dNK3'],
# Line 110:             ax=ax)
# Line 112: Creates a figure and axes for a network plot with specified target cells. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 113: Generates a CellphoneDB network with a target cell subset. -- ov.pl.cpdb_network(adata,interaction['interaction_edges'],celltype_key='cell_labels',
# Line 114:             counts_min=60,
# Line 115:             nodesize_scale=5,
# Line 116:             target_cells=['EVT_1','EVT_2','dNK1','dNK2','dNK3'],
# Line 117:             ax=ax)
# Line 119: Generates a CellphoneDB network plot with detailed customizations. -- ov.single.cpdb_plot_network(adata=adata,
# Line 120:                   interaction_edges=interaction['interaction_edges'],
# Line 121:                   celltype_key='cell_labels',
# Line 122:                   nodecolor_dict=None,title='EVT Network',
# Line 123:                   edgeswidth_scale=25,nodesize_scale=10,
# Line 124:                   pos_scale=1,pos_size=10,figsize=(6,6),
# Line 125:                   legend_ncol=3,legend_bbox=(0.8,0.2),legend_fontsize=10)
# Line 127: Assigns the interaction edges to a new variable sub_i. -- sub_i=interaction['interaction_edges']
# Line 128: Filters the interaction edges for source cells. -- sub_i=sub_i.loc[sub_i['SOURCE'].isin(['EVT_1','EVT_2','dNK1','dNK2','dNK3'])]
# Line 129: Filters the interaction edges for target cells. -- sub_i=sub_i.loc[sub_i['TARGET'].isin(['EVT_1','EVT_2','dNK1','dNK2','dNK3'])]
# Line 131: Creates a subset AnnData object based on specified cell labels. -- sub_adata=adata[adata.obs['cell_labels'].isin(['EVT_1','EVT_2','dNK1','dNK2','dNK3'])]
# Line 132: Displays the sub-AnnData object. -- sub_adata
# Line 134: Generates a CellphoneDB network plot for a subset of cells. -- ov.single.cpdb_plot_network(adata=sub_adata,
# Line 135:                   interaction_edges=sub_i,
# Line 136:                   celltype_key='cell_labels',
# Line 137:                   nodecolor_dict=None,title='Sub-EVT Network',
# Line 138:                   edgeswidth_scale=25,nodesize_scale=1,
# Line 139:                   pos_scale=1,pos_size=10,figsize=(5,5),
# Line 140:                   legend_ncol=3,legend_bbox=(0.8,0.2),legend_fontsize=10)
# Line 142: Generates a CellphoneDB chord diagram for the sub-AnnData object. -- fig=ov.pl.cpdb_chord(sub_adata,sub_i,celltype_key='cell_labels',
# Line 143:           count_min=10,fontsize=12,padding=60,radius=100,save=None,)
# Line 144: Displays the chord diagram figure. -- fig.show()
# Line 146: Creates a figure and axes for a network plot for the sub-AnnData object. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 147: Generates a CellphoneDB network graph for the sub-AnnData object. -- ov.pl.cpdb_network(sub_adata,sub_i,celltype_key='cell_labels',
# Line 148:              counts_min=10,
# Line 149:             nodesize_scale=5,
# Line 150:                   ax=ax)
# Line 152: Creates a figure and axes for a heatmap plot for the sub-AnnData object. -- fig, ax = plt.subplots(figsize=(3,3))
# Line 153: Generates a CellphoneDB heatmap for the sub-AnnData object. -- ov.pl.cpdb_heatmap(sub_adata,sub_i,celltype_key='cell_labels',
# Line 154:           ax=ax,legend_kws={'fontsize':12,'bbox_to_anchor':(5, -0.9),'loc':'center left',})
# Line 156: Extracts exact target interaction means from the CellphoneDB results. -- sub_means=ov.single.cpdb_exact_target(cpdb_results['means'],['eEVT','iEVT'])
# Line 157: Extracts exact source interaction means from the sub_means dataframe. -- sub_means=ov.single.cpdb_exact_source(sub_means,['dNK1','dNK2','dNK3'])
# Line 158: Displays the head of the sub_means DataFrame. -- sub_means.head()
# Line 160: Generates a CellphoneDB interacting heatmap with specific source and target cells. -- ov.pl.cpdb_interacting_heatmap(adata=adata,
# Line 161:                          celltype_key='cell_labels',
# Line 162:                             means=cpdb_results['means'],
# Line 163:                             pvalues=cpdb_results['pvalues'],
# Line 164:                             source_cells=['dNK1','dNK2','dNK3'],
# Line 165:                             target_cells=['eEVT','iEVT'],
# Line 166:                             plot_secret=True,
# Line 167:                             min_means=3,
# Line 168:                             nodecolor_dict=None,
# Line 169:                             ax=None,
# Line 170:                             figsize=(2,6),
# Line 171:                             fontsize=10,)
# Line 173: Generates a CellphoneDB group heatmap with specified source and target cells. -- ov.pl.cpdb_group_heatmap(adata=adata,
# Line 174:                          celltype_key='cell_labels',
# Line 175:                             means=cpdb_results['means'],
# Line 176:                             cmap={'Target':'Blues','Source':'Reds'},
# Line 177:                             source_cells=['dNK1','dNK2','dNK3'],
# Line 178:                             target_cells=['eEVT','iEVT'],
# Line 179:                             plot_secret=True,
# Line 180:                             min_means=3,
# Line 181:                             nodecolor_dict=None,
# Line 182:                             ax=None,
# Line 183:                             figsize=(2,6),
# Line 184:                             fontsize=10,)
# Line 186: Generates a CellphoneDB interacting network with specified source and target cells. -- ov.pl.cpdb_interacting_network(adata=adata,
# Line 187:                          celltype_key='cell_labels',
# Line 188:                             means=cpdb_results['means'],
# Line 189:                             source_cells=['dNK1','dNK2','dNK3'],
# Line 190:                             target_cells=['eEVT','iEVT'],
# Line 191:                             means_min=1,
# Line 192:                              means_sum_min=1,
# Line 193:                             nodecolor_dict=None,
# Line 194:                             ax=None,
# Line 195:                             figsize=(6,6),
# Line 196:                             fontsize=10)
# Line 198: Filters the sub_means DataFrame to remove rows with null 'gene_a'. -- sub_means=sub_means.loc[~sub_means['gene_a'].isnull()]
# Line 199: Filters the sub_means DataFrame to remove rows with null 'gene_b'. -- sub_means=sub_means.loc[~sub_means['gene_b'].isnull()]
# Line 200: Creates a list of genes from 'gene_a' and 'gene_b' columns. -- enrichr_genes=sub_means['gene_a'].tolist()+sub_means['gene_b'].tolist()
# Line 202: Prepares a dictionary of pathway gene sets. -- pathway_dict=ov.utils.geneset_prepare('genesets/GO_Biological_Process_2023.txt',organism='Human')
# Line 205: Performs gene set enrichment analysis. -- enr=ov.bulk.geneset_enrichment(gene_list=enrichr_genes,
# Line 206:                                 pathways_dict=pathway_dict,
# Line 207:                                 pvalue_type='auto',
# Line 208:                                 organism='human')
# Line 210: Sets the plotting style using omicverse. -- ov.plot_set()
# Line 211: Generates a gene set enrichment plot. -- ov.bulk.geneset_plot(enr,figsize=(2,4),fig_title='GO-Bio(EVT)',
# Line 212:                     cax_loc=[2, 0.45, 0.5, 0.02],num=8,
# Line 213:                     bbox_to_anchor_used=(-0.25, -13),custom_ticks=[10,100],
# Line 214:                     cmap='Greens')
```