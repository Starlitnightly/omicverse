```
# Line 1: Imports the omicverse library as ov -- import omicverse as ov
# Line 2: Imports the scanpy library as sc -- import scanpy as sc
# Line 3: Imports the matplotlib.pyplot library as plt -- import matplotlib.pyplot as plt
# Line 5: Sets the plotting style using omicverse's plot_set function -- ov.plot_set()
# Line 7: Downloads gene ID annotation pairs using omicverse's utility function -- ov.utils.download_geneid_annotation_pair()
# Line 9: Reads count data from a file using omicverse's read function -- data=ov.read('data/counts.txt',index_col=0,header=1)
# Line 11: Renames the columns of the data by extracting the file name and removing the '.bam' suffix -- data.columns=[i.split('/')[-1].replace('.bam','') for i in data.columns]
# Line 12: Displays the head of the data -- data.head()
# Line 14: Maps gene IDs in the data using omicverse's Matrix_ID_mapping function with a given gene pair file. -- data=ov.bulk.Matrix_ID_mapping(data,'genesets/pair_GRCm39.tsv')
# Line 15: Displays the head of the updated data after mapping. -- data.head()
# Line 17: Creates a DEG object from the data using omicverse's pyDEG function -- dds=ov.bulk.pyDEG(data)
# Line 19: Removes duplicate indices from the DEG object -- dds.drop_duplicates_index()
# Line 20: Prints a success message after removing duplicate indices -- print('... drop_duplicates_index success')
# Line 22: Normalizes the data in the DEG object -- dds.normalize()
# Line 23: Prints a success message after normalizing the data -- print('... estimateSizeFactors and normalize success')
# Line 25: Defines a list of treatment groups for DEG analysis -- treatment_groups=['4-3','4-4']
# Line 26: Defines a list of control groups for DEG analysis -- control_groups=['1--1','1--2']
# Line 27: Performs differential gene expression analysis using t-tests with given treatment and control groups -- result=dds.deg_analysis(treatment_groups,control_groups,method='ttest')
# Line 28: Displays the head of the DEG analysis result -- result.head()
# Line 30: Prints the shape of the DEG analysis result dataframe -- print(result.shape)
# Line 31: Filters the result dataframe, keeping only rows where 'log2(BaseMean)' is greater than 1 -- result=result.loc[result['log2(BaseMean)']>1]
# Line 32: Prints the shape of the filtered DEG analysis result dataframe -- print(result.shape)
# Line 35: Sets fold change and p-value thresholds for the DEG results -- dds.foldchange_set(fc_threshold=-1,
# Line 36: Sets the p-value threshold to 0.05 --                    pval_threshold=0.05,
# Line 37: Sets the maximum log p-value to 6 --                    logp_max=6)
# Line 39: Creates a volcano plot of the DEG analysis results -- dds.plot_volcano(title='DEG Analysis',figsize=(4,4),
# Line 40: Specifies the number of genes to label in the volcano plot and sets font size --                  plot_genes_num=8,plot_genes_fontsize=12,)
# Line 42: Generates boxplots for specified genes, comparing treatment and control groups -- dds.plot_boxplot(genes=['Ckap2','Lef1'],treatment_groups=treatment_groups,
# Line 43: Specifies boxplot's figure size, font size, and legend position --                 control_groups=control_groups,figsize=(2,3),fontsize=12,
# Line 44: Specifies boxplot's legend position --                  legend_bbox=(2,0.55))
# Line 46: Generates boxplot for gene 'Ckap2', comparing treatment and control groups -- dds.plot_boxplot(genes=['Ckap2'],treatment_groups=treatment_groups,
# Line 47: Specifies boxplot's figure size, font size, and legend position --                 control_groups=control_groups,figsize=(2,3),fontsize=12,
# Line 48: Specifies boxplot's legend position --                  legend_bbox=(2,0.55))
# Line 50: Downloads pathway database using omicverse's utility function -- ov.utils.download_pathway_database()
# Line 52: Prepares pathway dictionary from the specified file using omicverse's geneset_prepare -- pathway_dict=ov.utils.geneset_prepare('genesets/WikiPathways_2019_Mouse.txt',organism='Mouse')
# Line 54: Extracts a list of differentially expressed genes from DEG analysis results -- deg_genes=dds.result.loc[dds.result['sig']!='normal'].index.tolist()
# Line 55: Performs gene set enrichment analysis using omicverse's geneset_enrichment function -- enr=ov.bulk.geneset_enrichment(gene_list=deg_genes,
# Line 56: Provides pathways dict and p-value type, and organism for geneset enrichment --                                 pathways_dict=pathway_dict,
# Line 57: Automatically determine the type of p-value to use in enrichment analysis --                                 pvalue_type='auto',
# Line 58: Provides the organism information for geneset enrichment --                                 organism='mouse')
# Line 60: Plots the gene set enrichment results using omicverse's geneset_plot function -- ov.bulk.geneset_plot(enr,figsize=(2,5),fig_title='Wiki Pathway enrichment',
# Line 61: Sets the color bar location and the bounding box for plot --                     cax_loc=[2, 0.45, 0.5, 0.02],
# Line 62: Specifies the bounding box for plot and node diameter --                     bbox_to_anchor_used=(-0.25, -13),node_diameter=10,
# Line 63: Sets custom ticks and text knockout for plot --                      custom_ticks=[5,7],text_knock=3,
# Line 64: Sets the color map to 'Reds' --                     cmap='Reds')
# Line 66: Prepares GO Biological Process pathway dictionary -- pathway_dict=ov.utils.geneset_prepare('genesets/GO_Biological_Process_2023.txt',organism='Mouse')
# Line 67: Performs GO Biological Process gene set enrichment analysis -- enr_go_bp=ov.bulk.geneset_enrichment(gene_list=deg_genes,
# Line 68: Provides pathways dict and p-value type, and organism for geneset enrichment --                                pathways_dict=pathway_dict,
# Line 69: Automatically determine the type of p-value to use in enrichment analysis --                                pvalue_type='auto',
# Line 70: Provides the organism information for geneset enrichment --                                organism='mouse')
# Line 71: Prepares GO Molecular Function pathway dictionary -- pathway_dict=ov.utils.geneset_prepare('genesets/GO_Molecular_Function_2023.txt',organism='Mouse')
# Line 72: Performs GO Molecular Function gene set enrichment analysis -- enr_go_mf=ov.bulk.geneset_enrichment(gene_list=deg_genes,
# Line 73: Provides pathways dict and p-value type, and organism for geneset enrichment --                                pathways_dict=pathway_dict,
# Line 74: Automatically determine the type of p-value to use in enrichment analysis --                                pvalue_type='auto',
# Line 75: Provides the organism information for geneset enrichment --                                organism='mouse')
# Line 76: Prepares GO Cellular Component pathway dictionary -- pathway_dict=ov.utils.geneset_prepare('genesets/GO_Cellular_Component_2023.txt',organism='Mouse')
# Line 77: Performs GO Cellular Component gene set enrichment analysis -- enr_go_cc=ov.bulk.geneset_enrichment(gene_list=deg_genes,
# Line 78: Provides pathways dict and p-value type, and organism for geneset enrichment --                                pathways_dict=pathway_dict,
# Line 79: Automatically determine the type of p-value to use in enrichment analysis --                                pvalue_type='auto',
# Line 80: Provides the organism information for geneset enrichment --                                organism='mouse')
# Line 82: Creates a dictionary containing GO enrichment results -- enr_dict={'BP':enr_go_bp,
# Line 83: Adds Molecular Function GO enrichment results in the dict --          'MF':enr_go_mf,
# Line 84: Adds Cellular Component GO enrichment results in the dict --          'CC':enr_go_cc}
# Line 85: Defines color mapping for the GO categories -- colors_dict={
# Line 86: Adds Red color for Biological Process GO term --     'BP':ov.pl.red_color[1],
# Line 87: Adds Green color for Molecular Function GO term --     'MF':ov.pl.green_color[1],
# Line 88: Adds Blue color for Cellular Component GO term --     'CC':ov.pl.blue_color[1],
# Line 89: Closes color mapping dictionary -- }
# Line 91: Plots multiple gene set enrichment results using omicverse's function -- ov.bulk.geneset_plot_multi(enr_dict,colors_dict,num=3,
# Line 92: Specifies the figure size --                    figsize=(2,5),
# Line 93: Sets the text knockout and fontsize --                    text_knock=3,fontsize=8,
# Line 94: Sets the color map to 'Reds' --                     cmap='Reds'
# Line 95: Closes function call --                   )
# Line 98: Defines a function `geneset_plot_multi` to plot multiple gene set enrichment results. -- def geneset_plot_multi(enr_dict,colors_dict,num:int=5,fontsize=10,
# Line 99: Specifies title, x label, figure size, color map, text knockout, max size, and axis --                         fig_title:str='',fig_xlabel:str='Fractions of genes',
# Line 100: Specifies figure size, color map, text knock, max size, and axes --                         figsize:tuple=(2,4),cmap:str='YlGnBu',
# Line 101: Specifies text knock, max size, and axes --                         text_knock:int=5,text_maxsize:int=20,ax=None,
# Line 102: Closes function definition --                         ):
# Line 103: Imports necessary classes from PyComplexHeatmap library --     from PyComplexHeatmap import HeatmapAnnotation,DotClustermapPlotter,anno_label,anno_simple,AnnotationBase
# Line 104: Iterates through the enrichment dictionaries and adds a 'Type' column --     for key in enr_dict.keys():
# Line 105: Adds 'Type' column in each dictionary --         enr_dict[key]['Type']=key
# Line 106: Concatenates the top 'num' rows of all enrichment results into a single DataFrame --     enr_all=pd.concat([enr_dict[i].iloc[:num] for i in enr_dict.keys()],axis=0)
# Line 107: Shortens and sets text for plot term labels --     enr_all['Term']=[ov.utils.plot_text_set(i.split('(')[0],text_knock=text_knock,text_maxsize=text_maxsize) for i in enr_all.Term.tolist()]
# Line 108: Sets the index of the DataFrame to the modified term labels --     enr_all.index=enr_all.Term
# Line 109: Stores the index values in a new column named "Term1" --     enr_all['Term1']=[i for i in enr_all.index.tolist()]
# Line 110: Deletes the original term column --     del enr_all['Term']
# Line 112: Assigns the defined colors for each type of GO analysis --     colors=colors_dict
# Line 114: Creates a HeatmapAnnotation for left side with labels, type mapping and axis specification --     left_ha = HeatmapAnnotation(
# Line 115: Configures label annotation with merging, rotation, colors, and rel position --                           label=anno_label(enr_all.Type, merge=True,rotation=0,colors=colors,relpos=(1,0.8)),
# Line 116: Configures Category annotation with colors, legend, text etc --                           Category=anno_simple(enr_all.Type,cmap='Set1',
# Line 117: Adds details for annotation with text, legend and colors --                                            add_text=False,legend=False,colors=colors),
# Line 118: Specifies the axis for left annotation and label properties --                            axis=0,verbose=0,label_kws={'rotation':45,'horizontalalignment':'left','visible':False})
# Line 119: Creates a HeatmapAnnotation for right side with labels, type mapping and axis specification --     right_ha = HeatmapAnnotation(
# Line 120: Configures label annotation with merging, rotation, colors, position, arrows etc --                               label=anno_label(enr_all.Term1, merge=True,rotation=0,relpos=(0,0.5),arrowprops=dict(visible=True),
# Line 121: Sets colors to labels by mapping to dict with set colors by each type of annotation and sets font --                                                colors=enr_all.assign(color=enr_all.Type.map(colors)).set_index('Term1').color.to_dict(),
# Line 122: Sets font size and luminance values --                                               fontsize=fontsize,luminance=0.8,height=2),
# Line 123: Sets the axis and label keyword of annotation --                                axis=0,verbose=0,#label_kws={'rotation':45,'horizontalalignment':'left'},
# Line 124: Specifies the orientation --                                 orientation='right')
# Line 125: Creates subplots if no axes object given --     if ax==None:
# Line 126: Creates the figure and axes object with specific figure size --         fig, ax = plt.subplots(figsize=figsize) 
# Line 127: Assigns the provided axis object if one provided --     else:
# Line 128: Sets the axes to the provided axes object --         ax=ax
# Line 130: Creates dotclustermap plot with data, x,y values and heatmap properties --     cm = DotClustermapPlotter(data=enr_all, x='fraction',y='Term1',value='logp',c='logp',s='num',
# Line 131: Sets the color map --                               cmap=cmap,
# Line 132: Sets the row clustering --                               row_cluster=True,#col_cluster=True,#hue='Group',
# Line 134: Sets the vmin and vmax value in the heatmap colorbar --                               vmin=-1*np.log10(0.1),vmax=-1*np.log10(1e-10),
# Line 137: Sets row and column label properties --                               show_rownames=True,show_colnames=False,row_dendrogram=False,
# Line 138: Specifies the side of the labels --                               col_names_side='top',row_names_side='right',
# Line 139: Sets the label properties of the x-axis ticks --                               xticklabels_kws={'labelrotation': 30, 'labelcolor': 'blue','labelsize':fontsize},
# Line 141: Sets left and right annotation properties --                               left_annotation=left_ha,right_annotation=right_ha,
# Line 142: Sets spines property of the plot --                               spines=False,
# Line 143: Splits rows based on type of GO term --                               row_split=enr_all.Type,# row_split_gap=1,
# Line 145: Sets the verbosity, legend properties --                               verbose=1,legend_gap=10,
# Line 148: Sets x label for plot --                               xlabel='Fractions of genes',xlabel_side="bottom",
# Line 149: Sets label padding, font weight, and font size of x axis --                               xlabel_kws=dict(labelpad=8,fontweight='normal',fontsize=fontsize+2),
# Line 151: Gets axes from figure --     tesr=plt.gcf().axes
# Line 152: Iterates through each axis on figure --     for ax in plt.gcf().axes:
# Line 153: Check if each of the axes contains get_xlabel property --         if hasattr(ax, 'get_xlabel'):
# Line 154: Checks if xlabel property is Fractions of genes --             if ax.get_xlabel() == 'Fractions of genes':  # 假设 colorbar 有一个特定的标签
# Line 155: Sets the cbar object as the axes object --                 cbar = ax
# Line 156: Disables grid in the colorbar --                 cbar.grid(False)
# Line 157: Checks if y label is logp --             if ax.get_ylabel() == 'logp':  # 假设 colorbar 有一个特定的标签
# Line 158: Sets the cbar object as the axes object --                 cbar = ax
# Line 159: Sets the label size of the axis --                 cbar.tick_params(labelsize=fontsize+2)
# Line 160: Sets the y label of the colorbar --                 cbar.set_ylabel(r'$−Log_{10}(P_{adjusted})$',fontsize=fontsize+2)
# Line 161: Disables grid in the colorbar --                 cbar.grid(False)
# Line 162: Returns the axis object --     return ax
```
