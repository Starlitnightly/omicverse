```
# Line 1:  -- import omicverse as ov
# Line 1: Imports the omicverse library and aliases it as ov.
# Line 2:  -- import scanpy as sc
# Line 2: Imports the scanpy library and aliases it as sc.
# Line 4:  -- ov.plot_set()
# Line 4: Sets the plotting style for omicverse.
# Line 6:  -- adata = ov.read('data/DentateGyrus/10X43_1.h5ad')
# Line 6: Reads an AnnData object from an h5ad file into the variable 'adata' using omicverse's read function.
# Line 7:  -- adata
# Line 7: Displays the AnnData object 'adata'.
# Line 9:  -- optim_palette=ov.pl.optim_palette(adata,basis='X_umap',colors='clusters')
# Line 9: Generates an optimized color palette for plotting based on UMAP embeddings and 'clusters' using omicverse.
# Line 11:  -- import matplotlib.pyplot as plt
# Line 11: Imports the matplotlib.pyplot module as plt.
# Line 12:  -- fig,ax=plt.subplots(figsize = (4,4))
# Line 12: Creates a matplotlib figure and an axes object with a specified size of 4x4.
# Line 13:  -- ov.pl.embedding(adata,
# Line 13: Calls omicverse's embedding plot function on the 'adata' object.
# Line 14:  --                 basis='X_umap',
# Line 14: Specifies 'X_umap' as the embedding basis for the plot.
# Line 15:  --                color='clusters',
# Line 15: Colors the embedding plot based on the 'clusters' annotation in the AnnData object.
# Line 16:  --                frameon='small',
# Line 16: Sets a small frame for the plot.
# Line 17:  --                show=False,
# Line 17: Prevents the plot from being shown immediately, allowing for further customization.
# Line 18:  --                palette=optim_palette,
# Line 18: Sets the color palette for the plot using the previously generated optimized palette.
# Line 19:  --                ax=ax,)
# Line 19: Specifies the previously created axes object 'ax' to draw the plot on.
# Line 20:  -- plt.title('Cell Type of DentateGyrus',fontsize=15)
# Line 20: Sets the title of the plot to 'Cell Type of DentateGyrus' with a font size of 15.
# Line 22:  -- ov.pl.embedding(adata,
# Line 22: Calls omicverse's embedding plot function on the 'adata' object again.
# Line 23:  --                 basis='X_umap',
# Line 23: Specifies 'X_umap' as the embedding basis for the plot.
# Line 24:  --                color='age(days)',
# Line 24: Colors the embedding plot based on the 'age(days)' annotation.
# Line 25:  --                frameon='small',
# Line 25: Sets a small frame for the plot.
# Line 26:  --                show=False,)
# Line 26: Prevents the plot from being shown immediately.
# Line 28:  -- import matplotlib.pyplot as plt
# Line 28: Imports the matplotlib.pyplot module as plt.
# Line 29:  -- fig,ax=plt.subplots(figsize = (1,4))
# Line 29: Creates a matplotlib figure and an axes object with a specified size of 1x4.
# Line 30:  -- ov.pl.cellproportion(adata=adata,celltype_clusters='clusters',
# Line 30: Creates a cell proportion plot using omicverse, specifying the 'clusters' annotation as cell types.
# Line 31:  --                     groupby='age(days)',legend=True,ax=ax)
# Line 31: Groups the cell proportion plot by 'age(days)', displays the legend, and draws on the specified axes.
# Line 33:  -- fig,ax=plt.subplots(figsize = (2,2))
# Line 33: Creates a matplotlib figure and axes with a size of 2x2.
# Line 34:  -- ov.pl.cellproportion(adata=adata,celltype_clusters='age(days)',
# Line 34: Creates a cell proportion plot, using 'age(days)' as the cell types.
# Line 35:  --                     groupby='clusters',groupby_li=['nIPC','Granule immature','Granule mature'],
# Line 35: Groups the plot by 'clusters', and specifies a list of clusters to show.
# Line 36:  --                      legend=True,ax=ax)
# Line 36: Displays the legend and draws on the specified axes.
# Line 38:  -- fig,ax=plt.subplots(figsize = (2,2))
# Line 38: Creates a matplotlib figure and axes with a size of 2x2.
# Line 39:  -- ov.pl.cellstackarea(adata=adata,celltype_clusters='age(days)',
# Line 39: Creates a stacked area plot of cell proportions, using 'age(days)' as the cell types.
# Line 40:  --                     groupby='clusters',groupby_li=['nIPC','Granule immature','Granule mature'],
# Line 40: Groups the plot by 'clusters' and specifies a list of clusters to show.
# Line 41:  --                      legend=True,ax=ax)
# Line 41: Displays the legend and draws on the specified axes.
# Line 43:  -- ov.pl.embedding_celltype(adata,figsize=(7,4),basis='X_umap',
# Line 43: Calls omicverse's embedding_celltype function to create a cell type specific plot.
# Line 44:  --                             celltype_key='clusters',
# Line 44: Specifies that the 'clusters' annotation should be used to delineate cell types.
# Line 45:  --                             title='            Cell type',
# Line 45: Sets the title of the plot to 'Cell type'.
# Line 46:  --                             celltype_range=(1,10),
# Line 46: Sets the range of cell type categories to display.
# Line 47:  --                             embedding_range=(4,10),)
# Line 47: Sets the range of the embedding coordinates to display.
# Line 49:  -- import matplotlib.pyplot as plt
# Line 49: Imports the matplotlib.pyplot module as plt.
# Line 50:  -- fig,ax=plt.subplots(figsize = (4,4))
# Line 50: Creates a matplotlib figure and axes with a size of 4x4.
# Line 52:  -- ov.pl.embedding(adata,
# Line 52: Calls omicverse's embedding plot function on the 'adata' object.
# Line 53:  --                 basis='X_umap',
# Line 53: Specifies 'X_umap' as the embedding basis for the plot.
# Line 54:  --                 color=['clusters'],
# Line 54: Colors the embedding plot based on the 'clusters' annotation.
# Line 55:  --                 frameon='small',
# Line 55: Sets a small frame for the plot.
# Line 56:  --                 show=False,
# Line 56: Prevents the plot from being shown immediately.
# Line 57:  --                 ax=ax)
# Line 57: Specifies the axes to draw the plot on.
# Line 59:  -- ov.pl.ConvexHull(adata,
# Line 59: Calls omicverse's ConvexHull function to add convex hull outlines to the embedding plot.
# Line 60:  --                 basis='X_umap',
# Line 60: Specifies 'X_umap' as the embedding basis.
# Line 61:  --                 cluster_key='clusters',
# Line 61: Specifies 'clusters' as the annotation key for identifying clusters.
# Line 62:  --                 hull_cluster='Granule mature',
# Line 62: Specifies that a convex hull should be drawn for the 'Granule mature' cluster.
# Line 63:  --                 ax=ax)
# Line 63: Specifies the axes to draw the convex hull on.
# Line 66:  -- import matplotlib.pyplot as plt
# Line 66: Imports the matplotlib.pyplot module as plt.
# Line 67:  -- fig,ax=plt.subplots(figsize = (4,4))
# Line 67: Creates a matplotlib figure and axes with a size of 4x4.
# Line 69:  -- ov.pl.embedding(adata,
# Line 69: Calls omicverse's embedding plot function on the 'adata' object.
# Line 70:  --                 basis='X_umap',
# Line 70: Specifies 'X_umap' as the embedding basis for the plot.
# Line 71:  --                 color=['clusters'],
# Line 71: Colors the embedding plot based on the 'clusters' annotation.
# Line 72:  --                 frameon='small',
# Line 72: Sets a small frame for the plot.
# Line 73:  --                 show=False,
# Line 73: Prevents the plot from being shown immediately.
# Line 74:  --                 ax=ax)
# Line 74: Specifies the axes to draw the plot on.
# Line 76:  -- ov.pl.contour(ax=ax,adata=adata,groupby='clusters',clusters=['Granule immature','Granule mature'],
# Line 76: Adds contour lines to the embedding plot using omicverse.
# Line 77:  --        basis='X_umap',contour_threshold=0.1,colors='#000000',
# Line 77: Specifies the basis, threshold, color, and line style for the contour lines.
# Line 78:  --         linestyles='dashed',)
# Line 78: Specifies dashed line style for the contour lines.
# Line 81:  -- from matplotlib import patheffects
# Line 81: Imports the patheffects module from matplotlib.
# Line 82:  -- import matplotlib.pyplot as plt
# Line 82: Imports the matplotlib.pyplot module as plt.
# Line 83:  -- fig, ax = plt.subplots(figsize=(4,4))
# Line 83: Creates a matplotlib figure and an axes object with a specified size of 4x4.
# Line 85:  -- ov.pl.embedding(adata,
# Line 85: Calls omicverse's embedding plot function on the 'adata' object.
# Line 86:  --                   basis='X_umap',
# Line 86: Specifies 'X_umap' as the embedding basis for the plot.
# Line 87:  --                   color=['clusters'],
# Line 87: Colors the embedding plot based on the 'clusters' annotation.
# Line 88:  --                    show=False, legend_loc=None, add_outline=False, 
# Line 88: Prevents showing the plot, turns off the legend and outline.
# Line 89:  --                    frameon='small',legend_fontoutline=2,ax=ax
# Line 89: Sets a small frame, sets the legend font outline width, and draws to specified axes.
# Line 90:  --                  )
# Line 92:  -- ov.pl.embedding_adjust(
# Line 92: Calls omicverse's embedding_adjust function to adjust the embedding plot.
# Line 93:  --     adata,
# Line 94:  --     groupby='clusters',
# Line 94: Specifies that the adjustments are based on the 'clusters' annotation.
# Line 95:  --     exclude=("OL",),  
# Line 95: Specifies that the 'OL' cluster should be excluded from adjustments.
# Line 96:  --     basis='X_umap',
# Line 96: Specifies 'X_umap' as the embedding basis.
# Line 97:  --     ax=ax,
# Line 97: Specifies the axes to draw the adjustments on.
# Line 98:  --     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
# Line 98: Sets the style for the arrows in the adjustment.
# Line 99:  --     text_kwargs=dict(fontsize=12 ,weight='bold',
# Line 99: Sets the style for the text in the adjustment.
# Line 100:  --                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ),
# Line 100: Adds a white outline to the text.
# Line 102:  -- ov.pl.embedding_density(adata,
# Line 102: Calls omicverse's embedding_density function to create a density map on the embedding.
# Line 103:  --                  basis='X_umap',
# Line 103: Specifies 'X_umap' as the embedding basis.
# Line 104:  --                  groupby='clusters',
# Line 104: Specifies that the density map is based on the 'clusters' annotation.
# Line 105:  --                  target_clusters='Granule mature',
# Line 105: Specifies that the density map is only for the 'Granule mature' cluster.
# Line 106:  --                  frameon='small',
# Line 106: Sets a small frame for the plot.
# Line 107:  --                 show=False,cmap='RdBu_r',alpha=0.8)
# Line 107: Prevents the plot from being shown immediately and sets colormap and transparency.
# Line 109:  -- ov.single.geneset_aucell(adata,
# Line 109: Calls omicverse's geneset_aucell function to compute AUC scores for a gene set.
# Line 110:  --                             geneset_name='Sox',
# Line 110: Specifies the name of the gene set as 'Sox'.
# Line 111:  --                             geneset=['Sox17', 'Sox4', 'Sox7', 'Sox18', 'Sox5'])
# Line 111: Specifies the genes included in the 'Sox' gene set.
# Line 113:  -- ov.pl.embedding(adata,
# Line 113: Calls omicverse's embedding plot function.
# Line 114:  --                 basis='X_umap',
# Line 114: Specifies 'X_umap' as the embedding basis.
# Line 115:  --                 color=['Sox4'],
# Line 115: Colors the embedding plot based on expression of the gene 'Sox4'.
# Line 116:  --                 frameon='small',
# Line 116: Sets a small frame for the plot.
# Line 117:  --                 show=False,)
# Line 117: Prevents the plot from being shown immediately.
# Line 119:  -- ov.pl.violin(adata,keys='Sox4',groupby='clusters',figsize=(6,3))
# Line 119: Creates a violin plot using omicverse, visualizing 'Sox4' expression grouped by 'clusters' with a 6x3 size.
# Line 121:  -- fig, ax = plt.subplots(figsize=(6,2))
# Line 121: Creates a matplotlib figure and axes with a size of 6x2.
# Line 122:  -- ov.pl.bardotplot(adata,groupby='clusters',color='Sox_aucell',figsize=(6,2),
# Line 122: Creates a bar dot plot using omicverse, grouped by clusters and coloured by Sox_aucell scores with a 6x2 size.
# Line 123:  --            ax=ax,
# Line 123: Sets the axes to draw the plot on.
# Line 124:  --           ylabel='Expression',
# Line 124: Sets the y-axis label to "Expression".
# Line 125:  --            bar_kwargs={'alpha':0.5,'linewidth':2,'width':0.6,'capsize':4},
# Line 125: Sets the styling parameters for the bars.
# Line 126:  --            scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'})
# Line 126: Sets the styling parameters for the dots.
# Line 128:  -- ov.pl.add_palue(ax,line_x1=3,line_x2=4,line_y=0.1,
# Line 128: Adds a p-value annotation to the plot using omicverse.
# Line 129:  --           text_y=0.02,
# Line 129: Sets the y-position for the p-value text.
# Line 130:  --           text='$p={}$'.format(round(0.001,3)),
# Line 130: Sets the text for the p-value annotation.
# Line 131:  --           fontsize=11,fontcolor='#000000',
# Line 131: Sets the font size and color for the annotation text.
# Line 132:  --              horizontalalignment='center',)
# Line 132: Sets the horizontal alignment of the annotation text.
# Line 134:  -- fig, ax = plt.subplots(figsize=(6,2))
# Line 134: Creates a matplotlib figure and axes with a size of 6x2.
# Line 135:  -- ov.pl.bardotplot(adata,groupby='clusters',color='Sox17',figsize=(6,2),
# Line 135: Creates a bar dot plot using omicverse, grouped by clusters and coloured by Sox17 expression.
# Line 136:  --            ax=ax,
# Line 136: Sets the axes to draw the plot on.
# Line 137:  --           ylabel='Expression',xlabel='Cell Type',
# Line 137: Sets the y-axis label to "Expression" and the x-axis label to "Cell Type".
# Line 138:  --            bar_kwargs={'alpha':0.5,'linewidth':2,'width':0.6,'capsize':4},
# Line 138: Sets the styling parameters for the bars.
# Line 139:  --            scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'})
# Line 139: Sets the styling parameters for the dots.
# Line 141:  -- ov.pl.add_palue(ax,line_x1=3,line_x2=4,line_y=2,
# Line 141: Adds a p-value annotation to the plot using omicverse.
# Line 142:  --           text_y=0.2,
# Line 142: Sets the y-position for the p-value text.
# Line 143:  --           text='$p={}$'.format(round(0.001,3)),
# Line 143: Sets the text for the p-value annotation.
# Line 144:  --           fontsize=11,fontcolor='#000000',
# Line 144: Sets the font size and color for the annotation text.
# Line 145:  --              horizontalalignment='center',)
# Line 145: Sets the horizontal alignment of the annotation text.
# Line 147:  -- import pandas as pd
# Line 147: Imports the pandas library and aliases it as pd.
# Line 148:  -- import seaborn as sns
# Line 148: Imports the seaborn library and aliases it as sns.
# Line 150:  -- ov.pl.single_group_boxplot(adata,groupby='clusters',
# Line 150: Creates a boxplot using omicverse, grouped by clusters.
# Line 151:  --              color='Sox_aucell',
# Line 151: Sets the color of the boxes based on 'Sox_aucell' scores.
# Line 152:  --              type_color_dict=dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
# Line 152: Creates a color dictionary using existing cluster colors.
# Line 153:  --              x_ticks_plot=True,
# Line 153: Enables plotting of x-ticks.
# Line 154:  --              figsize=(5,2),
# Line 154: Sets the size of the plot to 5x2.
# Line 155:  --              kruskal_test=True,
# Line 155: Performs a Kruskal-Wallis test for statistical significance.
# Line 156:  --              ylabel='Sox_aucell',
# Line 156: Sets the y-axis label to "Sox_aucell".
# Line 157:  --              legend_plot=False,
# Line 157: Disables legend plotting.
# Line 158:  --              bbox_to_anchor=(1,1),
# Line 158: Sets the bounding box for the legend if it were to be plotted.
# Line 159:  --              title='Expression',
# Line 159: Sets the title of the plot to "Expression".
# Line 160:  --              scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'},
# Line 160: Sets the styling parameters for the scatter dots.
# Line 161:  --              point_number=15,
# Line 161: Sets the number of points to plot.
# Line 162:  --              sort=False,
# Line 162: Disables sorting of the plot.
# Line 163:  --              save=False,
# Line 163: Disables saving the plot automatically.
# Line 164:  --              )
# Line 165:  -- plt.grid(False)
# Line 165: Disables the grid on the plot.
# Line 166:  -- plt.xticks(rotation=90,fontsize=12)
# Line 166: Rotates the x-axis ticks by 90 degrees and sets font size to 12.
# Line 168:  -- import pandas as pd
# Line 168: Imports the pandas library as pd.
# Line 169:  -- marker_genes_dict = {
# Line 169: Defines a dictionary holding marker genes for various cell types.
# Line 170:  --     'Sox':['Sox4', 'Sox7', 'Sox18', 'Sox5'],
# Line 170: Specifies the 'Sox' gene set.
# Line 172:  -- color_dict = {'Sox':'#EFF3D8',}
# Line 172: Defines a dictionary holding colors for marker gene sets.
# Line 174:  -- gene_color_dict = {}
# Line 174: Initializes an empty dictionary for gene colors.
# Line 175:  -- gene_color_dict_black = {}
# Line 175: Initializes an empty dictionary for gene colors (black).
# Line 176:  -- for cell_type, genes in marker_genes_dict.items():
# Line 176: Iterates over the marker gene sets in the dictionary.
# Line 177:  --     cell_type_color = color_dict.get(cell_type)
# Line 177: Gets the color for the current cell type.
# Line 178:  --     for gene in genes:
# Line 178: Iterates over the genes in the current cell type.
# Line 179:  --         gene_color_dict[gene] = cell_type_color
# Line 179: Assigns the cell type color to the current gene.
# Line 180:  --         gene_color_dict_black[gene] = '#000000'
# Line 180: Assigns black as color to current gene in black color dict.
# Line 182:  -- cm = ov.pl.complexheatmap(adata,
# Line 182: Creates a complex heatmap using omicverse.
# Line 183:  --                        groupby ='clusters',
# Line 183: Groups the heatmap by 'clusters'.
# Line 184:  --                        figsize =(5,2),
# Line 184: Sets the size of the heatmap to 5x2.
# Line 185:  --                        layer = None,
# Line 185: Specifies no layer.
# Line 186:  --                        use_raw = False,
# Line 186: Specifies that the raw data is not to be used.
# Line 187:  --                        standard_scale = 'var',
# Line 187: Specifies standard scaling by variance.
# Line 188:  --                        col_color_bars = dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
# Line 188: Sets the column color bars based on cluster colors.
# Line 189:  --                        col_color_labels = dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
# Line 189: Sets the column color labels based on cluster colors.
# Line 190:  --                        left_color_bars = color_dict,
# Line 190: Sets the left color bars.
# Line 191:  --                        left_color_labels = None,
# Line 191: Specifies no left color labels.
# Line 192:  --                        right_color_bars = color_dict,
# Line 192: Sets the right color bars.
# Line 193:  --                        right_color_labels = gene_color_dict_black,
# Line 193: Sets the right color labels.
# Line 194:  --                        marker_genes_dict = marker_genes_dict,
# Line 194: Specifies the marker genes dictionary.
# Line 195:  --                        cmap = 'coolwarm', #parula,jet
# Line 195: Sets the colormap for the heatmap.
# Line 196:  --                        legend_gap = 15,
# Line 196: Sets the gap between the heatmap and the legend.
# Line 197:  --                        legend_hpad = 0,
# Line 197: Sets the horizontal padding of the legend.
# Line 198:  --                        left_add_text = True,
# Line 198: Enables the addition of left-side text.
# Line 199:  --                        col_split_gap = 2,
# Line 199: Sets the column split gap.
# Line 200:  --                        row_split_gap = 1,
# Line 200: Sets the row split gap.
# Line 201:  --                        col_height = 6,
# Line 201: Sets the column height of the heatmap.
# Line 202:  --                        left_height = 4,
# Line 202: Sets the left side height.
# Line 203:  --                        right_height = 6,
# Line 203: Sets the right side height.
# Line 204:  --                        col_split = None,
# Line 204: Specifies no column split.
# Line 205:  --                        row_cluster = False,
# Line 205: Disables row clustering.
# Line 206:  --                        col_cluster = False,
# Line 206: Disables column clustering.
# Line 207:  --                        value_name='Gene',
# Line 207: Sets the value name for the heatmap.
# Line 208:  --                        xlabel = "Expression of selected genes",
# Line 208: Sets the x-axis label.
# Line 209:  --                        label = 'Gene Expression',
# Line 209: Sets the label for the heatmap.
# Line 210:  --                        save = True,
# Line 210: Enables saving of the heatmap.
# Line 211:  --                        show = False,
# Line 211: Prevents the heatmap from showing immediately.
# Line 212:  --                        legend = False,
# Line 212: Prevents a legend from being plotted.
# Line 213:  --                        plot_legend = False,
# Line 213: Disables plotting of the legend.
# Line 215:  --                             )
# Line 217:  -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 217: Preprocesses the AnnData object using omicverse with specified mode and number of highly variable genes.
# Line 219:  -- marker_genes_dict = {'Granule immature': ['Sepw1', 'Camk2b', 'Cnih2'],
# Line 219: Defines a dictionary of marker genes for specific cell types.
# Line 220:  --  'Radial Glia-like': ['Dbi', 'Fabp7', 'Aldoc'],
# Line 220: Specifies marker genes for the 'Radial Glia-like' cell type.
# Line 221:  --  'Granule mature': ['Malat1', 'Rasl10a', 'Ppp3ca'],
# Line 221: Specifies marker genes for the 'Granule mature' cell type.
# Line 222:  --  'Neuroblast': ['Igfbpl1', 'Tubb2b', 'Tubb5'],
# Line 222: Specifies marker genes for the 'Neuroblast' cell type.
# Line 223:  --  'Microglia': ['Lgmn', 'C1qa', 'C1qb'],
# Line 223: Specifies marker genes for the 'Microglia' cell type.
# Line 224:  --  'Cajal Retzius': ['Diablo', 'Ramp1', 'Stmn1'],
# Line 224: Specifies marker genes for the 'Cajal Retzius' cell type.
# Line 225:  --  'OPC': ['Olig1', 'C1ql1', 'Pllp'],
# Line 225: Specifies marker genes for the 'OPC' cell type.
# Line 226:  --  'Cck-Tox': ['Tshz2', 'Cck', 'Nap1l5'],
# Line 226: Specifies marker genes for the 'Cck-Tox' cell type.
# Line 227:  --  'GABA': ['Gad2', 'Gad1', 'Snhg11'],
# Line 227: Specifies marker genes for the 'GABA' cell type.
# Line 228:  --  'Endothelial': ['Sparc', 'Myl12a', 'Itm2a'],
# Line 228: Specifies marker genes for the 'Endothelial' cell type.
# Line 229:  --  'Astrocytes': ['Apoe',  'Atp1a2'],
# Line 229: Specifies marker genes for the 'Astrocytes' cell type.
# Line 230:  --  'OL': ['Plp1', 'Mog', 'Mag'],
# Line 230: Specifies marker genes for the 'OL' cell type.
# Line 231:  --  'Mossy': ['Arhgdig', 'Camk4'],
# Line 231: Specifies marker genes for the 'Mossy' cell type.
# Line 232:  --  'nIPC': ['Hmgn2', 'Ptma', 'H2afz']}
# Line 232: Specifies marker genes for the 'nIPC' cell type.
# Line 234:  -- ov.pl.marker_heatmap(
# Line 234: Creates a marker heatmap using omicverse.
# Line 235:  --     adata,
# Line 236:  --     marker_genes_dict,
# Line 236: Specifies the marker genes dictionary to use for the heatmap.
# Line 237:  --     groupby='clusters',
# Line 237: Groups the heatmap by the 'clusters' annotation.
# Line 238:  --     color_map="RdBu_r",
# Line 238: Sets the color map to 'RdBu_r'.
# Line 239:  --     use_raw=False,
# Line 239: Specifies that raw data should not be used.
# Line 240:  --     standard_scale="var",
# Line 240: Specifies standard scaling by variance.
# Line 241:  --     expression_cutoff=0.0,
# Line 241: Sets the minimum expression cutoff.
# Line 242:  --     fontsize=12,
# Line 242: Sets the font size for the plot.
# Line 243:  --     bbox_to_anchor=(7, -2),
# Line 243: Sets the bounding box anchor