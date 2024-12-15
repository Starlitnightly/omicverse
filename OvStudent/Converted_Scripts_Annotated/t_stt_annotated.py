```
# Line 1:  Import the omicverse library -- import omicverse as ov
# Line 3:  Import the scvelo library as scv -- import scvelo as scv
# Line 4:  Import the scanpy library as sc -- import scanpy as sc
# Line 5: Set plot parameters using omicverse -- ov.plot_set()
# Line 7: Read an h5ad file into an AnnData object named adata -- adata = sc.read_h5ad('mouse_brain.h5ad')
# Line 8: Show the AnnData object -- adata
# Line 10: Create an STT object with spatial and region information -- STT_obj=ov.space.STT(adata,spatial_loc='xy_loc',region='Region')
# Line 12: Estimate stages for the STT object -- STT_obj.stage_estimate()
# Line 14: Train the STT model with specific parameters -- STT_obj.train(n_states = 9, n_iter = 15, weight_connectivities = 0.5, 
# Line 15: Continue training the STT model with specific parameters --            n_neighbors = 50,thresh_ms_gene = 0.2, spa_weight =0.3)
# Line 17: Plot embedding colored by attractors -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 18: Continue plotting embedding colored by attractors --                 color=["attractor"],frameon='small',
# Line 19: Continue plotting embedding colored by attractors --                palette=ov.pl.sc_color[11:])
# Line 21: Plot embedding colored by region -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 22: Continue plotting embedding colored by region --                 color=["Region"],frameon='small',
# Line 23: Continue plotting embedding colored by region --                )
# Line 25: Prepare pathway gene sets from a file -- pathway_dict=ov.utils.geneset_prepare('genesets/KEGG_2019_Mouse.txt',organism='Mouse')
# Line 27: Compute pathway scores -- STT_obj.compute_pathway(pathway_dict)
# Line 29: Create a plot of pathway scores -- fig = STT_obj.plot_pathway(figsize = (10,8),size = 100,fontsize = 12)
# Line 30: Loop through the axes of the pathway plot -- for ax in fig.axes:
# Line 31: Set x-axis label for pathway plot with specified font size --     ax.set_xlabel('Embedding 1', fontsize=20)  # Adjust font size as needed
# Line 32: Set y-axis label for pathway plot with specified font size --     ax.set_ylabel('Embedding 2', fontsize=20)  # Adjust font size as needed
# Line 33: Show the pathway plot -- fig.show()
# Line 35: Import the matplotlib.pyplot module -- import matplotlib.pyplot as plt
# Line 36: Create a figure and axes for a single plot -- fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# Line 37: Plot tensor pathway for 'Wnt signaling pathway' -- STT_obj.plot_tensor_pathway(pathway_name = 'Wnt signaling pathway',basis = 'xy_loc',
# Line 38: Continue plotting tensor pathway --                            ax=ax)
# Line 40: Create another figure and axes for a single plot -- fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# Line 41: Plot tensor pathway for 'TGF-beta signaling pathway' -- STT_obj.plot_tensor_pathway( 'TGF-beta signaling pathway',basis = 'xy_loc',
# Line 42: Continue plotting tensor pathway --                            ax=ax)
# Line 44: Plot tensor for attractors with filter and density parameters -- STT_obj.plot_tensor(list_attractor = [1,3,5,6],
# Line 45: Continue plotting tensor for attractors with filter and density parameters --                 filter_cells = True, member_thresh = 0.1, density = 1)
# Line 48: Construct landscape for STT object using coordinate key -- STT_obj.construct_landscape(coord_key = 'X_xy_loc')
# Line 50: Plot embedding colored by attractors and regions on the transformed coordinates -- sc.pl.embedding(adata, color = ['attractor', 'Region'],basis= 'trans_coord')
# Line 52: Infer lineage information with specific parameters -- STT_obj.infer_lineage(si=3,sf=4, method = 'MPPT',flux_fraction=0.8,color_palette_name = 'tab10',size_point = 8,
# Line 53: Continue inferring lineage with specific parameters --                    size_text=12)
# Line 55: Create a sankey diagram -- fig = STT_obj.plot_sankey(adata.obs['attractor'].tolist(),adata.obs['Region'].tolist())
# Line 60: Write AnnData object to a file -- STT_obj.adata.write('data/mouse_brain_adata.h5ad')
# Line 61: Write aggregated AnnData object to a file -- STT_obj.adata_aggr.write('data/mouse_brain_adata_aggr.h5ad')
# Line 63: Read AnnData object from a file -- adata=ov.read('data/mouse_brain_adata.h5ad')
# Line 64: Read aggregated AnnData object from a file -- adata_aggr=ov.read('data/mouse_brain_adata_aggr.h5ad')
# Line 66: Create an STT object with spatial and region information, loading from adata -- STT_obj=ov.space.STT(adata,spatial_loc='xy_loc',region='Region')
# Line 67: Load the STT object with stored data -- STT_obj.load(adata,adata_aggr)
# Line 69: Sort the variable r2_test values -- adata.var['r2_test'].sort_values(ascending=False)
# Line 71: Plot the top genes -- STT_obj.plot_top_genes(top_genes = 6, ncols = 2, figsize = (8,8),)
# Line 73: Import the matplotlib.pyplot module -- import matplotlib.pyplot as plt
# Line 74: Create a figure and axes for a subplot -- fig, axes = plt.subplots(1, 4, figsize=(12, 3))
# Line 75: Plot embedding for 'Sim1' expression using Ms layer -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 76: Continue plotting embedding for 'Sim1' expression using Ms layer --                 color=["Sim1"],frameon='small',
# Line 77: Continue plotting embedding for 'Sim1' expression using Ms layer --                 title='Sim1:Ms',show=False,
# Line 78: Continue plotting embedding for 'Sim1' expression using Ms layer --                 layer='Ms',cmap='RdBu_r',ax=axes[0]
# Line 79: Continue plotting embedding for 'Sim1' expression using Ms layer --                )
# Line 80: Plot embedding for 'Sim1' expression using Mu layer -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 81: Continue plotting embedding for 'Sim1' expression using Mu layer --                 color=["Sim1"],frameon='small',
# Line 82: Continue plotting embedding for 'Sim1' expression using Mu layer --                 title='Sim1:Mu',show=False,
# Line 83: Continue plotting embedding for 'Sim1' expression using Mu layer --                 layer='Mu',cmap='RdBu_r',ax=axes[1]
# Line 84: Continue plotting embedding for 'Sim1' expression using Mu layer --                )
# Line 85: Plot embedding for 'Sim1' expression using velocity layer -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 86: Continue plotting embedding for 'Sim1' expression using velocity layer --                 color=["Sim1"],frameon='small',
# Line 87: Continue plotting embedding for 'Sim1' expression using velocity layer --                 title='Sim1:Velo',show=False,
# Line 88: Continue plotting embedding for 'Sim1' expression using velocity layer --                 layer='velo',cmap='RdBu_r',ax=axes[2]
# Line 89: Continue plotting embedding for 'Sim1' expression using velocity layer --                )
# Line 90: Plot embedding for 'Sim1' expression using expression layer -- ov.pl.embedding(adata, basis="xy_loc", 
# Line 91: Continue plotting embedding for 'Sim1' expression using expression layer --                 color=["Sim1"],frameon='small',
# Line 92: Continue plotting embedding for 'Sim1' expression using expression layer --                 title='Sim1:exp',show=False,
# Line 94: Continue plotting embedding for 'Sim1' expression using expression layer --                 cmap='RdBu_r',ax=axes[3]
# Line 95: Continue plotting embedding for 'Sim1' expression using expression layer --                )
# Line 96: Adjust the layout to fit the subplots -- plt.tight_layout()
```
