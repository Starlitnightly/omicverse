```
# Line 1:  import omicverse as ov -- import omicverse as ov
# Line 2:  import scanpy as sc -- import scanpy as sc
# Line 4:  ov.plot_set() -- ov.plot_set()
# Line 6:  adata = ov.read('data/DentateGyrus/10X43_1.h5ad') -- adata = ov.read('data/DentateGyrus/10X43_1.h5ad')
# Line 7:  adata -- adata
# Line 9:  fb=ov.pl.ForbiddenCity() -- fb=ov.pl.ForbiddenCity()
# Line 11: from IPython.display import HTML -- from IPython.display import HTML
# Line 12: HTML(fb.visual_color(loc_range=(0,384), -- HTML(fb.visual_color(loc_range=(0,384),
# Line 13:                     num_per_row=24)) --                     num_per_row=24))
# Line 15: fb.get_color(name='凝夜紫') -- fb.get_color(name='凝夜紫')
# Line 17: import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 18: fig, axes = plt.subplots(1,3,figsize=(9,3)) -- fig, axes = plt.subplots(1,3,figsize=(9,3))
# Line 19: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 20:                    basis='X_umap', --                    basis='X_umap',
# Line 21:                     frameon='small', --                     frameon='small',
# Line 22:                    color=["clusters"], --                    color=["clusters"],
# Line 23:                    palette=fb.red[:], --                    palette=fb.red[:],
# Line 24:                    ncols=3, --                    ncols=3,
# Line 25:                 show=False, --                 show=False,
# Line 26:                 legend_loc=None, --                 legend_loc=None,
# Line 27:                     ax=axes[0]) --                     ax=axes[0])
# Line 29: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 30:                    basis='X_umap', --                    basis='X_umap',
# Line 31:                     frameon='small', --                     frameon='small',
# Line 32:                    color=["clusters"], --                    color=["clusters"],
# Line 33:                    palette=fb.pink1[:], --                    palette=fb.pink1[:],
# Line 34:                    ncols=3,show=False, --                    ncols=3,show=False,
# Line 35:                 legend_loc=None, --                 legend_loc=None,
# Line 36:                     ax=axes[1]) --                     ax=axes[1])
# Line 38: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 39:                    basis='X_umap', --                    basis='X_umap',
# Line 40:                     frameon='small', --                     frameon='small',
# Line 41:                    color=["clusters"], --                    color=["clusters"],
# Line 42:                    palette=fb.red1[:4]+fb.blue1, --                    palette=fb.red1[:4]+fb.blue1,
# Line 43:                    ncols=3,show=False, --                    ncols=3,show=False,
# Line 44:                     ax=axes[2]) --                     ax=axes[2])
# Line 48: color_dict={'Astrocytes': '#e40414', -- color_dict={'Astrocytes': '#e40414',
# Line 49:  'Cajal Retzius': '#ec5414', --  'Cajal Retzius': '#ec5414',
# Line 50:  'Cck-Tox': '#ec4c2c', --  'Cck-Tox': '#ec4c2c',
# Line 51:  'Endothelial': '#d42c24', --  'Endothelial': '#d42c24',
# Line 52:  'GABA': '#2c5ca4', --  'GABA': '#2c5ca4',
# Line 53:  'Granule immature': '#acd4ec', --  'Granule immature': '#acd4ec',
# Line 54:  'Granule mature': '#a4bcdc', --  'Granule mature': '#a4bcdc',
# Line 55:  'Microglia': '#8caccc', --  'Microglia': '#8caccc',
# Line 56:  'Mossy': '#8cacdc', --  'Mossy': '#8cacdc',
# Line 57:  'Neuroblast': '#6c9cc4', --  'Neuroblast': '#6c9cc4',
# Line 58:  'OL': '#6c94cc', --  'OL': '#6c94cc',
# Line 59:  'OPC': '#5c74bc', --  'OPC': '#5c74bc',
# Line 60:  'Radial Glia-like': '#4c94c4', --  'Radial Glia-like': '#4c94c4',
# Line 61:  'nIPC': '#3474ac'} --  'nIPC': '#3474ac'}
# Line 63: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 64:                    basis='X_umap', --                    basis='X_umap',
# Line 65:                     frameon='small', --                     frameon='small',
# Line 66:                    color=["clusters"], --                    color=["clusters"],
# Line 67:                    palette=color_dict, --                    palette=color_dict,
# Line 68:                    ncols=3,show=False, --                    ncols=3,show=False,
# Line 69:                     ) --                     )
# Line 72: colors=[ -- colors=[
# Line 73:     fb.get_color_rgb('群青'), --     fb.get_color_rgb('群青'),
# Line 74:     fb.get_color_rgb('半见'), --     fb.get_color_rgb('半见'),
# Line 75:     fb.get_color_rgb('丹罽'), --     fb.get_color_rgb('丹罽'),
# Line 76: ] -- ]
# Line 77: fb.get_cmap_seg(colors) -- fb.get_cmap_seg(colors)
# Line 79: colors=[ -- colors=[
# Line 80:     fb.get_color_rgb('群青'), --     fb.get_color_rgb('群青'),
# Line 81:     fb.get_color_rgb('山矾'), --     fb.get_color_rgb('山矾'),
# Line 82:     fb.get_color_rgb('丹罽'), --     fb.get_color_rgb('丹罽'),
# Line 83: ] -- ]
# Line 84: fb.get_cmap_seg(colors) -- fb.get_cmap_seg(colors)
# Line 86: colors=[ -- colors=[
# Line 87:     fb.get_color_rgb('山矾'), --     fb.get_color_rgb('山矾'),
# Line 88:     fb.get_color_rgb('丹罽'), --     fb.get_color_rgb('丹罽'),
# Line 89: ] -- ]
# Line 90: fb.get_cmap_seg(colors) -- fb.get_cmap_seg(colors)
# Line 92: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 93:                 basis='X_umap', --                 basis='X_umap',
# Line 94:                 frameon='small', --                 frameon='small',
# Line 95:                 color=["Sox7"], --                 color=["Sox7"],
# Line 96:                 cmap=fb.get_cmap_seg(colors), --                 cmap=fb.get_cmap_seg(colors),
# Line 97:                 ncols=3,show=False, --                 ncols=3,show=False,
# Line 98:                 #vmin=-1,vmax=1 --                 #vmin=-1,vmax=1
# Line 99:                 ) --                 )
```