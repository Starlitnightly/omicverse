```
# Line 1:  import omicverse as ov -- import omicverse as ov
# Line 2:  import scanpy as sc -- import scanpy as sc
# Line 3:  import infercnvpy as cnv -- import infercnvpy as cnv
# Line 4:  import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 5:  import os -- import os
# Line 7:  sc.settings.verbosity = 3 -- sc.settings.verbosity = 3
# Line 8:  sc.settings.set_figure_params(dpi=80, facecolor='white') -- sc.settings.set_figure_params(dpi=80, facecolor='white')
# Line 11:  adata = cnv.datasets.maynard2020_3k() -- adata = cnv.datasets.maynard2020_3k()
# Line 13:  ov.utils.get_gene_annotation( -- ov.utils.get_gene_annotation(
# Line 14:      adata, gtf="gencode.v43.basic.annotation.gtf.gz", --     adata, gtf="gencode.v43.basic.annotation.gtf.gz",
# Line 15:      gtf_by="gene_name" --     gtf_by="gene_name"
# Line 16:  ) -- )
# Line 19:  adata=adata[:,~adata.var['chrom'].isnull()] -- adata=adata[:,~adata.var['chrom'].isnull()]
# Line 20:  adata.var['chromosome']=adata.var['chrom'] -- adata.var['chromosome']=adata.var['chrom']
# Line 21:  adata.var['start']=adata.var['chromStart'] -- adata.var['start']=adata.var['chromStart']
# Line 22:  adata.var['end']=adata.var['chromEnd'] -- adata.var['end']=adata.var['chromEnd']
# Line 23:  adata.var['ensg']=adata.var['gene_id'] -- adata.var['ensg']=adata.var['gene_id']
# Line 24:  adata.var.loc[:, ["ensg", "chromosome", "start", "end"]].head() -- adata.var.loc[:, ["ensg", "chromosome", "start", "end"]].head()
# Line 26:  adata -- adata
# Line 29:  cnv.tl.infercnv( -- cnv.tl.infercnv(
# Line 30:      adata, --     adata,
# Line 31:      reference_key="cell_type", --     reference_key="cell_type",
# Line 32:      reference_cat=[ --     reference_cat=[
# Line 33:          "B cell", --         "B cell",
# Line 34:          "Macrophage", --         "Macrophage",
# Line 35:          "Mast cell", --         "Mast cell",
# Line 36:          "Monocyte", --         "Monocyte",
# Line 37:          "NK cell", --         "NK cell",
# Line 38:          "Plasma cell", --         "Plasma cell",
# Line 39:          "T cell CD4", --         "T cell CD4",
# Line 40:          "T cell CD8", --         "T cell CD8",
# Line 41:          "T cell regulatory", --         "T cell regulatory",
# Line 42:          "mDC", --         "mDC",
# Line 43:          "pDC", --         "pDC",
# Line 44:      ], --     ],
# Line 45:      window_size=250, --     window_size=250,
# Line 46:  ) -- )
# Line 47:  cnv.tl.pca(adata) -- cnv.tl.pca(adata)
# Line 48:  cnv.pp.neighbors(adata) -- cnv.pp.neighbors(adata)
# Line 49:  cnv.tl.leiden(adata) -- cnv.tl.leiden(adata)
# Line 50:  cnv.tl.umap(adata) -- cnv.tl.umap(adata)
# Line 51:  cnv.tl.cnv_score(adata) -- cnv.tl.cnv_score(adata)
# Line 53:  sc.pl.umap(adata, color="cnv_score", show=False) -- sc.pl.umap(adata, color="cnv_score", show=False)
# Line 55:  adata.obs["cnv_status"] = "normal" -- adata.obs["cnv_status"] = "normal"
# Line 56:  adata.obs.loc[ -- adata.obs.loc[
# Line 57:      adata.obs["cnv_score"]>0.03, "cnv_status" --     adata.obs["cnv_score"]>0.03, "cnv_status"
# Line 58:  ] = "tumor" -- ] = "tumor"
# Line 60:  sc.pl.umap(adata, color="cnv_status", show=False) -- sc.pl.umap(adata, color="cnv_status", show=False)
# Line 62:  tumor=adata[adata.obs['cnv_status']=='tumor'] -- tumor=adata[adata.obs['cnv_status']=='tumor']
# Line 63:  tumor.X.max() -- tumor.X.max()
# Line 65:  adata=tumor -- adata=tumor
# Line 66:  print('Preprocessing...') -- print('Preprocessing...')
# Line 67:  sc.pp.filter_cells(adata, min_genes=200) -- sc.pp.filter_cells(adata, min_genes=200)
# Line 68:  sc.pp.filter_genes(adata, min_cells=3) -- sc.pp.filter_genes(adata, min_cells=3)
# Line 69:  adata.var['mt'] = adata.var_names.str.startswith('MT-') -- adata.var['mt'] = adata.var_names.str.startswith('MT-')
# Line 70:  sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) -- sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# Line 71:  if not (adata.obs.pct_counts_mt == 0).all(): -- if not (adata.obs.pct_counts_mt == 0).all():
# Line 72:      adata = adata[adata.obs.pct_counts_mt < 30, :] --     adata = adata[adata.obs.pct_counts_mt < 30, :]
# Line 74:  adata.raw = adata.copy() -- adata.raw = adata.copy()
# Line 76:  sc.pp.highly_variable_genes(adata) -- sc.pp.highly_variable_genes(adata)
# Line 77:  adata = adata[:, adata.var.highly_variable] -- adata = adata[:, adata.var.highly_variable]
# Line 78:  sc.pp.scale(adata) -- sc.pp.scale(adata)
# Line 79:  sc.tl.pca(adata, svd_solver='arpack') -- sc.tl.pca(adata, svd_solver='arpack')
# Line 81:  sc.pp.neighbors(adata, n_pcs=20) -- sc.pp.neighbors(adata, n_pcs=20)
# Line 82:  sc.tl.umap(adata) -- sc.tl.umap(adata)
# Line 84:  ov.utils.download_GDSC_data() -- ov.utils.download_GDSC_data()
# Line 85:  ov.utils.download_CaDRReS_model() -- ov.utils.download_CaDRReS_model()
# Line 87:  adata, res,plot_df = ov.single.autoResolution(adata,cpus=4) -- adata, res,plot_df = ov.single.autoResolution(adata,cpus=4)
# Line 89:  results_file = os.path.join('./', 'scanpyobj.h5ad') -- results_file = os.path.join('./', 'scanpyobj.h5ad')
# Line 90:  adata.write(results_file) -- adata.write(results_file)
# Line 92:  results_file = os.path.join('./', 'scanpyobj.h5ad') -- results_file = os.path.join('./', 'scanpyobj.h5ad')
# Line 93:  adata=sc.read(results_file) -- adata=sc.read(results_file)
# Line 96:  !git clone https://github.com/CSB5/CaDRReS-Sc -- !git clone https://github.com/CSB5/CaDRReS-Sc
# Line 98:  import ov -- import ov
# Line 99:  job=ov.single.Drug_Response(adata,scriptpath='CaDRReS-Sc', -- job=ov.single.Drug_Response(adata,scriptpath='CaDRReS-Sc',
# Line 100:                                modelpath='models/', --                                modelpath='models/',
# Line 101:                                output='result') --                                output='result')
```