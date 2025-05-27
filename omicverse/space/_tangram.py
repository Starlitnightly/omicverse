"""Module providing a encapsulation of tangram."""
from typing import Any
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from .._settings import add_reference

tg_install=False

class Tangram(object):
    """Class representing the object of Tangram."""
    def check_tangram(self):
        """
        Check if tangram have been installed.
        """
        global tg_install
        try:
            import tangram as tg
            tg_install=True
            print('tangram have been install version:',tg.__version__)
        except ImportError as e:
            raise ImportError(
                'Please install the tangram: `pip install -U tangram-sc`.'
            ) from e

    def __init__(self,
            adata_sc: AnnData,
            adata_sp: AnnData,
            clusters: str = '',
            marker_size: int = 100,
            gene_to_lowercase: bool = False
        ) -> None:
        """
        Initialize the Tangram object.
        """
        self.check_tangram()
        global tg_install
        if tg_install==True:
            global_imports("tangram","tg")
        ad_map_dict={}

        #adata_sc=adata_raw_all
        adata_sc.uns['log1p']={}
        adata_sc.uns['log1p']['base']=None
        sc.pp.filter_genes(adata_sc, min_cells=1)
        sc.tl.rank_genes_groups(adata_sc, groupby=clusters,
                                key_added=f'{clusters}_rank_genes_groups',use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns[f"{clusters}_rank_genes_groups"]["names"]).iloc[0:marker_size, :]

        markers = list(np.unique(markers_df.melt().value.values))
        print('...Calculate The Number of Markers:',len(markers))

        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.clusters=clusters
        self.markers=markers

        #import tangram as tg
        tg.pp_adatas(self.adata_sc, self.adata_sp,
                      genes=self.markers,gene_to_lowercase=gene_to_lowercase)

        print('...Model prepared successfully')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')

    def train(self,
            mode: str = "clusters",
            num_epochs: int = 500,
            device: str = "cuda:0",
            **kwargs: Any
        ) -> None:
        """
        train the model.
        """
        ad_map = tg.map_cells_to_space(self.adata_sc, self.adata_sp,
            #mode="cells",
            mode=mode,
            cluster_label=self.clusters,  # .obs field w cell types
            #density_prior='rna_count_based',
            num_epochs=num_epochs,
            device=device,
            **kwargs
            #device='cpu',
        )

        tg.project_cell_annotations(ad_map, self.adata_sp, annotation=self.clusters)
        print('...Model train successfully')
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')

    def cell2location(self,annotation_list=None):
        """
        Project cell type annotations to spatial coordinates.
        """
        adata_plot=self.adata_sp.copy()
        # construct df_plot
        if annotation_list is None:
            annotation_list=list(set(self.adata_sc.obs[self.clusters]))

        df = adata_plot.obsm["tangram_ct_pred"][annotation_list]
        construct_obs_plot(df, adata_plot, perc=0)
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')
        return adata_plot

def construct_obs_plot(df_plot: pd.DataFrame,
                        adata: AnnData,
                        perc: int = 0,
                        suffix = None
                    ) -> None:
    """
    Construct adata.obs from df_plot.
    """
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())

    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)

def global_imports(modulename,shortname = None, asfunction = False):
    """
    Import a module and add it to the global namespace.
    """
    if shortname is None:
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:
        globals()[shortname] = __import__(modulename)
        # End-of-file (EOF)
