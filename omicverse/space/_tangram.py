import pandas as pd
import numpy as np
import scanpy as sc

tg_install=False

class Tangram(object):
    
    def check_tangram(self):
        """
        
        """
        global tg_install
        try:
            import tangram as tg
            tg_install=True
            print('tangram have been install version:',tg.__version__)
        except ImportError:
            raise ImportError(
                'Please install the tangram: `pip install -U tangram-sc`.'
            )
    
    def __init__(self,adata_sc,adata_sp,clusters,marker_size=100,
                gene_to_lowercase=False):
        
        self.check_tangram()
        global tg_install
        if tg_install==True:
            global_imports("tangram","tg")
        
        
        ad_map_dict={}

        #adata_sc=adata_raw_all
        adata_sc.uns['log1p']={}
        adata_sc.uns['log1p']['base']=None
        sc.pp.filter_genes(adata_sc, min_cells=1)
        sc.tl.rank_genes_groups(adata_sc, groupby=clusters, key_added=f'{clusters}_rank_genes_groups',use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns[f"{clusters}_rank_genes_groups"]["names"]).iloc[0:marker_size, :]
        
        markers = list(np.unique(markers_df.melt().value.values))
        print('...Calculate The Number of Markers:',len(markers))
        
        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.clusters=clusters
        self.markers=markers
        
        #import tangram as tg
        tg.pp_adatas(self.adata_sc, self.adata_sp, genes=self.markers,gene_to_lowercase=gene_to_lowercase)
        
        print('...Model prepared successfully')
        
    def train(self,mode="clusters",num_epochs=500,device="cuda:0",**kwargs):

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
        
    def cell2location(self,annotation_list=None):
        adata_plot=self.adata_sp.copy()
        # construct df_plot
        if annotation_list==None:
            annotation_list=list(set(self.adata_sc.obs[self.clusters]))
        
        df = adata_plot.obsm["tangram_ct_pred"][annotation_list]
        construct_obs_plot(df, adata_plot, perc=0)
        return adata_plot
        
        

def construct_obs_plot(df_plot, adata, perc=0, suffix=None):
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())

    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)