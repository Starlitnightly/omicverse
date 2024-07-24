"""Module providing a encapsulation of spatrio."""
from typing import Any
import pandas as pd
import numpy as np
import scanpy as sc
from ..externel.spatrio.spatrio import ot_alignment,assign_coord

class CellMap(object):
    """Class representing the object of CellMap."""
    def __init__(self,
                    adata_sc,
                    adata_sp,
                    use_rep_sc='X_pca',
                    use_rep_sp='X_pca',
                 ) -> None:
        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.use_rep_sc=use_rep_sc
        self.use_rep_sp=use_rep_sp
        self.spatrio_decon=None
        self.spatrio_map=None
        self.adata_sp.obsm['spatial']=pd.DataFrame(np.array(self.adata_sp.obsm['spatial']),
                                    columns=['x','y'],index=self.adata_sp.obs.index)
        self.adata_sc.obsm['reduction']=pd.DataFrame(self.adata_sc.obsm[self.use_rep_sc],
                                    index=self.adata_sc.obs.index)

    def map(self,
            sc_type: str = 'celltype',
            sp_type: str = 'leiden',
            alpha: float = 0.1,
            aware_power: int = 2,
            resolution: int = 1,
            aware_spatial: bool = True,
            aware_multi: bool = True,
            use_gpu: bool = True,
            **kwargs: Any
        ) -> pd.DataFrame:

        ##spatial type
        if sp_type=='leiden' and 'leiden' not in self.adata_sp.obs.columns:
            sc.pp.neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            sc.tl.leiden(self.adata_sp,resolution=resolution)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type=='louvain' and 'louvain' not in self.adata_sp.obs.columns:
            sc.pp.neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            sc.tl.louvain(self.adata_sp)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type is None:
            aware_spatial=False
        else:
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()

        ##single cell type
        if sc_type is None:
            aware_multi=False
        else:
            self.adata_sc.obs['type']=self.adata_sc.obs[sc_type].tolist()

        spatrio_decon = ot_alignment(adata1 = self.adata_sp, adata2 = self.adata_sc,
                                     alpha = alpha, aware_power = aware_power,
                aware_spatial = aware_spatial,
                aware_multi=aware_multi,use_gpu=use_gpu,**kwargs)
        self.spatrio_decon=spatrio_decon
        return spatrio_decon

    def assign_coord(self,**kwargs):
        """
        Assign spatial coordinates to single cell data.
        """
        spatrio_map = assign_coord(adata1 = self.adata_sp,adata2 = self.adata_sc,
                                   out_data = self.spatrio_decon,**kwargs)
        #self.adata_sp.obs=self.adata_sp.obs.join(spatrio_map.set_index('cell'))

        self.spatrio_map=spatrio_map

        loc1=pd.DataFrame(spatrio_map[['Cell_xcoord','Cell_ycoord','spot','spot_type',
                                       'value']].values,
                          columns=['Cell_xcoord','Cell_ycoord','spot','spot_type','value'],
                          index=spatrio_map['cell'].tolist())
        print('...assigning spatial coordinates to single cell data')
        self.adata_sc.obs['Cell_xcoord']=loc1.loc[self.adata_sc.obs.index,'Cell_xcoord'].tolist()
        self.adata_sc.obs['Cell_ycoord']=loc1.loc[self.adata_sc.obs.index,'Cell_ycoord'].tolist()
        self.adata_sc.obs['spot_type']=loc1.loc[self.adata_sc.obs.index,'spot_type'].tolist()
        self.adata_sc.obs['spot']=loc1.loc[self.adata_sc.obs.index,'spot'].tolist()
        self.adata_sc.obs['spot_value']=loc1.loc[self.adata_sc.obs.index,'value'].tolist()

        print('...adding spatial coordinates to single cell data')
        self.adata_sc.obsm['spatial']=loc1.loc[self.adata_sc.obs.index,
                                               ['Cell_xcoord','Cell_ycoord']].values
        self.adata_sc.uns['spatial']=self.adata_sp.uns['spatial'].copy()
        return self.adata_sc
    # End-of-file (EOF)
