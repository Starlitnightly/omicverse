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
        adata_sc_copy=self.adata_sc.copy()
        adata_sc_copy=adata_sc_copy[loc1.index.tolist()]
        adata_sc_copy.obs['Cell_xcoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_xcoord'].tolist()
        adata_sc_copy.obs['Cell_ycoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_ycoord'].tolist()
        adata_sc_copy.obs['spot_type']=loc1.loc[adata_sc_copy.obs.index,'spot_type'].tolist()
        adata_sc_copy.obs['spot']=loc1.loc[adata_sc_copy.obs.index,'spot'].tolist()
        adata_sc_copy.obs['spot_value']=loc1.loc[adata_sc_copy.obs.index,'value'].tolist()

        print('...adding spatial coordinates to single cell data')
        adata_sc_copy.obsm['spatial']=loc1.loc[adata_sc_copy.obs.index,
                                               ['Cell_xcoord','Cell_ycoord']].values
        adata_sc_copy.uns['spatial']=self.adata_sp.uns['spatial'].copy()
        return adata_sc_copy
    # End-of-file (EOF)


class CellLoc(object):
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

    def loc_map(self,
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

    def load_map(self,map_info,
                 sc_type: str = 'celltype',
                 sp_type: str = 'leiden',
                 resolution: int = 1,
                 aware_spatial: bool = True,
            aware_multi: bool = True,):
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
        self.spatrio_decon=map_info
        return map_info

    def save_map(self,path):
        self.spatrio_decon.to_csv(path)
    
    def loc_prob(self,spot_cell_prob,
                 sc_type: str = 'celltype',
                 sc_prop: float = 0.5,
                 n_cpu=6,):

        map_info=self.spatrio_decon.copy()
        from tqdm import tqdm
        df=spot_cell_prob

        # 创建字典
        first_max_dict = {}
        first_max_value_dict = {}
        
        # 找到每个索引得分第一高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[0]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            first_max_dict[idx] = third_max_column
            first_max_value_dict[idx]=third_max_value

        map_info['spot_prop1']=map_info['spot'].map(first_max_dict)
        map_info['spot_prop_value1']=map_info['spot'].map(first_max_value_dict)

        # 创建字典
        second_max_dict = {}
        second_max_value_dict = {}
        
        # 找到每个索引得分第二高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[1]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            second_max_dict[idx] = third_max_column
            second_max_value_dict[idx]=third_max_value
        map_info['spot_prop2']=map_info['spot'].map(second_max_dict)
        map_info['spot_prop_value2']=map_info['spot'].map(second_max_value_dict)

        # 创建字典
        third_max_dict = {}
        third_max_value_dict = {}

        # 找到每个索引得分第三高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[2]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            third_max_dict[idx] = third_max_column
            third_max_value_dict[idx]=third_max_value

     
        map_info['spot_prop3']=map_info['spot'].map(third_max_dict)
        map_info['spot_prop_value3']=map_info['spot'].map(third_max_value_dict)

        print('...adding spot prop to map info')

        map_info['cell_type']=map_info['cell'].map(dict(zip(
            self.adata_sc.obs.index.tolist(),
            self.adata_sc.obs[sc_type].tolist()   
        )))
        print('...adding cell type to map info')

        cond1 = (map_info['cell_type'] == map_info['spot_prop1']) & (map_info['spot_prop_value1'] > sc_prop)
        print('... finish spot type 1 filter')
        cond2 = (map_info['cell_type'] == map_info['spot_prop2']) & (map_info['spot_prop_value2'] > sc_prop)
        print('... finish spot type 2 filter')
        cond3 = (map_info['cell_type'] == map_info['spot_prop3']) & (map_info['spot_prop_value3'] > sc_prop)
        print('... finish spot type 3 filter')
        mask = cond1 | cond2 | cond3

        print('...filtering map info')
        # 应用过滤条件 ----------------------------------------------------------
        map_info.loc[~mask, 'value'] = -1
        map_info=map_info[map_info['value']!=-1]
        self.spatrio_decon = map_info[['spot', 'cell', 'value']]



    def loc_assign(self,**kwargs):
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
        adata_sc_copy=self.adata_sc.copy()
        adata_sc_copy=adata_sc_copy[loc1.index.tolist()]
        adata_sc_copy.obs['Cell_xcoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_xcoord'].tolist()
        adata_sc_copy.obs['Cell_ycoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_ycoord'].tolist()
        adata_sc_copy.obs['spot_type']=loc1.loc[adata_sc_copy.obs.index,'spot_type'].tolist()
        adata_sc_copy.obs['spot']=loc1.loc[adata_sc_copy.obs.index,'spot'].tolist()
        adata_sc_copy.obs['spot_value']=loc1.loc[adata_sc_copy.obs.index,'value'].tolist()

        print('...adding spatial coordinates to single cell data')
        adata_sc_copy.obsm['spatial']=loc1.loc[adata_sc_copy.obs.index,
                                               ['Cell_xcoord','Cell_ycoord']].values
        adata_sc_copy.uns['spatial']=self.adata_sp.uns['spatial'].copy()
        adata_sc_copy.obsm['spatial']=adata_sc_copy.obsm['spatial'].astype(float)
        del adata_sc_copy.obsm['reduction']
        
        return adata_sc_copy