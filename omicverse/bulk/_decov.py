

import pandas as pd
import random
import numpy as np
import torch
from typing import Union


class Deconvolution(object):
    def __init__(
        self,adata_bulk,adata_single,
        max_single_cells:int=5000,
        celltype_key:str='celltype',
        cellstate_key:str=None,
        gpu:Union[int,str]=0,
    ):
        self.adata_bulk=adata_bulk
        self.adata_single=adata_single

        self.celltype_key=celltype_key
        self.cellstate_key=cellstate_key

        self.adata_single.var_names_make_unique()
        self.adata_bulk.var_names_make_unique()

        if self.adata_single.shape[0]>max_single_cells:
            print(f"......random select {max_single_cells} single cells")
            import random
            cell_idx=random.sample(self.adata_single.obs.index.tolist(),max_single_cells)
            self.adata_single=self.adata_single[cell_idx,:]

        # build single-cell reference (memory-efficient: avoid to_df)
        var_names = self.adata_single.var_names
        celltypes = pd.unique(self.adata_single.obs[celltype_key])
        sc_ref = pd.DataFrame(index=celltypes, columns=var_names, dtype=float)

        # sum expression within each celltype directly on X (supports dense and sparse)
        test2=adata_single.to_df()
        sc_ref=pd.DataFrame(columns=test2.columns)
        sc_ref_index=[]
        for celltype in list(set(adata_single.obs[celltype_key])):
            sc_ref.loc[celltype]=adata_single[adata_single.obs[celltype_key]==celltype].to_df().sum()
            sc_ref_index.append(celltype)
        sc_ref.index=sc_ref_index
        self.sc_ref=sc_ref
        #self.sc_ref=self.sc_ref.T
        print('......single-cell reference built finished')

        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]

    def deconvolution(
        self,method='tape',
        sep='\t', scaler='mms',
        datatype='counts', genelenfile=None,
        mode='overall', adaptive=True, variance_threshold=0.98,
        save_model_name=None,
        batch_size=128, epochs=128, seed=1,scale_size=2,
        scale=True,n_cores=4,fast_mode=True,pseudobulk_size=2000,
        **kwargs,
    ):

        from ..external.tape import Deconvolution,ScadenDeconvolution
        if method=='scaden':
            CellFractionPrediction=ScadenDeconvolution(self.sc_ref, 
                           self.adata_bulk.to_df(), sep=sep,
                           batch_size=batch_size, epochs=epochs,scale=scale,pseudobulk_size=pseudobulk_size)
        elif method=='tape':
            SignatureMatrix, CellFractionPrediction = \
                Deconvolution(self.sc_ref, self.adata_bulk.to_df(), sep=sep, scaler=scaler,scale=scale,
                            datatype=datatype, genelenfile=genelenfile,
                            mode=mode, adaptive=adaptive, variance_threshold=variance_threshold,
                            save_model_name=save_model_name,batch_size=batch_size, epochs=epochs, seed=seed,
                            pseudobulk_size=pseudobulk_size)
        elif method=='bayesprism':
            from ..external.bulk2single.pybayesprism.prism import Prism
            my_prism=Prism.new_anndata(
                reference_adata=self.adata_single, mixture=self.adata_bulk.to_df(), 
                cell_type_key=self.celltype_key,
                cell_state_key=self.cellstate_key,
                key=None,
                input_type='count.matrix',
                outlier_cut=0.01,
                outlier_fraction=0.1,
            )
            bp_fast = my_prism.run(n_cores=n_cores, fast_mode=fast_mode,**kwargs)
            CellFractionPrediction=bp_fast.posterior_theta_f.theta
        elif method=='omicstweezer':
            from ..external.bulk2single.OmicsTweezer.train_predict import train_predict
            #from ..external.bulk2single.OmicsTweezer.deconvolution import mian
            CellFractionPrediction=train_predict(self.adata_single, self.adata_bulk,
                            ot_weight=1, num=pseudobulk_size,scale=scale,celltype_key=self.celltype_key)
        else:
            raise ValueError(f"method {method} not supported")
        return CellFractionPrediction