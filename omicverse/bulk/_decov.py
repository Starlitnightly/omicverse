

import pandas as pd
import random
import numpy as np
import torch
from typing import Union
from .._registry import register_function


@register_function(
    aliases=['bulk去卷积', 'bulk Deconvolution', 'bulk cell-fraction inference'],
    category="bulk",
    description="Deconvolution class for estimating cell-type composition of bulk RNA-seq using single-cell references and TAPE/Scaden backends.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'obs': ['celltype labels in single reference']},
    produces={'obs': ['predicted cell fractions'], 'uns': ['deconvolution results']},
    auto_fix='none',
    examples=['deconv_obj = ov.bulk.Deconvolution(adata_bulk, adata_single, celltype_key="celltype")', 'frac = deconv_obj.deconvolution(method="tape")'],
    related=['bulk.pyDEG', 'space.Deconvolution']
)
class Deconvolution(object):
    """
    Bulk RNA-seq deconvolution class for inferring cell-type fractions from single-cell references.

    Parameters
    ----------
    adata_bulk:AnnData
        Bulk expression matrix with samples in rows and genes in columns.
    adata_single:AnnData
        Single-cell reference matrix containing cell-level expression profiles
        and cell-type annotations used to build signature profiles.
    max_single_cells:int
        Maximum number of cells to keep from ``adata_single``. If the reference
        contains more cells, a random subset is used to control memory/runtime.
    celltype_key:str
        Column name in ``adata_single.obs`` storing cell-type labels.
    cellstate_key:str or None
        Optional column name in ``adata_single.obs`` storing finer cell-state
        labels (used by methods such as BayesPrism).
    gpu:Union[int,str]
        Compute device selector. Supports CUDA index (for example ``0``),
        explicit strings such as ``'cuda:0'``, ``'mps'``, or ``'cpu'``.
    
    Returns
    -------
    None
        Initializes deconvolution inputs and builds reference expression profiles.
    
    Examples
    --------
    >>> deconv_obj = ov.bulk.Deconvolution(adata_bulk, adata_single, celltype_key="celltype")
    """

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

        self.used_device = self._select_device(gpu)
        self.history=[]

    def _select_device(self, gpu):
        """
        Resolve a PyTorch device from user input and available backends.

        Parameters
        ----------
        gpu:Union[int,str,torch.device]
            Device hint provided by user.

        Returns
        -------
        torch.device
            Selected device with graceful fallback order (CUDA/MPS/CPU).
        """
        if isinstance(gpu, torch.device):
            return gpu
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if isinstance(gpu, str):
            gpu_lower = gpu.lower()
            if gpu_lower == 'mps':
                if mps_available:
                    print('Using Apple Metal Performance Shaders (MPS) backend.')
                    return torch.device('mps')
                print('MPS backend requested but not available, falling back to CPU.')
                return torch.device('cpu')
            if gpu_lower in ('cpu', 'none'):
                return torch.device('cpu')
            if gpu_lower.startswith('cuda'):
                if torch.cuda.is_available():
                    # Allow formats like 'cuda', 'cuda:0'
                    return torch.device(gpu_lower)
                print('CUDA backend requested but not available, falling back to CPU.')
                return torch.device('cpu')
            # Unknown string input, try best effort
            if torch.cuda.is_available():
                print(f"Unrecognized gpu spec '{gpu}', defaulting to CUDA:0.")
                return torch.device('cuda:0')
            if mps_available:
                print(f"Unrecognized gpu spec '{gpu}', defaulting to MPS.")
                return torch.device('mps')
            print(f"Unrecognized gpu spec '{gpu}', defaulting to CPU.")
            return torch.device('cpu')

        if isinstance(gpu, int):
            if torch.cuda.is_available():
                return torch.device(f'cuda:{gpu}')
            if mps_available:
                print('CUDA not available, using MPS backend instead.')
                return torch.device('mps')
            if gpu >= 0:
                print('CUDA not available, falling back to CPU.')
            return torch.device('cpu')

        # Fallback for unexpected types
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if mps_available:
            return torch.device('mps')
        return torch.device('cpu')

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
        """
        Estimate cell-type composition of bulk RNA-seq samples.

        Parameters
        ----------
        method:{'tape', 'scaden', 'bayesprism', 'omicstweezer'}
            Backend used for deconvolution.
        sep:str
            Delimiter used when exporting or reading intermediate text matrices.
        scaler:str
            Scaling method for TAPE input preprocessing, for example ``'mms'``.
        datatype:str
            Data type passed to TAPE (for example raw counts).
        genelenfile:str or None
            Path to gene-length file required by certain preprocessing modes.
        mode:str
            TAPE running mode controlling how sample-wise fractions are inferred.
        adaptive:bool
            Whether to enable adaptive feature selection in TAPE.
        variance_threshold:float
            Variance threshold used by TAPE to keep informative genes.
        save_model_name:str or None
            Prefix/path used to persist trained model weights for reuse.
        batch_size:int
            Mini-batch size for neural-network-based methods.
        epochs:int
            Number of training epochs for neural-network-based methods.
        seed:int
            Random seed for reproducible pseudo-bulk generation/training.
        scale_size:int
            Reserved scaling parameter for compatibility with legacy interfaces.
        scale:bool
            Whether to normalize/scale expression values before fitting.
        n_cores:int
            Number of CPU processes used by BayesPrism.
        fast_mode:bool
            Whether BayesPrism uses fast approximate updates.
        pseudobulk_size:int
            Number of pseudo-bulk mixtures generated for model training.
        **kwargs
            Additional backend-specific keyword arguments, forwarded to the
            selected method.

        Returns
        -------
        pandas.DataFrame
            Predicted cell-type fractions with samples in rows and cell types in columns.

        Examples
        --------
        >>> frac = dec.deconvolution(method='tape', batch_size=128, epochs=200)
        >>> frac = dec.deconvolution(method='bayesprism', n_cores=8, fast_mode=True)
        """

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
                            ot_weight=1, num=pseudobulk_size,scale=scale,celltype_key=self.celltype_key,
                            device=self.used_device,batch_size=batch_size, epochs=epochs)
        else:
            raise ValueError(f"method {method} not supported")
        return CellFractionPrediction
