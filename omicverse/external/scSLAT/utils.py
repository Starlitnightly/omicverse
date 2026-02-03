r"""
Miscellaneous utilities
"""
import random

#from pynvml import *
from anndata import AnnData
from typing import List, Optional, Union

import numpy as np
import torch


def norm_to_raw(
    adata: AnnData, 
    library_size: Optional[Union[str,np.ndarray]] = 'total_counts',
    check_size: Optional[int] = 100
) -> AnnData:
    r"""
    Convert normalized adata.X to raw counts
    
    Parameters
    ----------
    adata
        adata to be convert
    library_size
        raw library size of every cells, can be a key of `adata.obs` or a array
    check_size
        check the head `[0:check_size]` row and column to judge if adata normed
    
    Note
    ----------
    Adata must follow scanpy official norm step 
    """
    check_chunk = adata.X[0:check_size,0:check_size].todense()
    assert not all(isinstance(x, int) for x in check_chunk)
    
    from scipy import sparse
    scale_size = np.array(adata.X.expm1().sum(axis=1).round()).flatten()
    if isinstance(library_size, str):
        scale_factor = np.array(adata.obs[library_size])/scale_size
    elif isinstance(library_size, np.ndarray):
        scale_factor = library_size/scale_size
    else:
        try:
            scale_factor = np.array(library_size)/scale_size
        except:
            raise ValueError('Invalid `library_size`')
    scale_factor.resize((scale_factor.shape[0],1))
    raw_count = sparse.csr_matrix.multiply(sparse.csr_matrix(adata.X).expm1(), sparse.csr_matrix(scale_factor))
    raw_count = sparse.csr_matrix(np.round(raw_count))
    adata.X = raw_count
    # adata.layers['counts'] = raw_count
    return adata


def get_free_gpu() -> int:
    r"""
    Get index of GPU with least memory usage
    
    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    index = 0
    max = 0
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        index = i if info.free > max else index
        max = info.free if info.free > max else max
        
    # seed = np.random.randint(1000)
    # os.system(f'nvidia-smi -q -d Memory |grep Used > gpu-{str(seed)}.tmp')
    # memory_available = [int(x.split()[2]) for x in open('gpu.tmp', 'r').readlines()]
    # os.system(f'rm gpu-{str(seed)}.tmp')
    # print(memory_available)
    return index


def global_seed(seed: int) -> None:
    r"""
    Set seed
    
    Parameters
    ----------
    seed 
        int
    """
    if seed > 2**32 - 1 or seed <0:
        seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}.")