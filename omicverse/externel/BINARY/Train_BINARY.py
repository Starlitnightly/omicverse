import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .Model import Model
from .utils import Transfer_pytorch_Data

import torch

import torch.nn.functional as F


def train_BINARY(adata, 
                pos_weight=10.0, 
                hidden_dims=[512, 30], 
                n_epochs=1000, 
                lr=0.001, 
                key_added='BINARY',
                gradient_clipping=5.,  
                weight_decay=0.0001, 
                verbose=True, 
                random_seed=0,  
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                ):
    """
    Train a binary representation
    
    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix containing the samples to be embedded.
        
    pos_weight : float, optional (default=10.0)
        The positive weight parameter used in the BCEWithLogitsLoss.
        
    hidden_dims : list of int, optional (default=[512, 30])
        The dimensionality of the hidden layers in the neural network.
        
    n_epochs : int, optional (default=1000)
        Number of training epochs.
        
    lr : float, optional (default=0.001)
        The learning rate for the Adam optimizer.
        
    key_added : str, optional (default='BINARY')
        Key under which the binary representations are saved in the `obsm` slot of the `adata` object.
        
    gradient_clipping : float, optional (default=5.)
        Maximum allowed norm of the gradients for stabilization.
        
    weight_decay : float, optional (default=0.0001)
        Weight decay (L2 penalty) used in the Adam optimizer.
        
    verbose : bool, optional (default=True)
        Whether to print status messages during the process.
        
    random_seed : int, optional (default=0)
        Seed for random number generator for reproducibility.
        
    device : torch.device, optional
        The device (e.g., CPU or CUDA) on which the model and data should be loaded.
        
    Returns:
    -------
    AnnData
        The input `adata` object with the binary representations stored in `adata.obsm[key_added]`.
    """
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = True

    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Graph' not in adata.uns.keys():
        raise ValueError("Spatial_Graph is not existed! Run Construct_Spatial_Graph first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = Model(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        loss = criterion(out, data.x.float())
        loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    model.eval()
    z, out = model(data.x, data.edge_index)
    
    BINARY_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = BINARY_rep

    return adata