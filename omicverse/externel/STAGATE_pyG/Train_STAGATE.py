import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .STAGATE import STAGATE
from .utils import Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F


def train_STAGATE(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
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
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = STAGATE(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    model.eval()
    z, out = model(data.x, data.edge_index)
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STAGATE_rep

    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STAGATE_ReX'] = ReX

    return adata
