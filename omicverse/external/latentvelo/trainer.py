import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from .dataloader import Dataset
from .utils import normalize, sparse_mx_to_torch_sparse_tensor, as_float_ndarray
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import batch_func

# Detect device - use CUDA if available, otherwise use CPU
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_device(d):
    """Override module-level torch device (e.g., 'cpu', 'cuda', torch.device)."""
    global device
    device = th.device(d) if not isinstance(d, th.device) else d
    print(f'LatentVelo trainer using device: {device}')

# setup adjacency matrix
def set_adj(adata):
    import scipy as scp
    adata.obsp['adj'] = 0.9*adata.obsp['adj'] + scp.sparse.eye(adata.obsp['adj'].shape[0])

def train_vae(model, adata, epochs = 50, learning_rate = 1e-2, batch_size = 200, grad_clip = 1, shuffle=True, test=0.1, name = '', optimizer='adam', random_seed=42):

    results_folder = './' + name + '/'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    else:
        print('Warning, folder already exists. This may overwrite a previous fit.')
    
    if optimizer == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer == 'adamW':
        optimizer = th.optim.AdamW(model.parameters(), lr = learning_rate)
    
    scheduler_plateau = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
    factor = 0.75, threshold = 0.05, threshold_mode ='rel', patience = 5, 
    min_lr = 1e-5)

    if 'root' not in adata.obs:
        adata.obs['root'] = 0
    
    gcn = model.gcn
    batch_correction = model.batch_correction
    if not batch_correction:
        batch_id = None
        batch_onehot = None
    
    if not model.celltype_corr and not model.celltype_velo:
        celltype_id = None

    if not model.time_reg:
        exp_time = None
    
    if model.likelihood_model == 'gaussian':
        s_count_key = 'normedS'
        u_count_key = 'normedU'
    else:
        s_count_key = 'S'
        u_count_key = 'U'
    
    dataset = Dataset(adata, batch_size, shuffle=shuffle, test=test, random_seed=random_seed)
    loader = th.utils.data.DataLoader(dataset,
                                     batch_size = 1,
                                     shuffle = True, drop_last = False,
                                     num_workers = 0, pin_memory=True,
                                      collate_fn=lambda x: x[0])
    model = model.to(device)
    model_state_history = [model.state_dict()]
    epoch_history = [0]
    val_ae_history = [np.inf]
    val_traj_history = [np.inf]
    val_traj_rel_history = [np.inf]

    adata.uns['index_test'] = dataset.adata.uns['index_test']
    
    if adata.obsp['adj'][0,0] < 1.0:
        set_adj(adata)
    
    pbar = tqdm(range(epochs), desc='LatentVelo', unit='epoch')
    for epoch in pbar:
        
        model = model.train()
        val_loss = 0.0
        val_num = 0.0
        for batch in loader:
            
            optimizer.zero_grad()
            
            s = batch[s_count_key].to(device)
            normed_s = batch['normedS'].to(device)
            mask_s = batch['maskS'].to(device)
            s_size_factors = batch['spliced_size_factor'].to(device)[:,None]

            u = batch[u_count_key].to(device)
            normed_u = batch['normedU'].to(device)
            mask_u = batch['maskU'].to(device)
            u_size_factors = batch['unspliced_size_factor'].to(device)[:,None]  

            velo_genes_mask = batch['velo_genes_mask'].to(device)

            root_cells = batch['root'].to(device)

            index_train, index_test = batch['index_train'].to(device), batch['index_test'].to(device)

            adj = sparse_mx_to_torch_sparse_tensor(batch['adj']).to(device)
            if batch_correction:
                batch_id = batch['batch_id'].to(device)[:,None]
                batch_onehot = batch['batch_onehot'].to(device)
            if model.celltype_corr or model.celltype_velo:
                celltype_id = batch['celltype_id'].to(device)
            if model.time_reg:
                exp_time = batch['exp_time'].to(device)
            
            loss, validation_ae, validation_traj, validation_velo, orig_index = model.loss(normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, velo_genes_mask, adj, root_cells, batch_id=(batch_id, batch_onehot, celltype_id, exp_time), epoch=epoch)
            
            curr_index = th.arange(loss.shape[0]).to(device)
            index_train = th.stack([i for i in curr_index if orig_index[i] in index_train])
            train_loss = th.mean(loss[index_train])
            
            index_test_ = [i for i in curr_index  if orig_index[i] in index_test]
            if len(index_test_) > 0:
                index_test = th.stack(index_test_)
                vloss = loss.detach()[index_test]
            else:
                vloss = th.zeros(1).to(device)
            
            val_loss += th.sum(vloss)
            val_num += vloss.shape[0]
            
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        model = model.eval()
        with th.no_grad():
            
            adata = dataset.adata
            if model.likelihood_model == 'gaussian':
                s = th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device)
                u = th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device)
            else:
                s = th.Tensor(as_float_ndarray(adata.layers['spliced_counts'])).to(device)
                u = th.Tensor(as_float_ndarray(adata.layers['unspliced_counts'])).to(device)

            normed_s = th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device)
            normed_u = th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device)

            s_size_factors = th.Tensor(adata.obs['spliced_size_factor'].astype(float)).to(device)[:,None]
            u_size_factors = th.Tensor(adata.obs['unspliced_size_factor'].astype(float)).to(device)[:,None]

            mask_s = th.Tensor(as_float_ndarray(adata.layers['mask_spliced'])).to(device)
            mask_u = th.Tensor(as_float_ndarray(adata.layers['mask_unspliced'])).to(device)

            velo_genes_mask = th.Tensor(as_float_ndarray(adata.layers['velo_genes_mask'])).to(device)

            root_cells = th.Tensor(adata.obs['root'].astype(float)).to(device)[:,None]

            if model.time_reg:
                exp_time = th.Tensor(adata.obs['exp_time']).float().to(device)
            
            adj = adata.obsp['adj']
            if batch_correction:
                batch_id = th.Tensor(adata.obs['batch_id']).to(device)[:,None]
                batch_onehot = th.Tensor(adata.obsm['batch_onehot']).to(device)
            if model.celltype_corr or model.celltype_velo:
                celltype_id = th.Tensor(adata.obs['celltype_id']).to(device)
            
            loss, validation_ae, validation_traj, validation_velo, _ = batch_func(model.loss, (normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, velo_genes_mask, adj, root_cells, (batch_id, batch_onehot, celltype_id, exp_time)), 5, split_size = batch_size) #[:4]
            
            loss = loss.mean().cpu().numpy()
            validation_ae = validation_ae.mean().cpu().numpy()
            validation_traj = validation_traj.mean().cpu().numpy()
            validation_velo = validation_velo.mean().cpu().numpy()
            
            pbar.set_postfix({
                'epoch': epoch,
                'full': f"{loss:.3f}",
                'val': f"{(val_loss/val_num).cpu().numpy():.3f}",
                'recon': f"{validation_ae:.3f}",
                'traj': f"{validation_traj:.3f}",
                'reg': f"{validation_velo:.3f}"
            })
        
        scheduler_plateau.step(validation_traj + validation_ae)
        epoch_history.append(epoch)
        val_ae_history.append(validation_ae)
        val_traj_history.append(validation_traj)
        model_state_history.append(model.state_dict())
        
        th.save(model.state_dict(), results_folder+'model_state_epoch%d.params'%(epoch))
        
        if epoch == epochs - 1:
            th.save(model.state_dict(), results_folder+'model_state_epoch%d.params'%(epoch))
        
        del model_state_history[0]
        
    # determine best model
    val_history = np.array(val_ae_history) + np.array(val_traj_history)
    best_index = np.argmin(val_history)

    print('Loading best model at %d epochs.'%epoch_history[best_index])
    model.load_state_dict(th.load(results_folder+'model_state_epoch%d.params'%epoch_history[best_index], map_location=th.device('cuda')))
    return np.array(epoch_history)[1:], np.array(val_ae_history)[1:], np.array(val_traj_history)[1:]

def plot_history(epochs, val_traj, val_ae):
    
    plt.plot(epochs, val_traj, label = 'Velocity field')
    plt.plot(epochs, val_ae, label = 'Autoencoder')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
