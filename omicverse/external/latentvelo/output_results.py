import torch as th
import torch.nn as nn
from torch.nn import functional as F
from .dataloader import Dataset
from .utils import normalize, sparse_mx_to_torch_sparse_tensor, batch_func
from .trainer import set_adj
import os
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scvelo as scv

# Detect device - use CUDA if available, otherwise use CPU
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

def set_device(d):
    global device
    device = th.device(d) if not isinstance(d, th.device) else d
    print(f'LatentVelo output using device: {device}')

from sklearn.metrics import r2_score
 
def R2_score(estimated, true):
    """
    Compute R2 score using both spliced/unspliced
    """
    size = estimated.shape[1]//2
    
    if type(estimated) == th.Tensor:
        estimated = estimated.cpu().numpy()
    if type(true) == th.Tensor:
        true = true.cpu().numpy()
    
    scores = []
    for i in range(0, size):
        greater_zero = (true[:,i] > 0) & (true[:,i+size] > 0)
        if greater_zero.sum() > 100:
            scores.append(r2_score(estimated[greater_zero][:,[i,i+size]], true[greater_zero][:,[i,i+size]]))
        else:
            scores.append(np.nan)
            
    return np.array(scores)



@th.no_grad()
def output_results(model, adata, 
            gene_velocity = False, save_name = None, 
            samples=10, embedding='umap', decoded=False):
    """
    Output results of the model
    """
    
    annot = model.annot
    exp_time = model.exp_time
    
    model = model.to(device)

    def gene_velocity_func(z, vz, batch_id, mode = 's', create_graph=False):

        if model.batch_correction:
            if model.shared:
                jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id), z, vz, create_graph=False)[1]
            else:
                if mode == 's':
                    jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id, 's'), z, vz, create_graph=False)[1]
                else:
                    jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id, 'u'), z, vz, create_graph=False)[1]
        else:
            if model.shared:
                jvp = th.autograd.functional.jvp(model.decoder, z, v, create_graph=False)[1]
            else:
                if mode == 's':
                    jvp = th.autograd.functional.jvp(model.decoder_s, z, vz, create_graph=False)[1]
                else:
                    jvp = th.autograd.functional.jvp(model.decoder_u, z, vz, create_graph=False)[1]
        
        return jvp
    
    from .utils import as_float_ndarray
    s = th.Tensor(as_float_ndarray(adata.layers['spliced_counts'])).to(device)
    u = th.Tensor(as_float_ndarray(adata.layers['unspliced_counts'])).to(device)
    normed_s = th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device)
    normed_u = th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device)
    if model.likelihood_model == 'nb':
        s_size_factor = th.Tensor(adata.obs['spliced_size_factor'].astype(float)).to(device)[:,None]
        u_size_factor = th.Tensor(adata.obs['unspliced_size_factor'].astype(float)).to(device)[:,None]
    
    # check if adjacency matrix has already been normalized
    if adata.obsp['adj'][0,0] < 1:
        set_adj(adata)
        
    adj = adata.obsp['adj']
    batch_id = th.Tensor(adata.obs['batch_id']).to(device)[:,None].to(device)
    batch_onehot = th.Tensor(adata.obsm['batch_onehot']).to(device)
    if annot:
        celltype = th.Tensor(adata.obsm['celltype']).to(device)
    if exp_time:
        times = th.Tensor(adata.obs['exp_time']).to(device)[:,None]

    
    # compute latent embedding
    z = []
    ztraj = []
    velocity = []
    latent_time = []
    with th.no_grad():
        for i in range(samples):
            if model.batch_correction:
                if annot:
                    if exp_time:
                        z_, ztraj_, velocity_, latent_time_, hidden, u_latent = batch_func(model.reconstruct_latent, 
                                                                                             (normed_s, normed_u, (celltype, times), adj, batch_onehot), 6)
                    else:
                        z_, ztraj_, velocity_, latent_time_, hidden, u_latent = model.batch_func(model.reconstruct_latent, 
                                                                                                 (normed_s, normed_u, celltype, adj, batch_onehot), 6)
                else:
                    z_, ztraj_, velocity_, latent_time_, hidden = model.batch_func(model.reconstruct_latent, 
                                                                                   (normed_s, normed_u, adj, batch_onehot), 5)  
            else:
                if annot:
                    if exp_time:
                        z_, ztraj_, velocity_, latent_time_, hidden, u_latent = batch_func(model.reconstruct_latent, 
                                                                                                      (normed_s, normed_u, (celltype, times), adj), 6)
                    else:
                        z_, ztraj_, velocity_, latent_time_, hidden, u_latent = model.batch_func(model.reconstruct_latent, 
                                                                                                      (normed_s, normed_u, celltype, adj), 6)
                else:
                    z_, ztraj_, velocity_, latent_time_, hidden = model.batch_func(model.reconstruct_latent, 
                                                                                   (normed_s, normed_u, adj), 5)
            z.append(z_)
            ztraj.append(ztraj_)
            velocity.append(velocity_)
            latent_time.append(latent_time_)
    z = th.stack(z).mean(0)
    ztraj = th.stack(ztraj).mean(0)
    velocity = th.stack(velocity).mean(0)
    latent_time = th.stack(latent_time).mean(0)

    # save to anndata
    if 'X_'+embedding in adata.obsm:
        embedding_data = {'X_'+embedding: adata.obsm['X_'+embedding],
                          'X_pca': adata.obsm['X_pca'],
                          'zr': z[:,2*model.latent:].cpu().numpy()}
    else:
        embedding_data = {'X_pca': adata.obsm['X_pca'],
                          'zr': z[:,2*model.latent:].cpu().numpy()}
                        
    latent_adata = ad.AnnData(z[:,:model.latent].detach().cpu().numpy(), 
                          obsm=embedding_data,
                         layers = {'spliced': z[:,:model.latent].detach().cpu().numpy(),
                                   'spliced_traj': ztraj[:,:model.latent].detach().cpu().numpy(),
                                   'spliced_velocity': velocity[:,:model.latent].detach().cpu().numpy(),
                                   'unspliced': z[:,model.latent:2*model.latent].detach().cpu().numpy(),
                                   'unspliced_traj': ztraj[:,model.latent:2*model.latent].detach().cpu().numpy(),
                                   'unspliced_velocity': velocity[:,model.latent:2*model.latent].detach().cpu().numpy()},
                              obsp=adata.obsp, uns=adata.uns, obs=adata.obs)
    latent_adata.obs['latent_time'] = latent_time.cpu().numpy()


    if gene_velocity:
        vs = model.batch_func(lambda x,y,z: gene_velocity_func(x,y,z,mode='s'), (z[:,:model.latent].to(device), velocity[:,:model.latent].to(device), batch_id), 1)[0]
        vu = model.batch_func(lambda x,y,z: gene_velocity_func(x,y,z,mode='u'), (z[:,model.latent:2*model.latent].to(device), velocity[:,model.latent:2*model.latent].to(device),
                                                batch_id), 1)[0]
        
        adata.layers['velo_u'] = (vu.cpu().numpy())
        adata.layers['velo_s'] = (vs.cpu().numpy())
        adata.layers['velo'] = (vs.cpu().numpy())
        if model.batch_correction:
            if model.likelihood_model == 'gaussian':
                shat = model.decoder_batch(z[:,:model.latent].to(device), batch_id, 's')
                uhat = model.decoder_batch(z[:,model.latent:2*model.latent].to(device), batch_id, 'u')
                shat_traj = model.decoder_batch(ztraj[:,:model.latent].to(device), batch_id, 's')
                uhat_traj = model.decoder_batch(ztraj[:,model.latent:2*model.latent].to(device), batch_id, 'u')
            else:
                shat = F.softmax(model.decoder_batch(z[:,:model.latent].to(device), batch_id, 's'), dim=-1) * s_size_factor 
                uhat = F.softmax(model.decoder_batch(z[:,model.latent:2*model.latent].to(device), batch_id, 'u'), dim=-1) * u_size_factor
                shat_traj = F.softmax(model.decoder_batch(ztraj[:,:model.latent].to(device), batch_id, 's'), dim=-1) * s_size_factor
                uhat_traj = F.softmax(model.decoder_batch(ztraj[:,model.latent:2*model.latent].to(device), batch_id, 'u'), dim=-1) * u_size_factor
        else:
            if model.likelihood_model == 'gaussian':
                shat = model.decoder_s(z[:,:model.latent].to(device))
                uhat = model.decoder_u(z[:,model.latent:2*model.latent].to(device))
                shat_traj = model.decoder_s(ztraj[:,:model.latent].to(device))
                uhat_traj = model.decoder_u(ztraj[:,model.latent:2*model.latent].to(device))
            else:
                shat = F.softmax(model.decoder_s(z[:,:model.latent].to(device)), dim=-1) * s_size_factor 
                uhat = F.softmax(model.decoder_u(z[:,model.latent:2*model.latent].to(device)), dim=-1) * u_size_factor 
                shat_traj = F.softmax(model.decoder_s(ztraj[:,:model.latent].to(device)), dim=-1) * s_size_factor 
                uhat_traj = F.softmax(model.decoder_u(ztraj[:,model.latent:2*model.latent].to(device)), dim=-1) * u_size_factor 

        if model.likelihood_model == 'gaussian':
            R2 = R2_score(th.cat((shat, uhat), dim=-1), 
                          th.cat((normed_s, normed_u), dim=-1))
            R2_traj = R2_score(th.cat((shat_traj, uhat_traj), dim=-1), 
                               th.cat((normed_s, normed_u), dim=-1))
            
            test_index = adata.obs.index.isin(adata.uns['index_test'])
            R2_test = R2_score(th.cat((shat[test_index], uhat[test_index]), dim=-1), 
                               th.cat((normed_s[test_index], normed_u[test_index]), dim=-1))
            R2_traj_test = R2_score(th.cat((shat_traj[test_index], uhat_traj[test_index]), dim=-1), 
                                    th.cat((normed_s[test_index], normed_u[test_index]), dim=-1))
        else:
            R2 = R2_score(th.cat((shat, uhat), dim=-1), 
                          th.cat((s, u), dim=-1))
            R2_traj = R2_score(th.cat((shat_traj, uhat_traj), dim=-1), 
                               th.cat((s, u), dim=-1))
            
            test_index = adata.obs.index.isin(adata.uns['index_test'])
            R2_test = R2_score(th.cat((shat[test_index], uhat[test_index]), dim=-1), 
                               th.cat((s[test_index], u[test_index]), dim=-1))
            R2_traj_test = R2_score(th.cat((shat_traj[test_index], uhat_traj[test_index]), dim=-1), 
                                    th.cat((s[test_index], u[test_index]), dim=-1))
            
            
        adata.var['R2'] = R2
        adata.var['R2_traj'] = R2_traj
        adata.var['R2_test'] = R2_test
        adata.var['R2_traj_test'] = R2_traj_test
        adata.obs['latent_time'] = latent_time.cpu().numpy()

        if decoded:
            adata.layers['shat'] = shat.detach().cpu().numpy()
            adata.layers['uhat'] = uhat.detach().cpu().numpy()
            adata.layers['shat_traj'] = shat_traj.detach().cpu().numpy()
            adata.layers['uhat_traj'] = uhat_traj.detach().cpu().numpy()
    
    if save_name != None:
        latent_adata.write(save_name+'.h5ad')

    if gene_velocity:
        return latent_adata, adata
    else:
        return latent_adata



@th.no_grad()
def cell_trajectories(model, adata, mode='normal', time_steps = 50):
    """
    Infer cell trajectories
    """

    annot = model.annot
    exp_time = model.exp_time
    
    
    if annot:
        with th.no_grad():
            if not model.batch_correction:
                if exp_time:
                    z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     (th.Tensor(adata.obsm['celltype']).to(device),
                                                                      th.Tensor(adata.obs['exp_time'][:,None]).to(device)),
                                                                     adata.obsp['adj']), 2, split_size=100)
                else:
                    z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     th.Tensor(adata.obsm['celltype']).to(device),
                                                                     adata.obsp['adj']), 2, split_size=100)
            else:
                if exp_time:
                    z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     (th.Tensor(adata.obsm['celltype']).to(device),
                                                                      th.Tensor(adata.obs['exp_time'][:,None]).to(device)),
                                                                     adata.obsp['adj'],
                                                                     (th.Tensor(adata.obs['batch_id'][:,None]).to(device),
                                                                     th.Tensor(adata.obsm['batch_onehot']).to(device))), 2, split_size=100)
                else:
                    z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     th.Tensor(adata.obsm['celltype']).to(device),
                                                                     adata.obsp['adj'],
                                                                     (th.Tensor(adata.obs['batch_id'][:,None]).to(device),
                                                                     th.Tensor(adata.obsm['batch_onehot']).to(device))), 2, split_size=100)

    else:
        with th.no_grad():
            if not model.batch_correction:#batch_id == None and batch_onehot == None:
                z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     adata.obsp['adj']), 2, split_size=100)
            else:
                z_traj, times = batch_func(model.cell_trajectories, (th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device),
                                                                     th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device),
                                                                     adata.obsp['adj'],
                                                                     th.Tensor(adata.obs['batch_id'][:,None]).to(device),
                                                                     th.Tensor(adata.obsm['batch_onehot']).to(device)), 2, split_size=100)

    return z_traj, times




@th.no_grad()
def output_atac_results(model, adata, gene_velocity = False, save_name = None, samples=10, annot=False, embedding='umap', exp_time=False, decoded=False):
    """
    Output results of the atac model
    """
    model = model.to(device)

    def gene_velocity_func(z, vz, batch_id, mode = 's', create_graph=False):

        if model.batch_correction:
            if model.shared:
                jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id), z, vz, create_graph=False)[1]
            else:
                if mode == 's':
                    jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id, 's'), z, vz, create_graph=False)[1]
                else:
                    jvp = th.autograd.functional.jvp(lambda x: model.decoder_batch(x, batch_id, 'u'), z, vz, create_graph=False)[1]
        else:
            if model.shared:
                jvp = th.autograd.functional.jvp(model.decoder, z, v, create_graph=False)[1]
            else:
                if mode == 's':
                    jvp = th.autograd.functional.jvp(model.decoder_s, z, vz, create_graph=False)[1]
                elif mode == 'u':
                    jvp = th.autograd.functional.jvp(model.decoder_u, z, vz, create_graph=False)[1]
                elif mode == 'c':
                    jvp = th.autograd.functional.jvp(model.decoder_atac, z, vz, create_graph=False)[1]
        
        return jvp
    
    from .utils import as_float_ndarray
    s = th.Tensor(as_float_ndarray(adata.layers['spliced_counts'])).to(device)
    u = th.Tensor(as_float_ndarray(adata.layers['unspliced_counts'])).to(device)
    normed_s = th.Tensor(as_float_ndarray(adata.layers['spliced'])).to(device)

    normed_u = th.Tensor(as_float_ndarray(adata.layers['unspliced'])).to(device)
    normed_a = th.Tensor(as_float_ndarray(adata.layers['atac'])).to(device)
    
    if model.likelihood_model == 'nb':
        s_size_factors = th.Tensor(adata.obs['spliced_size_factor'].astype(float)).to(device)
        u_size_factors = th.Tensor(adata.obs['unspliced_size_factor'].astype(float)).to(device)
    
    # check if adjacency matrix has already been normalized
    if adata.obsp['adj'][0,0] < 1:
        set_adj(adata)
    
    adj = adata.obsp['adj']
    batch_id = th.Tensor(adata.obs['batch_id']).to(device)[:,None].to(device)
    batch_onehot = th.Tensor(adata.obsm['batch_onehot']).to(device)
    if annot:
        celltype = th.Tensor(adata.obsm['celltype']).to(device)
    if exp_time:
        times = th.Tensor(adata.obs['exp_time']).to(device)[:,None]
    
    z = []
    velocity = []
    latent_time = []
    with th.no_grad():
        for i in range(5):
            z_, velocity_, latent_time_, hidden = model.batch_func(model.reconstruct_latent, (normed_s, normed_u, normed_a, adj, 0, 0), 4)
            z.append(z_)
            velocity.append(velocity_)
            latent_time.append(latent_time_)
    z = th.stack(z).mean(0)
    ztraj = z#th.stack(z).mean(0)
    velocity = th.stack(velocity)
    velocity_error = velocity.std(0)
    velocity = velocity.mean(0)
    latent_time = th.stack(latent_time).mean(0)
    
    
    latent_adata = ad.AnnData(z[:,:model.latent].detach().cpu().numpy(), 
                          obsm={'X_'+embedding: adata.obsm['X_'+embedding],
                                'X_pca': adata.obsm['X_pca'],
                                'zr': z[:,2*model.latent:].cpu().numpy()},
                         layers = {'spliced': z[:,:model.latent].detach().cpu().numpy(),
                                   'unspliced': z[:,model.latent:2*model.latent].detach().cpu().numpy(),
                                   'chromatin': z[:,model.latent*2:3*model.latent].detach().cpu().numpy(),
                                   'spliced_velocity': velocity[:,:model.latent].detach().cpu().numpy(),
                                   'unspliced_velocity': velocity[:,model.latent:2*model.latent].detach().cpu().numpy(),
                                   'chromatin_velocity': velocity[:,2*model.latent:3*model.latent].detach().cpu().numpy()},
                              obsp=adata.obsp, uns=adata.uns, obs=adata.obs)
    latent_adata.obs['latent_time'] = latent_time.cpu().numpy()
    
    
    if gene_velocity:
        vs = model.batch_func(lambda x,y,z: gene_velocity_func(x,y,z,mode='s'), (z[:,:model.latent].to(device), velocity[:,:model.latent].to(device), batch_id), 1)[0]
        vu = model.batch_func(lambda x,y,z: gene_velocity_func(x,y,z,mode='u'), (z[:,model.latent:2*model.latent].to(device), velocity[:,model.latent:2*model.latent].to(device), batch_id), 1)[0]
        
        vc = model.batch_func(lambda x,y,z: gene_velocity_func(x,y,z,mode='c'), (z[:,2*model.latent:3*model.latent].to(device), velocity[:,2*model.latent:3*model.latent].to(device), batch_id), 1)[0]
        
        adata.layers['velo_c'] = (vc.cpu().numpy())
        adata.layers['velo_u'] = (vu.cpu().numpy())
        adata.layers['velo_s'] = (vs.cpu().numpy())
        adata.layers['velo'] = (vs.cpu().numpy())
        if model.batch_correction:
            if model.likelihood_model == 'gaussian':
                shat = model.decoder_batch(z[:,:model.latent].to(device), batch_id, 's')
                uhat = model.decoder_batch(z[:,model.latent:2*model.latent].to(device), batch_id, 'u')
                shat_traj = model.decoder_batch(ztraj[:,:model.latent].to(device), batch_id, 's')
                uhat_traj = model.decoder_batch(ztraj[:,model.latent:2*model.latent].to(device), batch_id, 'u')
            else:
                shat = F.softmax(model.decoder_batch(z[:,:model.latent].to(device), batch_id, 's'), dim=-1) * s_size_factor 
                uhat = F.softmax(model.decoder_batch(z[:,model.latent:2*model.latent].to(device), batch_id, 'u'), dim=-1) * u_size_factor
                shat_traj = F.softmax(model.decoder_batch(ztraj[:,:model.latent].to(device), batch_id, 's'), dim=-1) * s_size_factor
                uhat_traj = F.softmax(model.decoder_batch(ztraj[:,model.latent:2*model.latent].to(device), batch_id, 'u'), dim=-1) * u_size_factor
        else:
            if model.likelihood_model == 'gaussian':
                shat = model.decoder_s(z[:,:model.latent].to(device))
                uhat = model.decoder_u(z[:,model.latent:2*model.latent].to(device))
                shat_traj = model.decoder_s(ztraj[:,:model.latent].to(device))
                uhat_traj = model.decoder_u(ztraj[:,model.latent:2*model.latent].to(device))
            else:
                shat = F.softmax(model.decoder_s(z[:,:model.latent].to(device)), dim=-1) * s_size_factor 
                uhat = F.softmax(model.decoder_u(z[:,model.latent:2*model.latent].to(device)), dim=-1) * u_size_factor 
                shat_traj = F.softmax(model.decoder_s(ztraj[:,:model.latent].to(device)), dim=-1) * s_size_factor 
                uhat_traj = F.softmax(model.decoder_u(ztraj[:,model.latent:2*model.latent].to(device)), dim=-1) * u_size_factor 
        
        if model.likelihood_model == 'gaussian':
            R2 = R2_score(th.cat((shat, uhat), dim=-1), 
                          th.cat((normed_s, normed_u), dim=-1))
            R2_traj = R2_score(th.cat((shat_traj, uhat_traj), dim=-1), 
                               th.cat((normed_s, normed_u), dim=-1))
            
            test_index = adata.obs.index.isin(adata.uns['index_test'])
            R2_test = R2_score(th.cat((shat[test_index], uhat[test_index]), dim=-1), 
                               th.cat((normed_s[test_index], normed_u[test_index]), dim=-1))
            R2_traj_test = R2_score(th.cat((shat_traj[test_index], uhat_traj[test_index]), dim=-1), 
                                    th.cat((normed_s[test_index], normed_u[test_index]), dim=-1))
        else:
            R2 = R2_score(th.cat((shat, uhat), dim=-1), 
                          th.cat((s, u), dim=-1))
            R2_traj = R2_score(th.cat((shat_traj, uhat_traj), dim=-1), 
                               th.cat((s, u), dim=-1))
            
            test_index = adata.obs.index.isin(adata.uns['index_test'])
            R2_test = R2_score(th.cat((shat[test_index], uhat[test_index]), dim=-1), 
                               th.cat((s[test_index], u[test_index]), dim=-1))
            R2_traj_test = R2_score(th.cat((shat_traj[test_index], uhat_traj[test_index]), dim=-1), 
                                    th.cat((s[test_index], u[test_index]), dim=-1))
            
            
        adata.var['R2'] = R2
        adata.var['R2_traj'] = R2_traj
        adata.var['R2_test'] = R2_test
        adata.var['R2_traj_test'] = R2_traj_test
        adata.obs['latent_time'] = latent_time.cpu().numpy()

        if decoded:
            adata.layers['shat'] = shat.detach().cpu().numpy()
            adata.layers['uhat'] = uhat.detach().cpu().numpy()
            adata.layers['shat_traj'] = shat_traj.detach().cpu().numpy()
            adata.layers['uhat_traj'] = uhat_traj.detach().cpu().numpy()
    
    if save_name != None:
        latent_adata.write(save_name+'.h5ad')
    
    if gene_velocity:
        return latent_adata, adata
    else:
        return latent_adata
