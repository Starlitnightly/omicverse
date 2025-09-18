import torch as th
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint, odeint_adjoint

from .velocity_field import VelocityFieldReg
from .modules import create_encoder, MLP, GCN

from ..utils import normalize, sparse_mx_to_torch_sparse_tensor, batch_jacobian, gaussian_kl, paired_correlation, unique_index
from ..path_regularization import latent_time_path_reg

from torch.distributions.normal import Normal
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial

def create_decoder(observed, latent, decoder_hidden, linear_decoder, batch_correction, batches, shared, decoder_bn, positive_decoder):
    """
    Create decoders
    """
    if positive_decoder:
        if batch_correction:
            if linear_decoder:
                if shared:
                    decoder = []
                    for i in range(batches):
                        decoder.append(nn.Linear(latent, observed))
                    decoder.append(nn.Softplus())
                    decoder=nn.ModuleList(decoder)
                else:
                    decoder_s = []
                    decoder_u = []
                    for i in range(batches):
                        decoder_s.append(nn.Linear(latent, observed))
                        decoder_u.append(nn.Linear(latent, observed))
                    decoder_s.append(nn.Softplus())
                    decoder_u.append(nn.Softplus())
                    decoder_s=nn.ModuleList(decoder_s)
                    decoder_u=nn.ModuleList(decoder_u)
                
            else:
                if shared:
                    decoder = MLP(latent+batches, [decoder_hidden], self.observed, bn=decoder_bn)
                else:
                    decoder_s = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                    decoder_u = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                    
        else:
            if linear_decoder:
                if shared:
                    decoder = nn.Sequential(nn.Linear(latent, observed), nn.Softplus())
                else:
                    decoder_s = nn.Sequential(nn.Linear(latent, observed), nn.Softplus())
                    decoder_u = nn.Sequential(nn.Linear(latent, observed), nn.Softplus())
            else:
                if shared:
                    decoder = nn.Sequential(MLP(latent, [decoder_hidden], observed, bn=decoder_bn), nn.Softplus())
                else:
                    decoder_s = nn.sequential(MLP(latent, [decoder_hidden], observed, bn=decoder_bn), nn.Softplus())
                    decoder_u = nn.Sequential(MLP(latent, [decoder_hidden], observed, bn=decoder_bn), nn.Softplus())    

    else:
        if batch_correction:
            if linear_decoder:
                if shared:
                    decoder = []
                    for i in range(batches):
                        decoder.append(nn.Linear(latent, observed))
                    decoder=nn.ModuleList(decoder)
                else:
                    decoder_s = []
                    decoder_u = []
                    for i in range(batches):
                        decoder_s.append(nn.Linear(latent, observed))
                        decoder_u.append(nn.Linear(latent, observed))
                    decoder_s=nn.ModuleList(decoder_s)
                    decoder_u=nn.ModuleList(decoder_u)
                
            else:
                if shared:
                    decoder = MLP(latent+batches, [decoder_hidden], self.observed, bn=decoder_bn)
                else:
                    decoder_s = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                    decoder_u = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                    
        else:
            if linear_decoder:
                if shared:
                    decoder = nn.Linear(latent, observed)
                else:
                    decoder_s = nn.Linear(latent, observed)
                    decoder_u = nn.Linear(latent, observed)
            else:
                if shared:
                    decoder = MLP(latent, [decoder_hidden], observed, bn=decoder_bn)
                else:
                    decoder_s = MLP(latent, [decoder_hidden], observed, bn=decoder_bn)
                    decoder_u = MLP(latent, [decoder_hidden], observed, bn=decoder_bn) 
    
                
    return decoder_s, decoder_u

class RefineODE(nn.Module):
    """
    LatentVelo VAE model.

    observed: number of genes
    latent_dim: dimension of the latent space
    zr_dim: dimension of the latent regulatory state
    h_dim: dimension of the conditioning variable h
    encoder_hidden: size of the encoder hidden layers
    decoder_hidden: size of the decoder hidden layers
    root_weight: optional weight to include a term in the loss to include a root cell
    num_steps: number of ode solver steps
    encoder_bn: include batch normalization in the encoder
    decoder_bn: include batch normalization in the decoder
    likelihood_model: gaussian or negative-binomial likelihood
    include_time: include time in the latent dynamics
    kl_warmup_steps: warmup steps for kl annealing
    kl_final_weight: weight of the kl term in the loss
    batch_correction: include batch correction True/False
    linear_decoder: use linear or MLP decoder
    linear_splicing: use linear and MLP splicing dynamics
    use_velo_genes: use all genes (False) in the likelihood or restrict to velocity genes (True)
    correlation_reg: use correlation regulatization on gene space
    corr_weight_u: corr(vs, u) weight
    corr_weight_s: corr(vs, -s) weight
    corr_weight_uu: corr(vu, -u) weight
    batches: number of batches for batch correction
    time_reg: include time correlation reg corr(latent t, exp t)
    time_reg_weight: weight for time correlation reg
    shared: share encoder and decoder for spliced/unspliced
    corr_velo_mask: use velocity genes for correlation regularization
    celltype_corr: celltype specific correlization regularization
    celltypes: number of celltypes for celltype correlation regularization
    exp_time: include experimental time in the encoder
    max_sigma_z: restrict value of the latent sigma_z
    latent_reg: impose an increasing relationship on the latetn dynamics between dot{z}_s and u
    velo_reg: instead of regularizing on gene-space by correlation, directly regularize linear splicing velocity
    velo_reg_weight: weight of gene space velocity regularization
    gcn: use graph convolutional network for 
    """
    def __init__(self, ode_model, observed = 2000, latent_dim = 20, zr_dim = 2, h_dim = 2,
                 encoder_hidden = 25, decoder_hidden = 25, 
                 root_weight = 0, num_steps = 100,
                 encoder_bn = False, decoder_bn = False, likelihood_model = 'gaussian',
                 include_time=False, kl_warmup_steps=25, kl_final_weight=1,
                  batch_correction=False, linear_decoder=True, linear_splicing=True, use_velo_genes=False,
                 correlation_reg = True, corr_weight_u = 0.1, corr_weight_s = 0.1, corr_weight_uu = 0., batches = 1, 
                 time_reg = False, time_reg_weight = 0.1, time_reg_decay=0, shared=False, corr_velo_mask=True, celltype_corr=False, celltypes=0, exp_time=False, max_sigma_z = 0,
                 latent_reg = False, velo_reg = False, velo_reg_weight = 0.0001, celltype_velo=False, velo_offset = False, gcn=True, solver = 'dopri5', adjoint = False, path_reg=True,
                 pathreg_weight=100, weighted_mask=True, positive_decoder=False):
        super(RefineODE, self).__init__()
        
        # constant settings
        self.observed = observed
        self.latent = latent_dim
        self.zr_dim = zr_dim
        self.h_dim = h_dim
        self.root_weight = root_weight
        self.num_steps = num_steps
        self.gcn = gcn
        self.likelihood_model = likelihood_model
        self.include_time = include_time
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_final_weight = kl_final_weight
        self.batch_correction = batch_correction
        self.linear_decoder = linear_decoder
        self.linear_splicing = linear_splicing
        self.use_velo_genes = use_velo_genes
        self.correlation_reg = correlation_reg
        self.corr_weight_u = corr_weight_u
        self.corr_weight_s = corr_weight_s
        self.corr_weight_uu = corr_weight_uu
        self.batches = batches
        self.time_reg = time_reg
        self.time_reg_weight = time_reg_weight
        self.time_reg_decay = time_reg_decay
        self.shared = shared
        self.corr_velo_mask = corr_velo_mask
        self.celltype_corr = celltype_corr
        self.celltypes = celltypes
        self.exp_time = exp_time
        self.max_sigma_z = max_sigma_z
        self.latent_reg = latent_reg
        self.velo_reg = velo_reg
        self.velo_reg_weight = velo_reg_weight
        self.celltype_velo = celltype_velo
        self.velo_offset = velo_offset
        self.annot = False
        self.solver = solver
        self.adjoint=adjoint
        self.path_reg=path_reg
        self.pathreg_weight=pathreg_weight
        self.weighted_mask=weighted_mask
        self.positive_decoder=positive_decoder

        if adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint
        
        if not gcn:
            self.batch_correction = True
            batch_correction = True
        
        # encoder networks
        self.encoder_z0_s, self.encoder_z0_u, self.encoder_z_s, self.encoder_z_u = ode_model.encoder_z0_s, ode_model.encoder_z0_u, ode_model.encoder_z_s, ode_model.encoder_z_u 
        
        # h and t encoders
        if self.gcn:
            self.encoder_c = ode_model.encoder_c
            self.encoder_t = ode_model.encoder_t 
        else:
            self.encoder_c = ode_model.encoder_c
            self.encoder_t = ode_model.encoder_t
        
        for param in self.encoder_c.parameters():
            param.requires_grad = True#False
        for param in self.encoder_t.parameters():
            param.requires_grad = True
        for param in self.encoder_z0_s.parameters():
            param.requires_grad = False
        for param in self.encoder_z0_u.parameters():
            param.requires_grad = False
        if not self.batch_correction:
            for param in self.encoder_z_s.parameters():
                param.requires_grad = False
            for param in self.encoder_z_u.parameters():
                param.requires_grad = False
        
        # decoder networks
        self.decoder_s, self.decoder_u = ode_model.decoder_s, ode_model.decoder_u 
        
        # velocity field network
        self.velocity_field = ode_model.velocity_field 

        for param in self.velocity_field.parameters():
            param.requires_grad = True
        
        # learnable decoder variance
        self.theta = ode_model.theta
        self.theta_z = ode_model.theta_z 

        for param in self.decoder_s.parameters():
            param.requires_grad = True #False
        for param in self.decoder_u.parameters():
            param.requires_grad = True #False
        
        self.theta_z.requires_grad = True
        self.theta.requires_grad = False
        
        # regularize with linear splicing dynamics
        if self.velo_reg:
            if self.celltype_velo:
                self.beta = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.celltypes, self.observed) - 1/np.sqrt(self.observed))
                self.gamma = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.celltypes, self.observed) - 1/np.sqrt(self.observed))
                if self.velo_offset:
                    self.offset = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.celltypes, self.observed) - 1/np.sqrt(self.observed))
            else:
                self.beta = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
                self.gamma = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
                if self.velo_offset:
                    self.offset = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
            
        # initial state
        self.initial = ode_model.initial 
        for param in self.initial.parameters():
            param.requires_grad = True

    def decoder_batch(self, x, batch_id, mode = 's'):

        if self.linear_decoder:
            
            if self.shared:
                xhat = (batch_id==0) * self.decoder[0](x)
                for i in range(1, self.batches):
                    xhat = xhat + (batch_id==i)*self.decoder[i](x)
                return xhat
            else:
                if mode == 's':
                    xhat = (batch_id==0) * self.decoder_s[0](x)
                    for i in range(1, self.batches):
                        xhat = xhat + (batch_id==i)*self.decoder_s[i](x)
                    return xhat
                elif mode == 'u':
                    xhat = (batch_id==0) * self.decoder_u[0](x)
                    for i in range(1, self.batches):
                        xhat = xhat + (batch_id==i)*self.decoder_u[i](x)
                    return xhat
                else:
                    print('choose mode u or s')
        else:
            if self.shared:
                return self.decoder(th.cat((x, batch_id), dim=-1))
            else:
                if mode == 's':
                    return self.decoder_s(th.cat((x, batch_id), dim=-1))
                elif mode == 'u':
                    return self.decoder_u(th.cat((x, batch_id), dim=-1))
                else:
                    print('choose mode u or s')

    def decoder_x(self, x):
        return self.decoder_s[0](x)
                    
    def _run_dynamics(self, c, times, test=False):
        
        # set initial state
        h0 = self.initial(th.zeros(c.shape[0], 1).to(device))
        h0 = th.cat((h0, th.zeros(c.shape[0], self.zr_dim).to(device), c), dim=-1)
        
        if test:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).to(device), times), dim=-1), method='dopri8', options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        else:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).to(device), times), dim=-1), method=self.solver,rtol=1e-5, atol=1e-5, options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        ht_full = ht_full[:,1:]
        
        ht = ht_full[...,:2*self.latent+self.zr_dim]
        
        return ht, h0

    def _run_dynamics_grid(self, c):
        
        # set initial state
        h0 = self.initial(th.zeros(c.shape[0], 1).to(device))
        h0 = th.cat((h0, th.zeros(c.shape[0], self.zr_dim).to(device), c), dim=-1)
        ts = th.linspace(0, 1, 25).to(device)
        
        ht_full = odeint(self.velocity_field, h0, ts, method='dopri5',rtol=1e-5, atol=1e-5, options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        
        ht = ht_full[...,:2*self.latent+self.zr_dim]
        
        return ht[...,:self.latent*2], ts
    
    def loss(self, normed_s, s, s_size_factor, mask_s, normed_u, u, u_size_factor, mask_u, velo_genes_mask, adj, special_cells, batch_id = (None, None, None, None), epoch = None):

        if len(special_cells)==2:
            root_cells, terminal_cells = special_cells
        else:
            root_cells=special_cells
        batch_id, batch_onehot, celltype_id, exp_time = batch_id
        
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, adj, batch_id = batch_onehot)
        
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]
        
        orig_index = th.arange(normed_s.shape[0]).to(device)
        
        velo_genes_mask = velo_genes_mask[0]
        
        # get unique time indices for solver
        sort_index, index = unique_index(latent_time)
        
        # select data at these indices
        latent_time = latent_time[sort_index][index]
        z = z[sort_index][index]
        normed_s = normed_s[sort_index][index]
        normed_u = normed_u[sort_index][index]
        mask_s = mask_s[sort_index][index]
        mask_u = mask_u[sort_index][index]
        u = u[sort_index][index]
        s = s[sort_index][index]
        c=c[sort_index][index]
        root_cells = root_cells[sort_index][index]
        orig_index = orig_index[sort_index][index]
        s_size_factor = s_size_factor[sort_index][index]
        u_size_factor = u_size_factor[sort_index][index]
        latent_state = latent_state[sort_index][index]
        latent_mean = latent_mean[sort_index][index]
        latent_logvar = latent_logvar[sort_index][index]
        time_mean = time_mean[sort_index][index]
        time_logvar = time_logvar[sort_index][index]
        if self.batch_correction:
            batch_id = batch_id[sort_index][index]
            batch_onehot = batch_onehot[sort_index][index]
        if self.celltype_corr or self.celltype_velo:
            celltype_id = celltype_id[sort_index][index].long()
        if self.exp_time:
            exp_time = exp_time[sort_index][index]
        
        # zero masks
        if self.weighted_mask:
            mask_s = th.mean(normed_s[mask_s>0])*mask_s
            mask_u = th.mean(normed_u[mask_u>0])*mask_u 
        
        if not self.use_velo_genes:
            velo_genes_mask_ = th.ones_like(velo_genes_mask)
        else:
            velo_genes_mask_ = velo_genes_mask
        
        if not self.corr_velo_mask:
            velo_genes_mask = th.ones_like(velo_genes_mask)
        
        # run dynamics
        ht, h0 = self._run_dynamics(c, latent_time)
        zs, zu, zt = ht[np.arange(z.shape[0]), np.arange(z.shape[0]), :self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), self.latent:2*self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), 2*self.latent:2*self.latent+self.zr_dim]
        
        zs_data, zu_data = z[...,:self.latent], z[...,self.latent:2*self.latent]

        ht_grid, ts = self._run_dynamics_grid(c)
        zs_path, zu_path = ht_grid[...,:self.latent], ht_grid[...,self.latent:2*self.latent]
        
        if self.batch_correction:
            shat_data = self.decoder_batch(zs_data, batch_id, 's')
            uhat_data = self.decoder_batch(zu_data, batch_id, 'u')
            shat = self.decoder_batch(zs, batch_id, 's')
            uhat = self.decoder_batch(zu, batch_id, 'u')
            xhat = th.cat((shat, uhat ), dim=-1)
        else:
            if self.shared:
                shat_data = self.decoder(zs_data)
                uhat_data = self.decoder(zu_data)
                shat = self.decoder(zs)
                uhat = self.decoder(zu)
            else:
                shat_data = self.decoder_s(zs_data)
                uhat_data = self.decoder_u(zu_data)
                shat = self.decoder_s(zs)
                uhat = self.decoder_u(zu)
                xhat = th.cat((self.decoder_s(zs_path), self.decoder_u(zu_path) ), dim=-1)
        
        if self.likelihood_model == 'gaussian':
            
            likelihood_dist_s_z = Normal(shat_data, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = Normal(uhat_data, 1e-4 + F.softplus(self.theta))
            
            likelihood_dist_s_latentz = Normal(shat, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_latentz = Normal(uhat, 1e-4 + F.softplus(self.theta))

            if self.max_sigma_z > 0:
                likelihood_dist_z = Normal(th.cat((zs, zu), dim=-1), 1e-4 + self.max_sigma_z * th.sigmoid(th.cat(2*[self.theta_z], dim=-1)))
            else:
                likelihood_dist_z = Normal(th.cat((zs, zu), dim=-1), 1e-4 + F.softplus(th.cat(2*[self.theta_z], dim=-1))) #

        elif self.likelihood_model == 'nb':
            
            likelihood_dist_s_z = NegativeBinomial(F.softmax(shat_data, dim=-1) * s_size_factor, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = NegativeBinomial(F.softmax(uhat_data, dim=-1) * u_size_factor, 1e-4 + F.softplus(self.theta))
            
            likelihood_dist_s_latentz = NegativeBinomial(F.softmax(shat, dim=-1) * s_size_factor, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_latentz = NegativeBinomial(F.softmax(uhat, dim=-1) * u_size_factor, 1e-4 + F.softplus(self.theta))

            likelihood_dist_z = Normal(th.cat((zs, zu), dim=-1), 1e-4 + F.softplus(th.cat(2*[self.theta_z], dim=-1)))
        
        if len(root_cells.shape) > 1:
            root_cells = root_cells.squeeze(-1)
        
        if self.likelihood_model == 'gaussian':
            
            if self.time_reg_decay > 0 and epoch != None:
                reconstruction_loss = (-(mask_s*velo_genes_mask_*likelihood_dist_s_z.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_z.log_prob(normed_u)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                               (mask_s*velo_genes_mask_*likelihood_dist_s_latentz.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_latentz.log_prob(normed_u)).sum(-1)  +
                            (1 - min(1, epoch/self.time_reg_decay))*self.root_weight*(root_cells * ((latent_time)**2) ) )
            else:
                reconstruction_loss = (-(mask_s*velo_genes_mask_*likelihood_dist_s_z.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_z.log_prob(normed_u)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                               (mask_s*velo_genes_mask_*likelihood_dist_s_latentz.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_latentz.log_prob(normed_u)).sum(-1)  +
                               self.root_weight*(root_cells * ((latent_time)**2) ) )

        elif self.likelihood_model == 'nb':
            reconstruction_loss = (-(mask_s*velo_genes_mask_*likelihood_dist_s_z.log_prob(s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_z.log_prob(u)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                               (mask_s*velo_genes_mask_*likelihood_dist_s_latentz.log_prob(s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_latentz.log_prob(u)).sum(-1)  +
                               self.root_weight*(root_cells * ((h0[...,:2*self.latent] - z)**2).sum(-1) ) )
            

        if self.correlation_reg:
            corr_reg, corr_reg_val = self.corr_reg_func(normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask)
            
        else:
            corr_reg = 0
            corr_reg_val = th.zeros(1).to(device)

        if self.latent_reg:
            latent_reg = self.latent_reg_func(zs, zu, zt, zs_data, zu_data, latent_time)
        else:
            latent_reg = th.zeros(1).to(device)

        if self.velo_reg:
            velo_reg = self.velo_reg_weight*self.velo_reg_func(normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask)
        else:
            velo_reg = th.zeros(1).to(device)
            
        if self.time_reg:
            if self.time_reg_decay > 0 and epoch != None:
                time_reg_ = -self.time_reg_weight * (1 - min(1, epoch/self.time_reg_decay)) * th.mean(paired_correlation(latent_time[:,None], exp_time))
            else:
                time_reg_ = -self.time_reg_weight * th.mean(paired_correlation(latent_time[:,None], exp_time))
        else:
            time_reg_ = th.zeros(1).to(device)

        if self.likelihood_model == 'gaussian':
            validation_ae = th.sum(mask_s*(shat_data - normed_s)**2, dim=-1) + th.sum(mask_u*(uhat_data - normed_u)**2, dim=-1)
            validation_traj = th.sum(mask_s*(shat - normed_s)**2, dim=-1) + th.sum(mask_u*(uhat - normed_u)**2, dim=-1)
            
        elif self.likelihood_model == 'nb':
            validation_ae = th.sum(mask_s*(F.softmax(shat_data, dim=-1) * s_size_factor - s)**2, dim=-1) + th.sum(mask_u*(F.softmax(uhat_data, dim=-1) * u_size_factor - u)**2, dim=-1)
            validation_traj = th.sum(mask_s*(F.softmax(shat, dim=-1) * s_size_factor - s)**2, dim=-1) + th.sum(mask_u*(F.softmax(uhat, dim=-1) * u_size_factor - u)**2, dim=-1)
        
        if epoch != None:
            kl_reg = self.kl_final_weight * min(1, epoch/self.kl_warmup_steps) * (gaussian_kl(latent_mean[:,:self.latent*2], latent_logvar[:,:self.latent*2]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))
        else:
            kl_reg = self.kl_final_weight* (gaussian_kl(latent_mean[:,:self.latent*2], latent_logvar[:,:self.latent*2]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))


        if self.path_reg:
            path_reg_loss = self.pathreg_weight*latent_time_path_reg(ht_grid[...,:2*self.latent], z, ts.unsqueeze(-1), latent_time.unsqueeze(-1))
            
        else:
            path_reg_loss = th.zeros(1).to(device)
            
        return reconstruction_loss + kl_reg + corr_reg + time_reg_ + latent_reg + velo_reg + path_reg_loss, validation_ae, validation_traj, corr_reg_val.unsqueeze(-1) + velo_reg.unsqueeze(-1), orig_index
       
    
    def decoder_jvp(self, z, x, create_graph=True, mode='s'):
        
        v = self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1))
        
        jvp = th.autograd.functional.jvp(lambda y: self.decoder(y), zu, v, create_graph=create_graph)[1]
        
        return jvp 

    def latent_embedding(self, normed_s, normed_u, adj, batch_id = None):
        
        # compute base representation
        if self.batch_correction:
            
            # use neighbourhoods with GCN
            if self.shared:
                zs_params = self.encoder_z0(th.cat((normed_s, batch_id), dim=-1)) 
                zu_params = self.encoder_z0(th.cat((normed_u, batch_id), dim=-1))
            else:
                zs_params = self.encoder_z0_s(th.cat((normed_s, batch_id), dim=-1))
                zu_params = self.encoder_z0_u(th.cat((normed_u, batch_id), dim=-1))
            
        else:
            if self.shared:
                zs0 = self.encoder_z0(normed_s)
                zu0 = self.encoder_z0(normed_u)
                
                # use neighbourhoods with GCN
                zs_params = self.encoder_z(zs0, adj)
                zu_params = self.encoder_z(zu0, adj)
            else:
                zs0 = self.encoder_z0_s(normed_s)
                zu0 = self.encoder_z0_u(normed_u)
                
                # use neighbourhoods with GCN
                zs_params = self.encoder_z_s(zs0, adj)
                zu_params = self.encoder_z_u(zu0, adj)
        
        # sample
        zs_mean, zs_logvar = zs_params[:,:self.latent], zs_params[:,self.latent:]
        zu_mean, zu_logvar = zu_params[:,:self.latent], zu_params[:,self.latent:]
        zs = zs_mean + th.randn_like(zs_mean)*th.exp(0.5*zs_logvar)
        zu = zu_mean + th.randn_like(zu_mean)*th.exp(0.5*zu_logvar)
        
        # compute latent time
        if self.gcn:
            t_params = self.encoder_t(th.cat((zs, zu), dim=-1), adj)
        else:
            t_params = self.encoder_t(th.cat((zs, zu), dim=-1))
        t_mean, t_logvar = t_params[:,0], t_params[:,1]
        latent_time = th.sigmoid(t_mean + th.randn_like(t_mean)*th.exp(0.5*t_logvar))
        
        # compute hidden info
        if self.gcn:
            context_params = self.encoder_c(th.cat((zs, zu), dim=-1), adj)
        else:
            context_params = self.encoder_c(th.cat((zs, zu), dim=-1))
        context_mean, context_logvar = context_params[:,:self.h_dim], context_params[:,self.h_dim:]
        context = context_mean #+ th.randn_like(context_mean)*th.exp(0.5*context_logvar)
        
        # combine spliced/unspliced
        z = th.cat((zs, zu), dim=-1)
        
        return th.cat((z, context), dim=-1), th.cat((zs_mean, zu_mean, context_mean), dim=-1), th.cat((zs_logvar, zu_logvar, context_logvar), dim=-1), latent_time, t_mean, t_logvar

    def gene_likelihood(self, zs, zu, s, u):
        
        shat = self.decoder(zs)
        uhat = self.decoder(zu)
        
        if self.likelihood_model == 'gaussian':
            
            likelihood_dist_s_z = Normal(shat, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = Normal(uhat, 1e-4 + F.softplus(self.theta))
        
        elif self.likelihood_model == 'nb':
            
            likelihood_dist_s_z = NegativeBinomial(F.softplus(shat) * s_size_factor, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = NegativeBinomial(F.softplus(uhat) * u_size_factor, 1e-4 + F.softplus(self.theta))
        
        combined_likelihood = (likelihood_dist_s_z.log_prob(s) + likelihood_dist_u_z.log_prob(u)).mean(0)
        return combined_likelihood[None]
        
    
    def reconstruct_latent(self, normed_s, normed_u, adj, batch_id = None):

        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, adj, batch_id = batch_id)
        
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]
        
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        
        # run dynamics
        ht, h0 = self._run_dynamics(c, unique_times, test=False)

        # select times
        zs, zu, zt = ht[np.arange(z.shape[0]), inverse_indices,:self.latent], ht[np.arange(z.shape[0]), inverse_indices, self.latent:2*self.latent], ht[np.arange(z.shape[0]), inverse_indices, 2*self.latent:2*self.latent+self.zr_dim]
        
        velocity = self.velocity_field.drift(th.cat((z[:,:2*self.latent], zt, c, latent_time[:, None]), dim=-1))
        return th.cat((z,zt), dim=-1), th.cat((zs, zu, zt), dim=-1), velocity, latent_time, c
    
    def batch_func(self, func, inputs, num_outputs, split_size = 500):
      
      outputs = [[] for j in range(num_outputs)]
      
      for i in range(split_size, inputs[0].shape[0] + split_size, split_size):

        inputs_i = []
        for input in inputs:
          if input==None or type(input) == int or type(input) == float or len(input.shape) == 1:
            inputs_i.append(input)
          elif input.shape[0] != input.shape[1]:
            inputs_i.append(input[i-split_size:i])
          else:
            inputs_i.append(sparse_mx_to_torch_sparse_tensor(normalize(input[i-split_size:i, i-split_size:i])).to(device))
        
        outputs_i = func(*inputs_i)
        if type(outputs_i) != tuple:
          outputs_i = tuple((outputs_i,))
        
        if len(outputs_i) != num_outputs:
            print('error, expected different number of outputs')
        
        for j in range(num_outputs):
            outputs[j].append(outputs_i[j].cpu())
            
      outputs_tensor = [None for j in range(num_outputs)]
      for j in range(num_outputs):
          outputs_tensor[j] = th.cat(outputs[j], dim=0)
      
      return tuple(outputs_tensor)



    def cell_trajectories(self, normed_s, normed_u, adj, batch_id = (None, None, None), mode='normal', time_steps = 50):

        batch_id, batch_onehot, celltype_id = batch_id

        # estimate h for conditioning dynamics
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, 
                                                                                                           normed_u, 
                                                                                                               adj,
                                                                                                              batch_id = batch_onehot)
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]

        # choose times
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        times = th.linspace(0, unique_times.max(), time_steps).to(device)

        # run dyanmics
        ht, h0 = self._run_dynamics(c, times[1:], test=False)
        zs, zu, zt = ht[...,:self.latent], ht[..., self.latent:2*self.latent], ht[..., 2*self.latent:2*self.latent+self.zr_dim]
        z_traj = th.cat((zs, zu, zt), dim=-1)

        # decode
        if batch_id == None:
            #x_traj = th.cat((self.decoder_s(zs), self.decoder_u(zu)), dim=-1)
            x0 = th.cat((self.decoder_s(h0[:,:self.latent]), self.decoder_u(h0[:,self.latent:2*self.latent])), dim=-1)
        else:
            #x_traj = th.cat((self.decoder_batch(zs, batch_id[:,None]), self.decoder_batch(zu, batch_id[:,None], 'u')), dim=-1)
            x0 = th.cat((self.decoder_batch(h0[:,:self.latent], batch_id), self.decoder_batch(h0[:,self.latent:2*self.latent], batch_id, 'u')), dim=-1)
            
        #return th.cat((h0[:,None,:2*self.latent+self.zr_dim], z_traj), dim=1), th.cat((x0[:,None],x_traj), dim=1), times[None,:,None].repeat(z_traj.shape[0], 1, 1)
        return th.cat((h0[:,None,:2*self.latent+self.zr_dim], z_traj), dim=1), times[None,:,None].repeat(z_traj.shape[0], 1, 1)


    def corr_reg_func(self, normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask):
        
        if self.batch_correction:
            
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                
                if self.include_time:
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    
                else:
                    
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
                    
        else:
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(self.decoder, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                gene_velocity = th.autograd.functional.jvp(self.decoder_s, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder_s, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1] 
        m_s = (normed_s > 0)[:,velo_genes_mask==1] # normed_s
        m_u = (normed_u > 0)[:,velo_genes_mask==1] # normed_u
        
        if self.celltype_corr:
            
            corr_u = []
            corr_s = []
            corr_uu = []
            for i in range(self.celltypes):
                
                corr_u.append(paired_correlation(gene_velocity[:,velo_genes_mask==1][celltype_id == i], uhat[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], uhat_data[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1][celltype_id == i], normed_u[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], normed_u[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0))
                
                corr_s.append(paired_correlation(gene_velocity[:,velo_genes_mask==1][celltype_id == i], -shat[:,velo_genes_mask==1][celltype_id == i], m_s[celltype_id == i], dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], -shat_data[:,velo_genes_mask==1][celltype_id == i], m_s[celltype_id == i], dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1][celltype_id == i], -normed_s[:,velo_genes_mask==1][celltype_id == i], m_s[celltype_id == i], dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], -normed_s[:,velo_genes_mask==1][celltype_id == i], m_s[celltype_id == i], dim=0))
                
                corr_uu.append(paired_correlation(u_gene_velocity[:,velo_genes_mask==1][celltype_id == i], -uhat[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], -uhat_data[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(u_gene_velocity[:,velo_genes_mask==1][celltype_id == i], -normed_u[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1][celltype_id == i], -normed_u[:,velo_genes_mask==1][celltype_id == i], m_u[celltype_id == i], dim=0))
                
            corr_u = th.stack(corr_u).sum(0)
            corr_s = th.stack(corr_s).sum(0)
            corr_uu = th.stack(corr_uu).sum(0)
            
        else:
                
            corr_u = paired_correlation(gene_velocity[:,velo_genes_mask==1], uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0)
            
            corr_s = paired_correlation(gene_velocity[:,velo_genes_mask==1], -shat[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -shat_data[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0)
            
            corr_uu = paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0)
            
        corr_reg = -self.corr_weight_u * th.mean(corr_u) - self.corr_weight_s * th.mean(corr_s)  - self.corr_weight_uu * th.mean(corr_uu)
        corr_reg_val = -th.mean(corr_u) -th.mean(corr_s)  - th.mean(corr_uu)
        
        return corr_reg, corr_reg_val
    
    def latent_reg_func(self, zs, zu, zr, zs_data, zu_data, latent_time):
        
        u_splicing = []
        s_splicing = []
        s_splicing_data = []
        s_degradation = []
        split_size = 100
        for i in range(split_size, zs.shape[0] + split_size, split_size):
            
            if self.include_time:
                u_splicing.append(batch_jacobian(lambda x: self.velocity_field.unspliced_net(th.cat((x, zr[i-split_size:i], latent_time[i-split_size:i, None]), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
            else:
                u_splicing.append(batch_jacobian(lambda x: self.velocity_field.unspliced_net(th.cat((x, zr[i-split_size:i]), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
                
            s_splicing.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((zs[i-split_size:i], x), dim=-1)),
                                                 zu[i-split_size:i]).permute(1,0,2))

            s_splicing_data.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((zs_data[i-split_size:i], x), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
                
            s_degradation.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((x, zu_data[i-split_size:i]), dim=-1)),
                                                    zs_data[i-split_size:i]).permute(1,0,2))
        
        u_splicing = th.cat(u_splicing, dim=0)
        s_splicing = th.cat(s_splicing, dim=0)
        s_splicing_data = th.cat(s_splicing_data, dim=0)
        s_degradation = th.cat(s_degradation, dim=0)
        
        return 10000*(F.relu(-1*s_splicing).sum(dim=(-1,-2)) + F.relu(-1*s_splicing_data).sum(dim=(-1,-2)))#+ F.relu(u_splicing).sum(dim=(-1,-2)) + F.relu(s_degradation).sum(dim=(-1,-2)))



    def velo_reg_func(self, normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask):
        
        if self.batch_correction:
            
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                
                if self.include_time:
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    
                else:
                    
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
                    
        else:
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(self.decoder, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                gene_velocity = th.autograd.functional.jvp(self.decoder_s, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder_s, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
        
        if self.celltype_velo:
            loss = []
            for i in range(self.celltypes):
                if th.sum(celltype_id == i) > 0:
                    if self.velo_offset:
                        splicing_velo_data = F.softplus(self.beta[i]) * normed_u[celltype_id == i] - F.softplus(self.gamma[i]) * normed_s[celltype_id == i] + F.softplus(self.offset[i])
                        splicing_velo = F.softplus(self.beta[i]) * uhat[celltype_id == i] - F.softplus(self.gamma[i]) * shat[celltype_id == i] + F.softplus(self.offset[i])
                    else:
                        splicing_velo_data = F.softplus(self.beta[i]) * normed_u[celltype_id == i] - F.softplus(self.gamma[i]) * normed_s[celltype_id == i]
                        splicing_velo = F.softplus(self.beta[i]) * uhat[celltype_id == i] - F.softplus(self.gamma[i]) * shat[celltype_id == i]
                    
                    loss.append( th.sum(th.sum((splicing_velo_data - gene_velocity_data[celltype_id == i])[:,velo_genes_mask==1]**2, dim=-1) + th.sum((splicing_velo - gene_velocity[celltype_id == i])[:,velo_genes_mask==1]**2, dim=-1), dim=0)  )
            loss = th.stack(loss).sum(0)/celltype_id.shape[0]
        else:
            if self.velo_offset:
                splicing_velo_data = F.softplus(self.beta) * normed_u - F.softplus(self.gamma) * normed_s  + F.softplus(self.offset)
                splicing_velo = F.softplus(self.beta) * uhat - F.softplus(self.gamma) * shat  + F.softplus(self.offset)
            else:
                splicing_velo_data = F.softplus(self.beta) * normed_u - F.softplus(self.gamma) * normed_s
                splicing_velo = F.softplus(self.beta) * uhat - F.softplus(self.gamma) * shat
            loss = th.sum((splicing_velo_data - gene_velocity_data)[:,velo_genes_mask==1]**2, dim=-1) + th.sum((splicing_velo - gene_velocity)[:,velo_genes_mask==1]**2, dim=-1)
        return loss
