import torch as th
from torch.nn import functional as F
import torch.nn as nn

import numpy as np
import anndata as ad
import scipy as scp

from torchdiffeq import odeint

from .velocity_field import ATACRegVelocityField
from .modules import MLP, GCN, GCNCombined, ATACGCN
from ..utils import normalize, sparse_mx_to_torch_sparse_tensor, batch_jacobian, gaussian_kl, paired_correlation, unique_index

from torch.distributions.normal import Normal
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial

    
class ATACRegModel(nn.Module):
    
    def __init__(self, observed, latent_dim = 20, zr_dim = 2, h_dim = 2,
                 encoder_hidden = 25, decoder_hidden = 25, 
                 root_weight = 0, num_steps = 100,
                 encoder_bn = False, decoder_bn = False, likelihood_model = 'gaussian',
                 include_time=False, kl_warmup_steps=25, kl_final_weight=1,
                 batch_correction=False, linear_decoder=True, use_velo_genes = False,
                 correlation_reg = True, corr_weight_u = 0.1, corr_weight_s = 0.1, corr_weight_a = 0.1,
                 corr_weight_uu = 0.1, shared=False):
        super(ATACRegModel, self).__init__()
        
        # constant settings
        self.observed = observed
        self.latent = latent_dim
        self.zr_dim = zr_dim
        self.h_dim = h_dim
        self.root_weight = root_weight
        self.num_steps = num_steps
        self.gcn = True
        self.likelihood_model = likelihood_model
        self.include_time = include_time
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_final_weight = kl_final_weight
        self.batch_correction = batch_correction
        self.linear_decoder = linear_decoder
        self.use_velo_genes = use_velo_genes
        self.correlation_reg = correlation_reg
        self.corr_weight_u = corr_weight_u
        self.corr_weight_s = corr_weight_s
        self.corr_weight_a = corr_weight_a
        self.corr_weight_uu = corr_weight_uu
        self.shared=shared
        
        # encoder networks
        if batch_correction:
            self.encoder_z0 = MLP(observed+1, [encoder_hidden, encoder_hidden], 2*latent_dim, bn=encoder_bn)
        else:
            if not self.shared:
                self.encoder_z0_s = MLP(observed, [encoder_hidden], 2*latent_dim, bn=encoder_bn)
                self.encoder_z0_u = MLP(observed, [encoder_hidden], 2*latent_dim, bn=encoder_bn)
                self.encoder_z0_atac = MLP(observed, [encoder_hidden], 2*latent_dim, bn=encoder_bn)
            else: 
                self.encoder_z0 = MLP(observed, [encoder_hidden], 2*latent_dim, bn=encoder_bn)
                self.encoder_z0_atac = MLP(observed, [encoder_hidden], 2*latent_dim, bn=encoder_bn)
        
        if not self.batch_correction:
            
            if not self.shared:
                self.encoder_z_s = GCN(2*latent_dim, 2*latent_dim, gcn_layers=2, combine=False)
                self.encoder_z_u = GCN(2*latent_dim, 2*latent_dim, gcn_layers=2, combine=False)
                self.encoder_z_atac = GCN(2*latent_dim, 2*latent_dim, gcn_layers=2, combine=False)
            else:
                self.encoder_z = GCN(2*latent_dim, 2*latent_dim, gcn_layers=2, combine=False)
                self.encoder_z_atac = GCN(2*latent_dim, 2*latent_dim, gcn_layers=2, combine=False)
        
        self.encoder_c = GCNCombined(3*latent_dim, 2*h_dim, gcn_layers=1)
        self.encoder_t = GCNCombined(3*latent_dim, 2, gcn_layers=1)
        
        # decoder network
        if batch_correction:
            if self.linear_decoder:
                self.decoder = nn.Linear(latent_dim+1, observed)
            else:
                self.decoder = MLP(latent_dim+1, [decoder_hidden], observed, bn=decoder_bn)
        else:
            if self.linear_decoder:
                if not self.shared:
                    self.decoder_s = nn.Linear(latent_dim, observed)
                    self.decoder_u = nn.Linear(latent_dim, observed)
                    self.decoder_atac = nn.Linear(latent_dim, observed)
                else:
                    self.decoder = nn.Linear(latent_dim, observed)
                    self.decoder_atac = nn.Linear(latent_dim, observed) 
                
            else:
                self.decoder = MLP(latent_dim, [decoder_hidden], observed, bn=decoder_bn)
        
        # velocity field network
        self.velocity_field = ATACRegVelocityField(latent_dim, h_dim, zr_dim, include_time)
        
        # learnable decoder variance
        self.theta = nn.Parameter(2/np.sqrt(observed) * th.rand(observed) - 1/np.sqrt(observed))
        self.theta_z = nn.Parameter(2/np.sqrt(latent_dim) * th.rand(latent_dim) - 1/np.sqrt(latent_dim))

        # initial state
        self.initial = nn.Linear(1, 3*latent_dim)
            
    def _compute_time(self, z, trajectory, pseudotime = None, epoch = None):
        with th.no_grad():
            
            distance_matrix = th.cdist(z[:,None], trajectory)[:,0]
            index = distance_matrix.argmin(1)
        
        return index, self.time_grid[index]


    def _run_dynamics(self, c, times, test=False):
        
        # set initial state
        h0 = self.initial(th.zeros_like(c[:,0][:,None]))
        h0 = th.cat((h0, th.zeros(c.shape[0], self.h_dim).to(device), c), dim=-1)
        
        if test:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).to(device), times), dim=-1), method='dopri8', options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        else:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).to(device), times), dim=-1), method='dopri5',rtol=1e-5, atol=1e-5, options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        ht_full = ht_full[:,1:]
        
        ht = ht_full[...,:3*self.latent+self.zr_dim]
        ct = ht_full[...,3*self.latent+self.zr_dim:3*self.latent+self.zr_dim+self.h_dim]
        
        return ht, ct, h0
    
    def loss(self, normed_s, s, s_size_factor, mask_s, normed_u, u, u_size_factor, mask_u, normed_a, velo_genes_mask, adj, root_cells, batch_id = None, epoch = None):
      
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, normed_a, adj, batch_id = batch_id)
        
        z = latent_state[:,:self.latent*3]
        c = latent_state[:,self.latent*3:]
        
        orig_index = th.arange(normed_s.shape[0]).to(device)

        # get unique time indices for solver
        sort_index, index = unique_index(latent_time)
        
        latent_time = latent_time[sort_index][index]
        z = z[sort_index][index]
        normed_s = normed_s[sort_index][index]
        normed_u = normed_u[sort_index][index]
        normed_a = normed_a[sort_index][index]
        mask_s = mask_s[sort_index][index]
        mask_u = mask_u[sort_index][index]
        velo_genes_mask = velo_genes_mask[sort_index][index]
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
        
        mask_s = th.mean(normed_s[mask_s>0])*mask_s
        mask_u = th.mean(normed_u[mask_u>0])*mask_u
        mask_a = th.mean(normed_a[normed_a>0])

        if not self.use_velo_genes:
            velo_genes_mask =1
        
        # run dynamics
        ht, ct, h0 = self._run_dynamics(c, latent_time)
        zs, zu, za, zt = ht[np.arange(z.shape[0]), np.arange(z.shape[0]), :self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), self.latent:2*self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), 2*self.latent:3*self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), 3*self.latent:3*self.latent + self.zr_dim]
        hidden = ct[np.arange(z.shape[0]),np.arange(z.shape[0])]
        
        
        zs_data, zu_data, za_data = z[...,:self.latent], z[...,self.latent:2*self.latent], z[...,2*self.latent:3*self.latent]

        if self.batch_correction:
            shat_data = self.decoder(th.cat((zs_data, batch_id), dim=-1))
            uhat_data = self.decoder(th.cat((zu_data, batch_id), dim=-1))
            shat = self.decoder(th.cat((zs, batch_id), dim=-1))
            uhat = self.decoder(th.cat((zu_data, batch_id), dim=-1))
            
        else:
            if not self.shared:
                shat_data = self.decoder_s(zs_data)
                uhat_data = self.decoder_u(zu_data)
                ahat_data = self.decoder_atac(za_data)
                shat = self.decoder_s(zs)
                uhat = self.decoder_u(zu)
                ahat = self.decoder_atac(za)
            else:
                shat_data = self.decoder(zs_data)
                uhat_data = self.decoder(zu_data)
                ahat_data = self.decoder_atac(za_data)
                shat = self.decoder(zs)
                uhat = self.decoder(zu)
                ahat = self.decoder_atac(za)
        
        if self.likelihood_model == 'gaussian':
            
            likelihood_dist_s_z = Normal(shat_data, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = Normal(uhat_data, 1e-4 + F.softplus(self.theta))
            likelihood_dist_a_z = Normal(ahat_data, 1e-4 + F.softplus(self.theta))
            
            likelihood_dist_s_latentz = Normal(shat, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_latentz = Normal(uhat, 1e-4 + F.softplus(self.theta))
            likelihood_dist_a_latentz = Normal(ahat, 1e-4 + F.softplus(self.theta))

        elif self.likelihood_model == 'nb':
            
            likelihood_dist_s_z = NegativeBinomial(F.softplus(shat_data) * s_size_factor, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_z = NegativeBinomial(F.softplus(uhat_data) * u_size_factor, 1e-4 + F.softplus(self.theta))
            
            likelihood_dist_s_latentz = NegativeBinomial(F.softplus(shat) * s_size_factor, 1e-4 + F.softplus(self.theta))
            likelihood_dist_u_latentz = NegativeBinomial(F.softplus(uhat) * u_size_factor, 1e-4 + F.softplus(self.theta))
        
        likelihood_dist_z = Normal(th.cat((zs, zu, za), dim=-1), 1e-4 + F.softplus(th.cat(3*[self.theta_z], dim=-1)))
        
        if len(root_cells.shape) > 1:
            root_cells = root_cells.squeeze(-1)
        
        reconstruction_loss = (-(mask_s*likelihood_dist_s_z.log_prob(s)).sum(-1) - (mask_u*likelihood_dist_u_z.log_prob(u)).sum(-1) - (mask_a*likelihood_dist_a_z.log_prob(normed_a)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                               (mask_s*likelihood_dist_s_latentz.log_prob(s)).sum(-1) - (mask_u*likelihood_dist_u_latentz.log_prob(u)).sum(-1) - (mask_a*likelihood_dist_a_latentz.log_prob(normed_a)).sum(-1)  +
                               self.root_weight*(root_cells * ((h0[...,:3*self.latent] - z)**2).sum(-1) ) )
        
        if self.correlation_reg:

            if self.batch_correction:

                if self.shared:
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
                else:
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                if self.shared:
                    gene_velocity = th.autograd.functional.jvp(self.decoder, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(self.decoder, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, 0*zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, 0*zt), dim=-1)), create_graph=True)[1]

                else:
                    gene_velocity = th.autograd.functional.jvp(self.decoder_s, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(self.decoder_s, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, za, 0*zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, za_data, 0*zt), dim=-1)), create_graph=True)[1]
            
            m_s = (normed_s > 0)[:,velo_genes_mask==1]
            m_u = (normed_u > 0)[:,velo_genes_mask==1]
            m_a = th.ones_like(normed_a)[:,velo_genes_mask==1]
            
            corr_u = (paired_correlation(gene_velocity[:,velo_genes_mask==1], uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0))
            
            corr_s = (paired_correlation(gene_velocity[:,velo_genes_mask==1], -shat[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -shat_data[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0))
            
            corr_a = (paired_correlation(u_gene_velocity[:,velo_genes_mask==1], ahat[:,velo_genes_mask==1], m_a, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], ahat_data[:,velo_genes_mask==1], m_a, dim=0) + paired_correlation(u_gene_velocity[:,velo_genes_mask==1], normed_a[:,velo_genes_mask==1], m_a, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], normed_a[:,velo_genes_mask==1], m_a, dim=0))
            
            corr_uu = (paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0))
            
            corr_reg = -self.corr_weight_u * th.mean(corr_u) - self.corr_weight_s * th.mean(corr_s) - self.corr_weight_a  * th.mean(corr_a) - self.corr_weight_uu  * th.mean(corr_uu)
            
            corr_reg_val = -th.mean(corr_u) -th.mean(corr_s) -th.mean(corr_a) -th.mean(corr_uu)
            
        else:
            corr_reg = 0
            corr_reg_val = 0
        

        if self.likelihood_model == 'gaussian':

            validation_ae = th.sum(mask_s*(shat_data - s)**2, dim=-1) + th.sum(mask_u*(uhat_data - u)**2, dim=-1)
            validation_traj = th.sum(mask_s*(shat - s)**2, dim=-1) + th.sum(mask_u*(uhat - u)**2, dim=-1)
            validation_velo = corr_reg_val
            
        elif self.likelihood_model == 'nb':
            
            validation_ae = th.sum((F.softplus(shat_data) * s_size_factor - s)**2, dim=-1) + th.sum((F.softplus(uhat_data) * u_size_factor - u)**2, dim=-1)
            validation_traj = th.sum((F.softplus(shat) * s_size_factor - s)**2, dim=-1) + th.sum((F.softplus(uhat) * u_size_factor - u)**2, dim=-1)


        if epoch != None:
            kl_reg = self.kl_final_weight * min(1, epoch/self.kl_warmup_steps) * (gaussian_kl(latent_mean[:,:self.latent*3], latent_logvar[:,:self.latent*3]) + 0.*gaussian_kl(latent_mean[:,self.latent*3:], latent_logvar[:,self.latent*3:]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))
        else:
            kl_reg = self.kl_final_weight* (gaussian_kl(latent_mean[:,:self.latent*3], latent_logvar[:,:self.latent*3]) + 0.*gaussian_kl(latent_mean[:,self.latent*3:], latent_logvar[:,self.latent*3:]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))
        
        return reconstruction_loss + kl_reg, validation_ae, validation_traj, corr_reg_val.unsqueeze(-1), orig_index
    
    
    def reconstruct(self, normed_s, normed_u, size_factor, adj):
        
        z, c, latent_time = self.latent_embedding(normed_s, normed_u, adj)
        
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        unique_times = th.cat((th.zeros(1).to(device), unique_times), dim=-1)
        
        # run dynamics
        ht, ct, h0 = self._run_dynamics(c, latent_time)
        
        zs, zu, zt = ht[np.arange(z.shape[0]), inverse_indices,:self.latent], ht[np.arange(z.shape[0]), inverse_indices, self.latent:2*self.latent], ht[np.arange(z.shape[0]), inverse_indices, 2*self.latent:2*self.latent+self.target]
        hidden =  ct[np.arange(normed_s.shape[0]), inverse_indices]
        
        rhoz = self.decode_s(z[:,:self.latent], size_factor)[0]
        rho_latentz = self.decode_s(zs, size_factor)[0]

        
        return rhoz, rho_latentz, latent_time

    def latent_embedding(self, normed_s, normed_u, normed_a, adj, batch_id = None):
        
        # compute base representation
        if self.batch_correction:
            
            # use neighbourhoods with GCN
            zs_params = self.encoder_z0(th.cat((normed_s, batch_id), dim=-1))
            zu_params = self.encoder_z0(th.cat((normed_u, batch_id), dim=-1))
            
        else:
            
            if not self.shared:
                zs0 = self.encoder_z0_s(normed_s)
                zu0 = self.encoder_z0_u(normed_u)    
                za0 = self.encoder_z0_atac(normed_a)
                
                # use neighbourhoods with GCN
                zs_params = self.encoder_z_s(zs0, adj)
                zu_params = self.encoder_z_u(zu0, adj)
                za_params = self.encoder_z_atac(za0, adj)
            
            else:
                zs0 = self.encoder_z0(normed_s)
                zu0 = self.encoder_z0(normed_u)    
                za0 = self.encoder_z0_atac(normed_a)
                
                # use neighbourhoods with GCN
                zs_params = self.encoder_z(zs0, adj)
                zu_params = self.encoder_z(zu0, adj)
                za_params = self.encoder_z_atac(za0, adj)
        
        # sample
        zs_mean, zs_logvar = zs_params[:,:self.latent], zs_params[:,self.latent:]
        zu_mean, zu_logvar = zu_params[:,:self.latent], zu_params[:,self.latent:]
        za_mean, za_logvar = za_params[:,:self.latent], za_params[:,self.latent:]
        zs = zs_mean + th.randn_like(zs_mean)*th.exp(0.5*zs_logvar)
        zu = zu_mean + th.randn_like(zu_mean)*th.exp(0.5*zu_logvar)
        za = za_mean + th.randn_like(za_mean)*th.exp(0.5*za_logvar)
        
        # compute latent time
        t_params = self.encoder_t(th.cat((zs, zu, za), dim=-1), adj)
        t_mean, t_logvar = t_params[:,0], t_params[:,1]
        latent_time = th.sigmoid(t_mean + th.randn_like(t_mean)*th.exp(0.5*t_logvar))
        
        # compute hidden info
        context_params = self.encoder_c(th.cat((zs, zu, za), dim=-1), adj)
        context_mean, context_logvar = context_params[:,:self.h_dim], context_params[:,self.h_dim:]
        context = context_mean #+ th.randn_like(context_mean)*th.exp(0.5*context_logvar)
        
        # combine spliced/unspliced
        z = th.cat((zs, zu, za), dim=-1)
        
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

    
    def cell_trajectories(self, normed_s, normed_u, adj):
        
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, adj)
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]
        
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)

        times = th.linspace(0, unique_times.max(), 25).to(device)
        # run dynamics
        ht, ct, h0 = self._run_dynamics(c, times[1:], test=True)
        zs, zu, za = ht[...,:self.latent], ht[..., self.latent:2*self.latent], ht[..., 2*self.latent:2*self.latent+self.latent]
        hidden =  ct
        
        z_traj = th.cat((zs, zu, za), dim=-1)
        x_traj = th.cat((self.decoder(zs), self.decoder(zu), self.decoder_atac(za)), dim=-1)
        return z_traj, x_traj, times[None,:,None].repeat(z_traj.shape[0], 1, 1)
        
    
    def reconstruct_latent(self, normed_s, normed_u, normed_a, adj, batch_id = None, averaged = False):

        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, normed_a, adj, batch_id = batch_id)
        
        z = latent_state[:,:self.latent*3]
        c = latent_state[:,self.latent*3:]
        
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        
        # run dynamics
        ht, ct, h0 = self._run_dynamics(c, unique_times, test=True)
        zs, zu, za, zt = ht[np.arange(z.shape[0]), inverse_indices,:self.latent], ht[np.arange(z.shape[0]), inverse_indices, self.latent:2*self.latent], ht[np.arange(z.shape[0]), inverse_indices, 2*self.latent:3*self.latent+self.zr_dim], ht[np.arange(z.shape[0]), inverse_indices, 3*self.latent:3*self.latent+self.zr_dim]
        hidden =  ct[np.arange(normed_s.shape[0]), inverse_indices]
        
        if averaged == 1:
            velocity = self.velocity_field.drift(th.cat((zs, zu, za, zt, hidden, latent_time[:, None]), dim=-1))
            return th.cat((zs,zu,za,zt), dim=-1), velocity, latent_time, hidden
        if averaged == 2:
            velocity = self.velocity_field.drift(th.cat((z[:,:self.latent], zu, za, zt, hidden, c, latent_time[:, None]), dim=-1))
            return th.cat((zs,zu,za,zt), dim=-1), velocity, latent_time, hidden
        else:
            velocity = self.velocity_field.drift(th.cat((z[:,:3*self.latent], zt, hidden, latent_time[:, None]), dim=-1))
            return th.cat((z,zt), dim=-1), velocity, latent_time, hidden
    
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
          #print(outputs[j][0].shape, outputs[j][-1].shape)
          
          outputs_tensor[j] = th.cat(outputs[j], dim=0)
      return tuple(outputs_tensor)
