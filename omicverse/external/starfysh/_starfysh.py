from __future__ import print_function

import numpy as np
import pandas as pd
import os
import random

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints, Distribution, Normal, Gamma, Poisson, Dirichlet
from torch.distributions import kl_divergence as kl

# Module import
from .post_analysis import get_z_umap
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
random.seed(0)
np.random.seed(0)

import os
import multiprocessing
import logging

n_cores = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

LOGGER = logging.getLogger('Starfysh')

# TODO:
#  inherit `AVAE` (expr model) w/ `AVAE_PoE` (expr + histology model), update latest PoE model
class AVAE(nn.Module):
    """ 
    Model design
        p(x|z)=f(z)
        p(z|x)~N(0,1)
        q(z|x)~g(x)
    """

    def __init__(
        self,
        adata,
        gene_sig,
        win_loglib,
        alpha_mul=50,
        seed=0,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """
        Auxiliary Variational AutoEncoder (AVAE) - Core model for
        spatial deconvolution without H&E image integration

        Paramters
        ---------
        adata : sc.AnnData
            ST raw expression count (dim: [S, G])

        gene_sig : pd.DataFrame
            Normalized avg. signature expressions for each annotated cell type

        win_loglib : float
            Log-library size smoothed with neighboring spots

        alpha_mul : float (default=50)
            Multiplier of Dirichlet concentration parameter to control
            signature prior's confidence
        """
        super().__init__()
        torch.manual_seed(seed)
        
        self.win_loglib=torch.Tensor(win_loglib)

        self.c_in = adata.shape[1]  # c_in : Num. input features (# input genes)
        self.c_bn = 10  # c_bn : latent number, numbers of bottle-necks
        self.c_hidden = 256
        self.c_kn = gene_sig.shape[1]
        self.eps = 1e-5  # for r.v. w/ numerical constraints
        self.device = device

        self.alpha = torch.ones(self.c_kn) * alpha_mul
        self.alpha = self.alpha.to(device)

        self.qs_logm = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_m = torch.nn.Parameter(torch.randn(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_logv = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)

        self.c_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        self.c_enc_m = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_kn, bias=True),
            nn.BatchNorm1d(self.c_kn, momentum=0.01, eps=0.001),
            nn.Softmax(dim=-1)
        )

        self.l_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        self.l_enc_m = nn.Linear(self.c_hidden, 1)
        self.l_enc_logv = nn.Linear(self.c_hidden, 1)

        # neural network f1 to get the z, p(z|x), f1(x,\phi_1)=[z_m,torch.exp(z_logv)]
        self.z_enc = nn.Sequential(
            # nn.Linear(self.c_in+self.c_kn, self.c_hidden, bias=True),
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )

        self.z_enc_m = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        self.z_enc_logv = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)

        # gene dispersion
        self._px_r = torch.nn.Parameter(torch.randn(self.c_in), requires_grad=True)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.px_hidden_decoder = nn.Sequential(
            nn.Linear(self.c_bn, self.c_hidden, bias=True),
            nn.ReLU(),
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in),
            nn.Softmax(dim=-1)
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def inference(self, x):
        x_n = torch.log1p(x)

        hidden = self.l_enc(x_n)
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)

        hidden = self.c_enc(x_n)
        qc_m = self.c_enc_m(hidden)
        qc = Dirichlet(qc_m * self.alpha + self.eps).rsample()[:,:,None]

        hidden = self.z_enc(x_n)
        qz_m_ct = self.z_enc_m(hidden).reshape([x_n.shape[0], self.c_kn, self.c_bn])
        qz_m_ct = qc * qz_m_ct
        qz_m = qz_m_ct.sum(axis=1)

        qz_logv_ct = self.z_enc_logv(hidden).reshape([x_n.shape[0], self.c_kn, self.c_bn])
        qz_logv_ct = qc * qz_logv_ct
        qz_logv = qz_logv_ct.sum(axis=1)
        qz = self.reparameterize(qz_m, qz_logv)

        qu = self.reparameterize(self.qu_m, self.qu_logv)

        return dict(
            # q(u)
            qu=qu,
            
            # q(c | x)
            qc_m=qc_m,
            qc=qc,
            
            # q(z | c, x)
            qz_m=qz_m,
            qz_m_ct=qz_m_ct,
            qz_logv=qz_logv,
            qz_logv_ct=qz_logv_ct,
            qz=qz,
            
            # q(l | x)
            ql_m=ql_m,
            ql_logv=ql_logv,
            ql=ql,
        )

    def generative(
            self,
            inference_outputs,
            xs_k,
    ):
        qz = inference_outputs['qz']
        ql = inference_outputs['ql']

        hidden = self.px_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        px_rate = torch.exp(ql) * px_scale + self.eps
        pc_p = xs_k + self.eps

        return dict(
            px_rate=px_rate,
            px_r=self.px_r,
            pc_p=pc_p,
            xs_k=xs_k,
        )

    def get_loss(
        self,
        generative_outputs,
        inference_outputs,
        x,
        library,
        device
    ):
        # Variational params
        qs_logm = self.qs_logm
        qu_m, qu_logv, qu = self.qu_m, self.qu_logv, inference_outputs["qu"]
        qc_m, qc = inference_outputs["qc_m"], inference_outputs["qc"]
        qz_m, qz_logv = inference_outputs["qz_m"], inference_outputs["qz_logv"]
        ql_m, ql_logv = inference_outputs["ql_m"], inference_outputs['ql_logv']

        # p(x | z), p(c; \alpha), p(u; \sigma)
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        pc_p = generative_outputs["pc_p"]
        
        pu_m = torch.zeros_like(qu_m)
        pu_std = torch.ones_like(qu_logv) * 10

        # Regularization terms
        kl_divergence_u = kl(
            Normal(qu_m, torch.exp(qu_logv / 2)),
            Normal(pu_m, pu_std)
        ).sum(dim=1).mean()

        kl_divergence_l = kl(
            Normal(ql_m, torch.exp(ql_logv / 2)),
            Normal(library, torch.ones_like(ql_m))
        ).sum(dim=1).mean()

        kl_divergence_c = kl(
            Dirichlet(qc_m * self.alpha),
            Dirichlet(pc_p * self.alpha)
        ).mean()

        pz_m = (qu.unsqueeze(0) * qc).sum(axis=1)
        pz_std = (torch.exp(qs_logm / 2).unsqueeze(0) * qc).sum(axis=1)
        
        kl_divergence_z = kl(
            Normal(qz_m, torch.exp(qz_logv / 2)),
            Normal(pz_m, pz_std)
        ).sum(dim=1).mean()
        
        # Reconstruction term
        reconst_loss = -NegBinom(px_rate, torch.exp(px_r)).log_prob(x).sum(-1).mean()

        loss = reconst_loss.to(device) + \
               kl_divergence_u.to(device) + \
               kl_divergence_z.to(device) + \
               kl_divergence_c.to(device) + \
               kl_divergence_l.to(device)

        return (loss,
                reconst_loss,
                kl_divergence_u,
                kl_divergence_z,
                kl_divergence_c,
                kl_divergence_l
                )

    @property
    def px_r(self):
        return F.softplus(self._px_r) + self.eps


class AVAE_PoE(nn.Module):
    """ 
    Model design:
        p(x|z)=f(z)
        p(z|x)~N(0,1)
        q(z|x)~g(x)
    """

    def __init__(
        self,
        adata,
        gene_sig,
        patch_r,
        win_loglib,
        alpha_mul=50,
        n_img_chan=1,
        seed=0,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """
        Auxiliary Variational AutoEncoder (AVAE) with Joint H&E inference
        - Core model for spatial deconvolution w/ H&E image integration

        Paramters
        ---------
        adata : sc.AnnData
            ST raw expression count (dim: [S, G])

        gene_sig : pd.DataFrame
            Signature gene sets for each annotated cell type

        patch_r : int
            Mini-patch size sampled around each spot from raw H&E image

        win_loglib : float
            Log-library size smoothed with neighboring spots

        alpha_mul : float (default=50)
            Multiplier of Dirichlet concentration parameter to control
            signature prior's confidence

        """
        super().__init__()
        torch.manual_seed(seed)

        self.win_loglib = torch.Tensor(win_loglib)
        self.patch_r = patch_r
        self.c_in = adata.shape[1]  # c_in : Num. input features (# input genes)
        self.nimg_chan = n_img_chan
        self.c_in_img = self.patch_r**2 * 4 * n_img_chan  # c_in_img : (# pixels for the spot's img patch)
        self.c_bn = 10  # c_bn : latent number, numbers of bottleneck
        self.c_hidden = 256
        self.c_kn = gene_sig.shape[1]
        
        self.eps = 1e-5  # for r.v. w/ numerical constraints
        self.alpha = torch.ones(self.c_kn) * alpha_mul
        self.alpha = self.alpha.to(device)

        # --- neural nets for Expression view ---
        self.qs_logm = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_m = torch.nn.Parameter(torch.randn(self.c_kn, self.c_bn), requires_grad=True)
        self.qu_logv = torch.nn.Parameter(torch.zeros(self.c_kn, self.c_bn), requires_grad=True)

        self.c_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        self.c_enc_m = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_kn, bias=True),
            nn.BatchNorm1d(self.c_kn, momentum=0.01, eps=0.001),
            nn.Softmax(dim=-1)
        )

        self.l_enc = nn.Sequential(
            nn.Linear(self.c_in+self.c_in_img, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )
        self.l_enc_m = nn.Linear(self.c_hidden, 1)
        self.l_enc_logv = nn.Linear(self.c_hidden, 1)

        # neural network f1 to get the z, p(z|x), f1(x,\phi_1)=[z_m,torch.exp(z_logv)]
        self.z_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )
        self.z_enc_m = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        self.z_enc_logv = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)

        # gene dispersion
        self._px_r = torch.nn.Parameter(torch.randn(self.c_in), requires_grad=True)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.z_to_hidden_decoder = nn.Sequential(
            nn.Linear(self.c_bn, self.c_hidden, bias=True),
            nn.ReLU(),
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in),
            nn.Softmax(dim=-1)
        )

        # --- neural nets for Histology view ---
        # encoder paths:
        self.img_z_enc = nn.Sequential(
            nn.Linear(self.c_in_img, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )
        self.img_z_enc_m = nn.Linear(self.c_hidden, self.c_bn)
        self.img_z_enc_logv = nn.Linear(self.c_hidden, self.c_bn)

        # decoder paths:
        self.img_z_to_hidden_decoder = nn.Linear(self.c_bn, self.c_hidden, bias=True)
        self.py_mu_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in_img),
            nn.BatchNorm1d(self.c_in_img, momentum=0.01, eps=0.001),
            nn.ReLU()
        )
        self.py_logv_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in_img),
            nn.ReLU()
        )

        # --- PoE view ---
        self.z_to_hidden_poe_decoder = nn.Linear(self.c_bn, self.c_hidden, bias=True)
        self.px_scale_poe_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in),
            nn.ReLU()
        )
        self._px_r_poe = torch.nn.Parameter(torch.randn(self.c_in), requires_grad=True)

        self.py_mu_poe_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in_img),
            nn.BatchNorm1d(self.c_in_img, momentum=0.01, eps=0.001),
            nn.ReLU()
        )
        self.py_logv_poe_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in_img),
            nn.ReLU()
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def inference(self, x, y):
        # q(l | x)
        x_n = torch.log1p(x)  # l is inferred from log(x)
        y_n = torch.log1p(y)
        
        hidden = self.l_enc(torch.concat([x_n,y_n],axis=1))
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)

        # q(c | x)
        hidden = self.c_enc(x_n)
        qc_m = self.c_enc_m(hidden)
        qc = Dirichlet(qc_m * self.alpha + self.eps).rsample()[:,:,None]

        # q(z | c, x)
        hidden = self.z_enc(x_n)
        qz_m_ct = self.z_enc_m(hidden).reshape([x_n.shape[0], self.c_kn, self.c_bn])
        qz_m_ct = qc * qz_m_ct
        qz_m = qz_m_ct.sum(axis=1)

        qz_logv_ct = self.z_enc_logv(hidden).reshape([x_n.shape[0], self.c_kn, self.c_bn])
        qz_logv_ct = qc * qz_logv_ct
        qz_logv = qz_logv_ct.sum(axis=1)
        qz = self.reparameterize(qz_m, qz_logv)

        # q(u), mean-field VI
        qu = self.reparameterize(self.qu_m, self.qu_logv)

        return dict(
            # q(u)
            qu=qu,
            
            # q(c | x)
            qc_m=qc_m,
            qc=qc,
            
            # q(z | x)
            qz_m=qz_m,
            qz_m_ct=qz_m_ct,
            qz_logv=qz_logv,
            qz_logv_ct=qz_logv_ct,
            qz=qz,
            
            # q(l | x)
            ql_m=ql_m,
            ql_logv=ql_logv,
            ql=ql,
        )

    def generative(
        self,
        inference_outputs,
        xs_k,
    ):
        """
        xs_k : torch.Tensor
            Z-normed avg. gene exprs
        """
        qz = inference_outputs['qz']
        ql = inference_outputs['ql']

        hidden = self.z_to_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        px_rate = torch.exp(ql) * px_scale + self.eps
        pc_p = xs_k + self.eps

        return dict(
            px_rate=px_rate,
            px_r=self.px_r,
            px_scale=px_scale,
            pc_p=pc_p,
            xs_k=xs_k,
        )

    def predictor_img(self, y):
        """Inference & generative paths for image view"""
        # --- Inference path ---
        y_n = torch.log1p(y)

        # q(z | y)
        hidden_z = self.img_z_enc(y_n)
        qz_m = self.img_z_enc_m(hidden_z)
        qz_logv = self.img_z_enc_logv(hidden_z)
        qz = self.reparameterize(qz_m, qz_logv)

        # --- Generative path ---
        hidden_y = self.img_z_to_hidden_decoder(qz)
        py_m = self.py_mu_decoder(hidden_y)
        py_logv = self.py_logv_decoder(hidden_y)

        return dict(
            # q(z | y)
            qz_m_img=qz_m,
            qz_logv_img=qz_logv,
            qz_img=qz,

            # p(y | z) (image reconst)
            py_m=py_m,
            py_logv=py_logv
        )

    def generative_img(self, z):
        """Generative path of histology view given z"""
        hidden_y = self.img_z_to_hidden_decoder(z)
        py_m = self.py_mu_decoder(hidden_y)
        py_logv = self.py_logv_decoder(hidden_y)
        return dict(
            py_m=py_m,
            py_logv=py_logv
        )

    def predictor_poe(
        self,
        inference_outputs,
        img_outputs,
    ):
        """Inference & generative paths for Joint view"""

        # Variational params. (expression branch)
        ql = inference_outputs['ql']
        qz_m = inference_outputs['qz_m']
        qz_logv = inference_outputs['qz_logv']

        # Variational params. (img branch)
        qz_m_img = img_outputs['qz_m_img']
        qz_logv_img = img_outputs['qz_logv_img']

        batch, _ = qz_m.shape

        # PoE joint qz
        # --- Joint posterior qz with PoE ---
        qz_var_poe = torch.div(
            1.,
            torch.div(1., torch.exp(qz_logv)) + torch.div(1., torch.exp(qz_logv_img))
        )
        qz_m_poe = qz_var_poe * (
            qz_m * torch.div(1., torch.exp(qz_logv) + self.eps) +
            qz_m_img * torch.div(1., torch.exp(qz_logv_img) + self.eps)
        )
        qz = self.reparameterize(qz_m_poe, torch.log(qz_var_poe))  # Joint posterior

        # PoE joint & view-specific decoders
        hidden = self.z_to_hidden_poe_decoder(qz)

        # p(x | z_poe)
        px_scale = self.px_scale_poe_decoder(hidden)
        px_rate = torch.exp(ql) * px_scale + self.eps

        # p(y | z_poe)
        py_m = self.py_mu_poe_decoder(hidden)
        py_logv = self.py_logv_poe_decoder(hidden)

        return dict(
            # PoE q(z | x, y)
            qz_m=qz_m_poe,
            qz_logv=torch.log1p(qz_var_poe),
            qz=qz,

            # PoE p(x | z, l) & p(y | z)
            px_rate=px_rate,
            px_r=self.px_r_poe,
            py_m=py_m,
            py_logv=py_logv
        )

    def get_loss(
        self,
        generative_outputs,
        inference_outputs,
        img_outputs,
        poe_outputs,
        x,
        library,
        y,
        device
    ):
        lambda_poe = 0.2
        
        # --- Parse variables ---
        # Variational params
        qc_m, qc = inference_outputs["qc_m"], inference_outputs["qc"]
        
        qs_logm = self.qs_logm
        qu_m, qu_logv, qu = self.qu_m, self.qu_logv, inference_outputs["qu"]
        
        qz_m, qz_logv = inference_outputs["qz_m"], inference_outputs["qz_logv"]
        ql_m, ql_logv = inference_outputs["ql_m"], inference_outputs['ql_logv']

        qz_m_img, qz_logv_img = img_outputs['qz_m_img'], img_outputs['qz_logv_img']
        qz_m_poe, qz_logv_poe = poe_outputs['qz_m'], poe_outputs['qz_logv']

        # Generative params
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        pc_p = generative_outputs["pc_p"]

        py_m, py_logv = img_outputs['py_m'], img_outputs['py_logv']

        px_rate_poe = poe_outputs['px_rate']
        px_r_poe = poe_outputs['px_r']
        py_m_poe, py_logv_poe = poe_outputs['py_m'], poe_outputs['py_logv']
        

        # --- Losses ---
        # (1). Joint Loss
        y_n = torch.log1p(y)
        reconst_loss_x_poe = -NegBinom(px_rate_poe, torch.exp(px_r_poe)).log_prob(x).sum(-1).mean()
        reconst_loss_y_poe = -Normal(py_m_poe, torch.exp(py_logv_poe/2)).log_prob(y_n).sum(-1).mean()

        # prior: p(z | c, u)
        pz_m = (qu.unsqueeze(0) * qc).sum(axis=1)
        pz_std = (torch.exp(qs_logm / 2).unsqueeze(0) * qc).sum(axis=1)

        kl_divergence_z_poe = kl(
            Normal(qz_m_poe, torch.exp(qz_logv_poe / 2)),
            Normal(pz_m, pz_std)
        ).sum(dim=1).mean()

        Loss_IBJ = reconst_loss_x_poe + reconst_loss_y_poe + kl_divergence_z_poe.to(device)

        # (2). View-specific losses
        # Expression view
        kl_divergence_u = kl(
            Normal(qu_m, torch.exp(qu_logv / 2)),
            Normal(torch.zeros_like(qu_m), torch.ones_like(qu_m) * 10)
        ).sum(dim=1).mean()

        kl_divergence_z = kl(
            Normal(qz_m, torch.exp(qz_logv / 2)),
            Normal(pz_m, pz_std)
        ).sum(dim=1).mean()

        kl_divergence_l = kl(
            Normal(ql_m, torch.exp(ql_logv / 2)),
            Normal(library, torch.ones_like(ql_m))
        ).sum(dim=1).mean()

        kl_divergence_c = kl(
            Dirichlet(qc_m * self.alpha), # q(c | x; α) = Dir(α * λ(x))
            Dirichlet(pc_p * self.alpha)
        ).mean()

        reconst_loss_x = -NegBinom(px_rate, torch.exp(px_r)).log_prob(x).sum(-1).mean()
        loss_exp = reconst_loss_x.to(device) + \
                   kl_divergence_u.to(device) + \
                   kl_divergence_z.to(device) + \
                   kl_divergence_c.to(device) + \
                   kl_divergence_l.to(device)

        # Image view
        kl_divergence_z_img = kl(
            Normal(qz_m_img, torch.sqrt(torch.exp(qz_logv_img / 2))),
            Normal(pz_m, pz_std)
        ).sum(dim=1).mean()

        reconst_loss_y = -Normal(py_m, torch.exp(py_logv/2)).log_prob(y_n).sum(-1).mean()
        loss_img = reconst_loss_y.to(device) + kl_divergence_z_img.to(device)

        # PoE total loss: Joint Loss + a * \sum(marginal loss)
        Loss_IBM = (loss_exp + loss_img)
        loss = lambda_poe*Loss_IBJ + Loss_IBM
        # loss = self.lambda_poe*Loss_IBJ + Loss_IBM

        return (
            # Total loss
            loss,

            # sum of marginal reconstruction losses
            lambda_poe*(reconst_loss_x_poe+reconst_loss_y_poe) + (reconst_loss_x+reconst_loss_y), 

            # KL divergence
            kl_divergence_u,
            lambda_poe*kl_divergence_z_poe + kl_divergence_z + kl_divergence_z_img,
            kl_divergence_c,
            kl_divergence_l
        )

    @property
    def px_r(self):
        return F.softplus(self._px_r) + self.eps

    @property
    def px_r_poe(self):
        return F.softplus(self._px_r_poe) + self.eps


def train(
    model,
    dataloader,
    device,
    optimizer,
):
    model.train()

    running_loss = 0.0
    running_u = 0.0
    running_z = 0.0
    running_c = 0.0
    running_l = 0.0
    running_reconst = 0.0
    counter = 0
    corr_list = []
    for i, (x, xs_k, x_peri, library_i) in enumerate(dataloader):

        counter += 1
        x = x.float()
        x = x.to(device)
        xs_k = xs_k.to(device)
        x_peri = x_peri.to(device)
        library_i = library_i.to(device)

        inference_outputs = model.inference(x)
        generative_outputs = model.generative(inference_outputs, xs_k)

        # Check for NaNs
        #if torch.isnan(loss) or any(torch.isnan(p).any() for p in model.parameters()):
        if any(torch.isnan(p).any() for p in model.parameters()):
            LOGGER.warning('NaNs detected in model parameters, Skipping current epoch...')
            continue

        (loss,
         reconst_loss,
         kl_divergence_u,
         kl_divergence_z,
         kl_divergence_c,
         kl_divergence_l
         ) = model.get_loss(
            generative_outputs,
            inference_outputs,
            x,
            library_i,
            device
            )

        optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        running_loss += loss.item()
        running_reconst += reconst_loss.item()
        running_u += kl_divergence_u.item()
        running_z += kl_divergence_z.item()
        running_c += kl_divergence_c.item()
        running_l += kl_divergence_l.item()

    train_loss = running_loss / counter
    train_reconst = running_reconst / counter
    train_u = running_u / counter
    train_z = running_z / counter
    train_c = running_c / counter
    train_l = running_l / counter

    return train_loss, train_reconst, train_u, train_z, train_c, train_l, corr_list


def train_poe(
    model,
    dataloader,
    device,
    optimizer,
):
    model.train()

    running_loss = 0.0
    running_z = 0.0
    running_c = 0.0
    running_l = 0.0
    running_u = 0.0
    running_reconst = 0.0
    counter = 0
    corr_list = []
    for i, (x,
            x_peri,
            library_i,
            img,
            data_loc,
            xs_k,
            ) in enumerate(dataloader):
        counter += 1
        mini_batch, _ = x.shape

        x = x.float()
        x = x.to(device)
        x_peri = x_peri.to(device)
        library_i = library_i.to(device)
        xs_k = xs_k.to(device)

        img = img.reshape(mini_batch, -1).float()
        img = img.to(device)

        inference_outputs = model.inference(x,img)  # inference for 1D expr. data
        generative_outputs = model.generative(inference_outputs, xs_k)
        img_outputs = model.predictor_img(img)  # inference & generative for 2D img. data
        poe_outputs = model.predictor_poe(inference_outputs, img_outputs)  # PoE generative outputs

        # Check for NaNs
        if any(torch.isnan(p).any() for p in model.parameters()):
            LOGGER.warning('NaNs detected in model parameters, Skipping current epoch...')
            continue

        (loss,
         reconst_loss,
         kl_divergence_u,
         kl_divergence_z,
         kl_divergence_c,
         kl_divergence_l
         ) = model.get_loss(
            generative_outputs,
            inference_outputs,
            img_outputs,
            poe_outputs,
            x,
            library_i,
            img,
            device
        )

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        running_loss += loss.item()
        running_reconst += reconst_loss.item()
        running_z += kl_divergence_z.item()
        running_c += kl_divergence_c.item()
        running_l += kl_divergence_l.item()
        running_u += kl_divergence_u.item()
    
    train_loss = running_loss / counter
    train_reconst = running_reconst / counter
    train_z = running_z / counter
    train_c = running_c / counter
    train_l = running_l / counter
    train_u = running_u / counter

    return train_loss, train_reconst, train_u, train_z, train_c, train_l, corr_list


# Reference:
# https://github.com/YosefLab/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py
class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """
    arg_constraints = {
        'mu': constraints.greater_than_eq(0),
        'theta': constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(self, mu, theta, eps=1e-10):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        self.mu = mu
        self.theta = theta
        self.eps = eps
        super(NegBinom, self).__init__(validate_args=True)

    def sample(self):
        lambdas = Gamma(
            concentration=self.theta + self.eps,
            rate=(self.theta + self.eps) / (self.mu + self.eps),
        ).rsample()

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll


def model_eval(
    model,
    adata,
    visium_args,
    poe=False,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    adata_ = adata.copy()
    model.eval()
    model = model.to(device)

    x_in = torch.Tensor(adata_.to_df().values).to(device)
    sig_means = torch.Tensor(visium_args.sig_mean_norm.values).to(device)
    anchor_idx = torch.Tensor(visium_args.pure_idx).to(device)

    with torch.no_grad():
        if not poe: 
            inference_outputs = model.inference(x_in)
            generative_outputs = model.generative(inference_outputs, sig_means)
        
        if poe:
            
            img_in = torch.Tensor(visium_args.get_img_patches()).float().to(device)
            inference_outputs = model.inference(x_in,img_in)
            generative_outputs = model.generative(inference_outputs, sig_means)
        
            img_outputs = model.predictor_img(img_in)
            poe_outputs = model.predictor_poe(inference_outputs, img_outputs)

            # Parse image view / PoE inference & generative outputs
            # Save to `inference_outputs` & `generative_outputs`
            for k, v in img_outputs.items():
                if 'q' in k:
                    inference_outputs[k] = v
                else:
                    generative_outputs[k] = v

            for k, v in poe_outputs.items():
                if 'q' in k:
                    inference_outputs[k+'_poe'] = v
                else:
                    generative_outputs[k+'_poe'] = v

    try:
        px = NegBinom(
            mu=generative_outputs["px_rate"],
            theta=torch.exp(generative_outputs["px_r"])
        ).sample().detach().cpu().numpy()
        adata_.obsm['px'] = px
    except ValueError as ve:
        LOGGER.warning('Invalid Gamma distribution parameters `px_rate` or `px_r`, unable to sample inferred p(x | z)')

    # Save inference & generative outputs in adata
    for rv in inference_outputs.keys():
        val = inference_outputs[rv].detach().cpu().numpy().squeeze()
        if "qu" not in rv and "qs" not in rv:
            adata_.obsm[rv] = val
        else:
            adata_.uns[rv] = val

    for rv in generative_outputs.keys():
        try:
            if rv == 'px_r' or rv == 'px_r_poe':
                val = generative_outputs[rv].data.detach().cpu().numpy().squeeze()
                adata_.varm[rv] = val
            else:
                val = generative_outputs[rv].data.detach().cpu().numpy().squeeze()
                adata_.obsm[rv] = val
        except:
            print("rv: {} can't be stored".format(rv))

    qz_umap = get_z_umap(adata_.obsm['qz_m'])
    adata_.obsm['z_umap'] = qz_umap
    return inference_outputs, generative_outputs, adata_


def model_eval_integrate(
    model,
    adata,
    visium_args,
    poe=False,
    device=torch.device('cpu')
):
    """
    Model evaluation for sample integration
    TODO: code refactor
    """
    model.eval()
    model = model.to(device)
    adata_ = adata.copy()
    x_in = torch.Tensor(adata_.to_df().values).to(device)
    sig_means = torch.Tensor(visium_args.sig_mean_norm.values).to(device)
    anchor_idx = torch.Tensor(visium_args.pure_idx).to(device)

    with torch.no_grad():
        if not poe: 
            inference_outputs = model.inference(x_in)
            generative_outputs = model.generative(inference_outputs, sig_means)
        if poe:           
            img_in = torch.Tensor(visium_args.get_img_patches()).float().to(device)
            
            inference_outputs = model.inference(x_in, img_in)
            generative_outputs = model.generative(inference_outputs, sig_means)
        
            img_outputs = model.predictor_img(img_in) if img_in.max() <= 1 else model.predictor_img(img_in/255)
            poe_outputs = model.predictor_poe(inference_outputs, img_outputs)

            # Parse image view / PoE inference & generative outputs
            # Save to `inference_outputs` & `generative_outputs`
            for k, v in img_outputs.items():
                if 'q' in k:
                    inference_outputs[k] = v
                else:
                    generative_outputs[k] = v

            for k, v in poe_outputs.items():
                if 'q' in k:
                    inference_outputs[k+'_poe'] = v
                else:
                    generative_outputs[k+'_poe'] = v

            # TODO: move histology reconstruction to a separate func 
            
            # # Reconst histology prediction 
            # # reconst_img_patches = img_outputs['py_m']            
            # reconst_poe_img_patches = poe_outputs['py_m']
            # # reconst_img_all = {}
            # reconst_poe_img_all = {}
            
            # batch_idx = 0   # img metadata counter for each counter

            # for sample_id in visium_args.adata.obs['sample'].unique():

            #     img_dim = visium_args.img[sample_id].shape
            #     # reconst_img = np.ones((img_dim + (model.nimg_chan,))) * (-1/255)
            #     reconst_poe_img = np.ones((img_dim + (model.nimg_chan,))) * (-1/255)
            
            #     # image_col = img_metadata[i]['map_info']['imagecol']*img_metadata[i]['scalefactor']['tissue_hires_scalef']
            #     # image_row = img_metadata[i]['map_info']['imagerow']*img_metadata[i]['scalefactor']['tissue_hires_scalef']
            #     map_info = visium_args.map_info[adata.obs['sample'] == sample_id]
            #     scalefactor = visium_args.scalefactor[sample_id]
            #     image_col = map_info['imagecol'] * scalefactor['tissue_hires_scalef']
            #     image_row = map_info['imagerow'] * scalefactor['tissue_hires_scalef']

            #     for idx in range(image_col.shape[0]):
                    
            #         patch_y = slice(int(image_row[idx])-model.patch_r, int(image_row[idx])+model.patch_r)
            #         patch_x = slice(int(image_col[idx])-model.patch_r, int(image_col[idx])+model.patch_r)

            #         """
            #         reconst_img[patch_y, patch_x, :] = reconst_img_patches[idx+batch_idx].reshape([
            #             model.patch_r*2,
            #             model.patch_r*2,
            #             model.nimg_chan
            #         ]).cpu().detach().numpy()
            #         """
            #         reconst_poe_img[patch_y, patch_x, :] = reconst_poe_img_patches[idx+batch_idx].reshape([
            #             model.patch_r*2,
            #             model.patch_r*2,
            #             model.nimg_chan
            #         ]).cpu().detatch().numpy()

            #     # reconst_img_all[sample_id] = reconst_img
            #     reconst_poe_img_all[sample_id] = reconst_poe_img
            #     batch_idx += image_col.shape[0]

            # Update reconstructed image
            # adata.uns['reconst_img'] = reconst_poe_img_all
                 
    try:
        px = NegBinom(
            mu=generative_outputs["px_rate"],
            theta=torch.exp(generative_outputs["px_r"])
        ).sample().detach().cpu().numpy()
        adata_.obsm['px'] = px
    except ValueError as ve:
        LOGGER.warning('Invalid Gamma distribution parameters `px_rate` or `px_r`, unable to sample inferred p(x | z)')

    # Save inference & generative outputs in adata
    for rv in inference_outputs.keys():
        val = inference_outputs[rv].detach().cpu().numpy().squeeze()
        if "qu" not in rv and "qs" not in rv:
            adata_.obsm[rv] = val
        else:
            adata_.uns[rv] = val

    for rv in generative_outputs.keys():
        try:
            if rv == 'px_r' or rv == 'reconstruction':  # Posterior avg. znorm signature means
                val = generative_outputs[rv].data.detach().cpu().numpy().squeeze()
                adata_.varm[rv] = val
            else:
                val = generative_outputs[rv].data.detach().cpu().numpy().squeeze()
                adata_.obsm[rv] = val
        except:
            print("rv: {} can't be stored".format(rv))

    return inference_outputs, generative_outputs, adata_


def model_ct_exp(
    model,
    adata,
    visium_args,
    poe = False,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Obtain generative cell-type specific expression in each spot (after model training)
    """
    sig_means = torch.Tensor(visium_args.sig_mean_norm.values).to(device)
    anchor_idx = torch.Tensor(visium_args.pure_idx).to(device)
    x_in = torch.Tensor(adata.to_df().values).to(device)
    if poe: 
        y_in = torch.Tensor(visium_args.get_img_patches()).float().to(device)

    
    model.eval()
    model = model.to(device)
    pred_exprs = {}

    for ct_idx, cell_type in enumerate(adata.uns['cell_types']):
        # Get inference outputs for the given cell type
        
        if poe: 
            inference_outputs = model.inference(x_in,y_in)
        else: 
            inference_outputs = model.inference(x_in)
        inference_outputs['qz'] = inference_outputs['qz_m_ct'][:, ct_idx, :]

        # Get generative outputs
        generative_outputs = model.generative(inference_outputs, sig_means)

        px = NegBinom(
            mu=generative_outputs["px_rate"],
            theta=torch.exp(generative_outputs["px_r"])
        ).sample()
        px = px.detach().cpu().numpy()

        # Save results in adata.obsm
        px_df = pd.DataFrame(px, index=adata.obs_names, columns=adata.var_names)
        pred_exprs[cell_type] = px_df
        adata.obsm[cell_type + '_inferred_exprs'] = px

    return pred_exprs


def model_ct_img(
    model,
    adata,
    visium_args,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Obtain generative cell-type specific images (after model training)
    """
    x_in = torch.Tensor(adata.to_df().values).to(device)
    y_in = torch.Tensor(visium_args.get_img_patches()).float().to(device)

    model.eval()
    model = model.to(device)
    ct_imgs = {}
    ct_imgs_poe = {}

    for ct_idx, cell_type in enumerate(adata.uns['cell_types']):
        # Get inference outputs for the given cell type
        inference_outputs = model.inference(x_in)
        qz_m_ct, qz_logv_ct = inference_outputs['qz_m_ct'][:, ct_idx, :], inference_outputs['qz_logv_ct'][:, ct_idx, :]
        qz_ct = model.reparameterize(qz_m_ct, qz_logv_ct)
        inference_outputs['qz_m'], inference_outputs['qz'] = qz_m_ct, qz_ct

        # Generate cell-type specific low-dim representations (z)
        img_outputs = model.predictor_img(y_in)
        qz_m_ct_img, qz_logv_ct_img = img_outputs['qz_m_ct_img'][:, ct_idx, :], img_outputs['qz_logv_ct_img'][:, ct_idx, :]
        qz_ct_img = model.reparameterize(qz_m_ct_img, qz_logv_ct_img)
        img_outputs['qz_m_img'], img_outputs['qz_img'] = qz_m_ct_img, qz_ct_img
        generative_outputs_img = model.generative_img(qz_ct_img)

        poe_outputs = model.predictor_poe(inference_outputs, img_outputs)

        ct_imgs[cell_type] = generative_outputs_img['py_m'].detach().cpu().numpy()
        ct_imgs_poe[cell_type] = poe_outputs['py_m'].detach().cpu().numpy()

    return ct_imgs, ct_imgs_poe

