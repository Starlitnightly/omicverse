import os
import warnings
warnings.filterwarnings('ignore')
from typing import Callable, Iterable, Optional, List, Union,Dict, Literal
import collections
import logging
import numpy as np
import anndata
#from math import ceil, floor

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs)
#import lightning.pytorch as pl

import scvi
from scvi import REGISTRY_KEYS#,settings
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.model.base import  BaseModelClass, VAEMixin 
from scvi.nn import one_hot 
from scvi.train import  TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.utils import setup_anndata_dsp
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField
)
from scvi.module import Classifier 
from scvi.utils._docstrings import devices_dsp
from scvi.dataloaders._ann_dataloader import AnnDataLoader


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.
    
    Developed based on SCVI (https://scvi-tools.org/)
    
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    """
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in,
                                n_out,
                                bias=True,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )


    def forward(self, x: torch.Tensor):
        """Forward computation on ``x``.
        
        Developed based on SCVI (https://scvi-tools.org/)
        
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
    
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        x = layer(x)
        return x

class myEncoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Developed based on SCVI (https://scvi-tools.org/)

    Parameters
    ----------
    geneset_len
        The length of the gene set score features
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    var_eps
        Minimum value for the variance;
        used for numerical stability
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    
    
    def __init__(
        self,
        geneset_len: int,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.distribution = 'normal'
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input-geneset_len,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.g_encoder = FCLayers(
            n_in=geneset_len,
            n_out=n_hidden,
            n_layers=n_layers+1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.geneset_len=geneset_len
        self.n_input = n_input
        self.n_hidden = n_hidden
        
        if geneset_len == 0:
            self.mean_encoder = nn.Linear(n_hidden, n_output)
            self.var_encoder = nn.Linear(n_hidden, n_output)
        else:
            self.mean_encoder = nn.Linear(2*n_hidden, n_output)
            self.var_encoder = nn.Linear(2*n_hidden, n_output)
        
        self.var_activation = torch.exp
   

    
    def forward(self, x: torch.Tensor, neighborx: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        
        Developed based on SCVI (https://scvi-tools.org/)
        
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        neighborx
            tensor augmented using cell neighbor graph.

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        
        b = x.shape[0]
        if neighborx != None: # if using cell graph
            neighborx = neighborx.reshape((b,-1))
            q = self.encoder(neighborx)
            
        else:
            q = self.encoder(x)

        if self.geneset_len != 0: # if augmented by geneset
            g = x[:,-self.geneset_len:]
            g = self.g_encoder(g)
            q = torch.cat([q,g],1)
            
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent



# Decoder
class myDecoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Developed based on SCVI (https://scvi-tools.org/)

    Parameters
    ----------
    geneset_len
        The length of the gene set score features
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to use batch norm in layers
    """
    
    
    def __init__(
        self,
        geneset_len: int,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output-geneset_len),
            nn.Softmax(dim=-1),
        )
        
        self.geneset_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, geneset_len),
            nn.Softmax(dim=-1),
        )
        
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
    
        self.geneset_len=geneset_len

    
    def forward(self, z: torch.Tensor, library: torch.Tensor, glibrary: torch.Tensor):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression

        Developed based on SCVI (https://scvi-tools.org/)

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        library
            gene expression data library size
        glibrary
            sum of gene set score

        Returns
        -------
        5-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression and reconstructed gene set score sum

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)

        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        
        if self.geneset_len != 0:
            g_scale = self.geneset_scale_decoder(px)
            g = torch.exp(glibrary) * g_scale
            px_rate = torch.cat([px_rate,g],axis=1)
        else:
            g = 0
        px_r = None
        return px_scale, px_r, px_rate, px_dropout, g



#torch.backends.cudnn.benchmark = True

class myVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of a VAE-GAN based on SCVI :cite:p:`Lopez18`.

    Parameters
    ----------
    variances
        Variances of input features across samples
    bulk
        Bulk data, i.e. the average feature values
    geneset_len
        Gene set scores feature length
    adata
        input dataset
    n_input
        Number of input genes
    countbulkweight
        Weight of bulk loss computed based on count data
    power
        'Temperature' factor of the loss computed
    upperbound
        Upperbound for gene expression values involved in the loss computation
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    use_batch_norm
        Whether to use batch norm in layers.
    """
    def __init__(
        self,
        variances: torch.Tensor,
        bulk: torch.Tensor,
        geneset_len: int,
        adata: anndata.AnnData,
        n_input: int,
        countbulkweight: float = 1,
        power:float=2,
        upperbound:float=99999,
        logbulkweight: float = 0,
        absbulkweight: float = 0,
        abslogbulkweight:float=0,
        corrbulkweight:float = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        
        self.latent_distribution = 'normal'

        self.variances = variances
        self.bulk = bulk
        self.geneset_len = geneset_len
        self._adata=adata
        self.n_input = n_input 
        
        self.logbulkweight = logbulkweight
        self.absbulkweight=absbulkweight
        self.abslogbulkweight=abslogbulkweight
        self.corrbulkweight=corrbulkweight
        self.countbulkweight = countbulkweight
        self.power=power
        self.upperbound=upperbound


        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"


        n_input_encoder = n_input 
        
        self.z_encoder = myEncoder(
            geneset_len,
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = myDecoder(
            geneset_len,
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder
        )

        
    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        
        if 'neighborx' in tensors.keys():
            neighborx =  tensors['neighborx'] 
        else:
            neighborx = None
        
        input_dict = dict(
            x=x,  neighborx=neighborx, geneset_len=self.geneset_len
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        glibrary = inference_outputs["glibrary"]

        input_dict = {
            "z": z,
            "library": library,
            "glibrary":glibrary,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, neighborx, geneset_len, n_samples=1):
        
        x_ = x
        if neighborx != None:
            genelen = neighborx.shape[1]
        else:
            genelen = x.shape[1]
        totallen = x.shape[1]

        if geneset_len>0:
            glibrary = torch.log(x_[:,-geneset_len:].sum(1)).unsqueeze(1)
            library = torch.log(x_[:,:-geneset_len].sum(1)).unsqueeze(1)
        else:
            glibrary = 0
            library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            if (neighborx != None ) and (neighborx.max() > 20):
                neighborx = torch.log(1 + neighborx)
            #neighborx = torch.log(1 + neighborx)
        encoder_input = x_
        qz_m, qz_v, z =  self.z_encoder(encoder_input, neighborx) 
                                 

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
            glibrary = glibrary.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        
        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, library=library,glibrary=glibrary)
        return outputs


    @auto_move_data
    def generative(
        self,
        z,
        library,
        glibrary
    ):
        decoder_input = z 


        px_scale, px_r, px_rate, px_dropout,g = self.decoder(
             decoder_input, library, glibrary
        )

        px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )


    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_scale = generative_outputs["px_scale"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)
        
        kl_divergence_l = torch.tensor(0.0, device=x.device)
        variances = self.variances
        bulk = self.bulk
        reconst_loss, bulk_loss = self.get_reconstruction_loss(x, px_scale, px_rate, px_r, px_dropout,\
                                                    variances,bulk)
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0, device=x.device)
        
        

        if (type(bulk) == type(None)):
            return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local,kl_global=kl_global)
        else:
            return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local,kl_global=bulk_loss)
            #return LossOutput(loss, reconst_loss, kl_local,bulk_loss)# kl_global)

        
    @torch.no_grad()
    def nb_sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
        bound=10.0
    ) -> np.ndarray:
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        dist = NegativeBinomial(mu=px_rate, theta=px_r)
        mdist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        mask = (mdist.mean>bound)
        exprs = dist.mean  # * mask
        
        return exprs.cpu(), mask  

    @torch.no_grad()
    def debugsample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
        bound=10.0,
    ) -> np.ndarray:
 
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]
        px_scale = generative_outputs["px_scale"]
        
        print(px_r)
        print(px_rate)
        print(px_dropout)
        
        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.mean #dist.sample()

        return exprs.cpu()
    
    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1e4,
        bound=None,
    ) -> np.ndarray:
 
        if bound == None:
            bound = library_size / 1e3
    
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]
        px_scale = generative_outputs["px_scale"]

        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.mean #dist.sample()
        
        exprs = exprs.cpu()
        exprs = exprs * (exprs > bound)
        
        return exprs


    def get_reconstruction_loss(self, x, px_scale,px_rate, px_r, px_dropout, variances, bulk) -> torch.Tensor:
        if type(variances) != type(None):
            normv = variances
            vs,idx=normv.sort()
            threshold =  vs[x.shape[1]//2] #normv.mean()
            normv = torch.tensor((normv>threshold)) 
            normv = 0.2*normv + 1
            normv = torch.tensor(normv)
        
        if self.gene_likelihood == "zinb":
            reconst_loss =   ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
                ).log_prob(x)
            if variances != None:
                normv = normv.reshape((1,-1))
                reconst_loss = reconst_loss*normv
            reconst_loss = -reconst_loss.sum(dim=-1)
            
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        
        
        if type(bulk)!= type(None):
            predicted_batch_mean = ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
                ).mean
            
            
            predicted_batch_mean = predicted_batch_mean.mean(axis=0) # average over batch dimension
                                                                     # shape should be (gene) now 
            predicted_batch_mean = predicted_batch_mean.reshape((-1))[:-self.geneset_len]
            
            bulk = bulk.reshape((-1))[:-self.geneset_len]
            
            bulk = bulk.to(predicted_batch_mean.device)
            predicted_batch_mean = predicted_batch_mean[:len(bulk)]
            
            ### expression transformation for bulk loss
            bulk_loss = self.countbulkweight * (predicted_batch_mean - bulk)**self.power + \
                        self.logbulkweight * torch.abs(torch.log(predicted_batch_mean+1) - torch.log(bulk+1)) +  \
                        self.absbulkweight * torch.abs(predicted_batch_mean - bulk) + \
                        self.abslogbulkweight * torch.abs(torch.log(predicted_batch_mean+1) - torch.log(bulk+1)) #+ \
                       # self.corrbulkweight * -(cp*cb).sum()/  ((cb**2).sum())**0.5 *  (((cp**2).sum())**0.5).sum()
            
            bulk_loss = bulk_loss * (predicted_batch_mean < self.upperbound)
            
            bulk_loss = bulk_loss.mean() # average over genes

            reconst_loss = reconst_loss + bulk_loss

        else:
            bulk_loss = 0
        
        return reconst_loss, bulk_loss


    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]
            glibrary = inference_output["glibrary"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            log_prob_sum += p_z + p_x_zl

            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            log_prob_sum -= q_z_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl



logger = logging.getLogger(__name__)

class D(torch.nn.Module):
    
    def __init__(self,indim,hdim=128):    
        super(D,self).__init__()
        self.l1 = torch.nn.Linear(indim,2*hdim)
        self.act1 = torch.nn.LeakyReLU()    
        
        self.l11 = torch.nn.Linear(2*hdim,hdim)
        self.act11 = torch.nn.LeakyReLU()
        
        self.l2 = torch.nn.Linear(hdim,10)
        self.act2 = torch.nn.LeakyReLU()    
        self.l3 = torch.nn.Linear(10,1)
        self.sig = torch.nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
        
    def forward(self,x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l11(x)
        x = self.act11(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        x = self.sig(x)
        return x



class AdversarialTrainingPlan(TrainingPlan):
    def __init__(
        self,
        module: BaseModuleClass,
        lr=1e-3,
        lr2=1e-3,
        kappa = 4.0,
        weight_decay = 1e-6,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier = True,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        if lr2>0:
            #self.lr_factor = 1
            self.lr_patience = 9999
        self.kappa = kappa
        self.n_output_classifier = 1
        self.lr2 = lr2
        self.adversarial_classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False
        
        
    def loss_adversarial_classifier(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
        
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        '''
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
        if optimizer_idx == 0:
            if True: #(batch_idx + 1) % 2 == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()'''
        pass
    

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = self.kappa
        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        #print('current_epoch',self.current_epoch, self.current_epoch%6)
        if (self.current_epoch % 6 < 3) or (self.lr2 == 0):
            
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch, loss_kwargs=self.loss_kwargs
            )

            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:

                px_scale = generative_outputs['px_scale']
                px_r = generative_outputs['px_r']
                px_rate = generative_outputs['px_rate']
                px_dropout = generative_outputs['px_dropout']

                dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
                )
                xh = dist.mean

                valid = torch.ones(xh.size(0), 1)
                valid = valid.type_as(xh)
                fool_loss = self.loss_adversarial_classifier(self.adversarial_classifier(xh), valid) * kappa
                loss = loss + fool_loss 

            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            self.log("fool_loss", fool_loss, on_epoch=True)

            # Note: scvi_loss is frozen, so we can't modify it directly
            # Instead, we use the original scvi_loss for metrics and log the total loss separately
            self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            
            f = open('vaegan_train.txt','a')
            f.write('1, '+str(loss) + ',  '+str(fool_loss) + '\n')
            f.close()
            
            #print(1, loss,fool_loss)
            
        else:
            
            xh = self.module.sample(batch)
            #xh,__ = self.module.nb_sample(batch)
            xh = xh.to(self.module.device)
            x = batch[REGISTRY_KEYS.X_KEY]
            
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(x)
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(x)

            fake_loss = self.loss_adversarial_classifier(self.adversarial_classifier(xh), fake)
            true_loss = self.loss_adversarial_classifier(self.adversarial_classifier(x), valid)
                                                             
            loss = (fake_loss+true_loss)/2
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()
            
            
            f = open('vaegan_train.txt','a')
            f.write('2, '+str(fake_loss) + ',  '+str(true_loss)+ '\n')
            f.close()
            #print(2, fake_loss,true_loss)

            '''
            #### debug
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch, loss_kwargs=self.loss_kwargs
            )

            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:

                px_scale = generative_outputs['px_scale']
                px_r = generative_outputs['px_r']
                px_rate = generative_outputs['px_rate']
                px_dropout = generative_outputs['px_dropout']

                dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout,scale=px_scale
                )
                xh = dist.mean

                valid = torch.ones(xh.size(0), 1)
                valid = valid.type_as(xh)
                fool_loss = self.loss_adversarial_classifier(self.adversarial_classifier(xh), valid) * kappa
                loss = loss + fool_loss 

            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            self.log("fool_loss", fool_loss, on_epoch=True)
            #print(1, loss,fool_loss)'''
            
    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
  
        optimizer1 = torch.optim.Adam(
            params1, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            )

        #if self.adversarial_classifier is not False:
        params2 = filter(
            lambda p: p.requires_grad, self.adversarial_classifier.parameters()
        )
        optimizer2 = torch.optim.Adam(
            params2, lr=self.lr2, eps=0.01, weight_decay=self.weight_decay
        )
        config2 = {"optimizer": optimizer2}

        # bug in pytorch lightning requires this way to return
        opts = [config1.pop("optimizer"), config2["optimizer"]]
        if "lr_scheduler" in config1:
            config1["scheduler"] = config1.pop("lr_scheduler")
            scheds = [config1]
            return opts, scheds
        else:
            return opts

        return config1

'''
def validate_data_split(
    n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            f"With n_samples={n_samples}, train_size={train_size} and "
            f"validation_size={validation_size}, the resulting train set will be empty. Adjust "
            "any of the aforementioned parameters."
        )

    return n_train, n_val'''


'''
class DataSplitter2(pl.LightningDataModule): # the data splutter
    #that drops the last batch to deal with the situation that the last batch is 1
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    """

    data_loader_cls = AnnDataLoader

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.load_sparse_tensor = load_sparse_tensor
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory

        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata_manager.adata.n_obs)

        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        self.val_idx = indices[:n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train) :]

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=True,
            #load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=True,
                #load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=True,
                #load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Converts sparse tensors to dense if necessary."""
        if self.load_sparse_tensor:
            for key, val in batch.items():
                layout = val.layout if isinstance(val, torch.Tensor) else None
                if layout is torch.sparse_csr or layout is torch.sparse_csc:
                    batch[key] = val.to_dense()

        return batch'''
    
    
    
class fastgenerator(
    BaseModelClass,VAEMixin
):

    def __init__(
        self,
        variances,
        bulk,
        geneset_len,
        adata: anndata.AnnData,
        countbulkweight: float = 1,
        power:float = 2.0,
        upperbound:float = 99999,
        logbulkweight: float = 0,
        absbulkweight:float=0,
        abslogbulkweight:float=0,
        corrbulkweight:float=0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        **model_kwargs,
    ):
        super().__init__(adata)
        #super(fastgenerator, self).__init__(adata)

        self.geneset_len = geneset_len
        self.module = myVAE(
            variances,
            bulk,
            geneset_len,
            self._adata,
            countbulkweight = countbulkweight,
            power=power,
            upperbound=upperbound,
            logbulkweight = logbulkweight,
            absbulkweight=absbulkweight,
            abslogbulkweight=abslogbulkweight,
            corrbulkweight=corrbulkweight,
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )
        self.adversarial_classifier = D(indim = self.module.n_input)
        
        
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, gene_likelihood: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            gene_likelihood
        )
        self.init_params_ = self._get_init_params(locals())
        
        
    #@devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        
        
        #if self.adata.n_obs%batch_size == 1:
        #    print('drop last batch with only 1 cell')
        #    data_splitter = DataSplitter2(
        #        self.adata_manager,
        #        train_size=1.0,
        #        validation_size=0,
        #        batch_size=batch_size,
        #    )
        #else:
        #    data_splitter = DataSplitter(
        #        self.adata_manager,
        #        train_size=1.0,
        #        validation_size=0,
        #        batch_size=batch_size,
        #    )
        
        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=1.0,
            validation_size=0,
            batch_size=batch_size,)
            
        training_plan = AdversarialTrainingPlan(self.module, 
                                                adversarial_classifier=self.adversarial_classifier ,
                                                **plan_kwargs)
        

        
        self.training_plan = training_plan
        
        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        # Convert use_gpu to accelerator/devices for PyTorch Lightning compatibility
        if use_gpu is not None:
            if use_gpu is False:
                accelerator = "cpu"
                devices = "auto"
            elif use_gpu is True:
                accelerator = "gpu" if accelerator == "auto" else accelerator
                devices = 1  # Use 1 GPU (will default to GPU 0)
            elif isinstance(use_gpu, int):
                accelerator = "gpu" if accelerator == "auto" else accelerator
                # For PyTorch Lightning, devices should be number of devices or list of device IDs
                if use_gpu == 0:
                    devices = [0]  # Specifically use GPU 0
                else:
                    devices = [use_gpu]  # Use the specified GPU ID
            elif isinstance(use_gpu, str):
                accelerator = "gpu" if accelerator == "auto" else accelerator
                # Extract device number from strings like 'cuda:0' or 'cuda:1'
                if use_gpu.startswith('cuda:'):
                    gpu_id = int(use_gpu.split(':')[1])
                    devices = [gpu_id]  # Use list format for specific GPU
                else:
                    # Try to parse as integer if it's a string number
                    try:
                        gpu_id = int(use_gpu)
                        devices = [gpu_id] if gpu_id >= 0 else 1
                    except ValueError:
                        devices = 1  # Default to 1 GPU

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            accelerator=accelerator,
            max_epochs=max_epochs,
            devices=devices,
            **trainer_kwargs,
        )
        
        return runner()
    
    @classmethod
    #@setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: anndata.AnnData,
        layer: Optional[str] = None,
        **kwargs,
    ):
        
        setup_method_args = cls._get_setup_method_args(**locals())
        
        if ('neighborx' in adata.obsm.keys()) and ('selfw' in adata.obsm.keys()):
            anndata_fields = [
                LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
                ObsmField(
                    'neighborx', 'neighborx'
                ),
                NumericalObsField(
                    'selfw', selfw
                )
            ]
        if ('neighborx' in adata.obsm.keys()) and ('selfw' not in adata.obsm.keys()):
            anndata_fields = [
                LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
                ObsmField(
                    'neighborx', 'neighborx'
                )
            ]
        if ('neighborx' not in adata.obsm.keys()) and ('selfw' in adata.obsm.keys()):
            anndata_fields = [
                LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
                NumericalObsField(
                    'selfw', selfw
                ),
            ]
        if ('neighborx' not in adata.obsm.keys()) and ('selfw' not in adata.obsm.keys()):
            anndata_fields = [
                LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ]
        
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

