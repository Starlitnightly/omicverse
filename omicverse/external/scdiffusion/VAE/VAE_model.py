import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
import anndata as ad
import scanpy as sc
from typing import List


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether or not to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        first_layer_key = ['network.0.1.weight',
            'network.0.1.bias',
            'network.0.2.weight',
            'network.0.2.bias',
            'network.0.2.running_mean',
            'network.0.2.running_var',
            'network.0.2.num_batches_tracked',
            'network.0.3.weight]',]
        for key in first_layer_key:
            if key in state_dict:  
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)


class Decoder(nn.Module):
    """A class that encapsulates the decoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        last_layer_key = ['network.3.weight',
                'network.3.bias',]
        for key in last_layer_key:
            if key in state_dict:  
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)
        # self.load_state_dict(ckpt["state_dict"])

class VAE(torch.nn.Module):
    """
    VAE base on compositional perturbation autoencoder (CPA)
    """
    def __init__(
        self,
        num_genes,
        device="cuda",
        seed=0,
        loss_ae="gauss",
        decoder_activation="linear",
        hidden_dim=128,
    ):
        super(VAE, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hidden_dim)

        # set models
        self.hidden_dim = [1024,1024,1024]
        self.dropout = 0.0
        self.input_dropout = 0.0
        self.residual = False
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        # losses
        self.loss_autoencoder = nn.MSELoss(reduction='mean')

        self.iteration = 0

        self.to(self.device)

        # optimizers
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
        )
        self.optimizer_autoencoder = torch.optim.AdamW(_parameters, lr=self.hparams["autoencoder_lr"], weight_decay=self.hparams["autoencoder_wd"],)


    def forward(self, genes, return_latent=False, return_decoded=False):
        """
        If return_latent=True, act as encoder only. If return_decoded, genes should 
        be the latent representation and this act as decoder only.
        """
        if return_decoded:
            gene_reconstructions = self.decoder(genes)
            gene_reconstructions = nn.ReLU()(gene_reconstructions)  # only relu when inference
            return gene_reconstructions

        latent_basal = self.encoder(genes)
        if return_latent:
            return latent_basal

        gene_reconstructions = self.decoder(latent_basal)

        return gene_reconstructions



    def set_hparams_(self, hidden_dim):
        """
        Set hyper-parameters to default values or values fixed by user.
        """

        self.hparams = {
            "dim": hidden_dim,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 0.01, 
            "autoencoder_lr": 5e-4, 
        }

        return self.hparams


    def train(self, genes):
        """
        Train VAE.
        """
        genes = genes.to(self.device)
        gene_reconstructions = self.forward(genes)

        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()

        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
        }
