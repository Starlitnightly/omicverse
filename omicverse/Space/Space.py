import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv
import scipy.sparse as sp
import logging

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


class Space:

    def __init__(self,single_data,spatial_data,hidden_dims):
        
        self.spatial_data = spatial_data
        self.single_cell_data = single_cell_data

        super(STAGATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        
    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)
    
    def pp_adatas(adata_sc, adata_sp, genes=None, gene_to_lowercase = True):
        sc.pp.filter_genes(adata_sc, min_cells=1)
        sc.pp.filter_genes(adata_sp, min_cells=1)
        if genes is None:
            genes = adata_sc.var.index

        if gene_to_lowercase:
            adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
            adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
            genes = list(g.lower() for g in genes)

        adata_sc.var_names_make_unique()
        adata_sp.var_names_make_unique()
    

    # Refine `marker_genes` so that they are shared by both adatas
        genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

        adata_sc.uns["training_genes"] = genes
        adata_sp.uns["training_genes"] = genes
        logging.info(
            "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
                len(genes)
            )
        )

    # Find overlap genes between two AnnDatas
        overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(overlap_genes)} shared genes.")

        adata_sc.uns["overlap_genes"] = overlap_genes
        adata_sp.uns["overlap_genes"] = overlap_genes
        logging.info(
            "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
                len(overlap_genes)
            )
        )

    # Calculate uniform density prior as 1/number_of_spots
        adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
        logging.info(f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
        )

    # Calculate rna_count_based density prior as % of rna molecule count
        rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
        adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
        logging.info(
            f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
        )
 
    def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):

        try:
            value_counts = adata.obs[cluster_label].value_counts(normalize=True)
        except KeyError as e:
            raise ValueError("Provided label must belong to adata.obs.")
        unique_labels = value_counts.index
        new_obs = pd.DataFrame({cluster_label: unique_labels})
        adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

        X_new = np.empty((len(unique_labels), adata.shape[1]))
        for index, l in enumerate(unique_labels):
            if not scale:
                X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
            else:
                X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)
        adata_ret.X = X_new

        if add_density:
            adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
                lambda i: value_counts[i]
            )

        return adata_ret

    def map_cells_to_space(
        adata_sc,
        adata_sp,
        cv_train_genes=None,
        cluster_label=None,
        mode="cells",
        device="cpu",
        learning_rate=0.1,
        num_epochs=1000,
        scale=True,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        random_state=None,
        verbose=True,
        density_prior='rna_count_based',
    ):

        # check invalid values for arguments
        if lambda_g1 == 0:
            raise ValueError("lambda_g1 cannot be 0.")

        if (type(density_prior) is str) and (
            density_prior not in ["rna_count_based", "uniform", None]
        ):
            raise ValueError("Invalid input for density_prior.")

        if density_prior is not None and (lambda_d == 0 or lambda_d is None):
            lambda_d = 1

        if lambda_d > 0 and density_prior is None:
            raise ValueError("When lambda_d is set, please define the density_prior.")

        if mode not in ["clusters", "cells", "constrained"]:
            raise ValueError('Argument "mode" must be "cells", "clusters" or "constrained')

        if mode == "clusters" and cluster_label is None:
            raise ValueError("A cluster_label must be specified if mode is 'clusters'.")

        if mode == "constrained" and not all([target_count, lambda_f_reg, lambda_count]):
            raise ValueError(
                "target_count, lambda_f_reg and lambda_count must be specified if mode is 'constrained'."
            )

        if mode == "clusters":
            adata_sc = adata_to_cluster_expression(
                adata_sc, cluster_label, scale, add_density=True
            )

        # Check if training_genes key exist/is valid in adatas.uns
        if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
            raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

        if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
            raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

        assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

        # get training_genes
        if cv_train_genes is None:
            training_genes = adata_sc.uns["training_genes"]
        elif cv_train_genes is not None:
            if set(cv_train_genes).issubset set(adata_sc.uns["training_genes"]):
                training_genes = cv_train_genes
            else:
                raise ValueError(
                    "Given training genes list should be subset of two AnnDatas."
                )

        logging.info("Allocate tensors for mapping.")
        # Allocate tensors (AnnData matrix can be sparse or not)

        if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
            S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
        elif isinstance(adata_sc.X, np.ndarray):
            S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
        else:
            X_type = type(adata_sc.X)
            logging.error
