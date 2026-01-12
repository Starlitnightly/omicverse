
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import sys

# Module import
from ._starfysh import LOGGER
from .dataloader import IntegrativeDataset, IntegrativePoEDataset
from ._starfysh import AVAE, AVAE_PoE, train, train_poe

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn.functional as F


class VisiumArguments_integrate:
    """
    Loading Visium AnnData, perform preprocessing, library-size smoothing & Anchor spot detection

    Parameters
    ----------
    adata : AnnData
        annotated visium count matrix

    adata_norm : AnnData
        annotated visium count matrix after normalization & log-transform

    gene_sig : pd.DataFrame
        list of signature genes for each cell type. (dim: [S, Cell_type])

    img_metadata : dict
        Spatial information metadata (histology image, coordinates, scalefactor)
    """
    def __init__(
        self,
        adata,
        adata_norm,
        gene_sig,
        img_metadata,
        individual_args,
        **kwargs
    ):

        self.adata = adata
        self.adata_norm = adata_norm
        self.gene_sig = gene_sig
        self.eps = 1e-6
        
        self.params = {
            'sample_id': 'ST', 
            'n_anchors': int(adata.shape[0]),
            'patch_r': 13,
            'signif_level': 3,
            'window_size': 1,
            'n_img_chan': 1
        }
        
        for k, v in kwargs.items():
            if k in self.params.keys():
                self.params[k] = v
                
        
        map_info_temp_all = []
        for i in self.params['sample_id']:
            map_info_temp = img_metadata[i]['map_info'].iloc[:, :4].astype(float)
            map_info_temp_all.append(map_info_temp)
        self.map_info = pd.concat(map_info_temp_all)
        
        img_temp = {}
        for i in self.params['sample_id']:
            img_temp[i] = img_metadata[i]['img']
        self.img = img_temp
        
        self.img_patches = None 
        
        scalefactor_temp = {}
        for i in self.params['sample_id']:
            scalefactor_temp[i] = img_metadata[i]['scalefactor']
        self.scalefactor = scalefactor_temp

        # Update parameters for library smoothing & anchor spot identification
        

        # Center expression for gene score calculation
        adata_scale = self.adata_norm.copy()
        sc.pp.scale(adata_scale)
        
        # Store cell types
        self.adata.uns['cell_types'] = list(self.gene_sig.columns)

        # Filter out signature genes X listed in expression matrix
        LOGGER.info('Subsetting highly variable & signature genes ...')
        self.adata, self.adata_norm = get_adata_wsig(adata, adata_norm, gene_sig)
        self.adata_scale = adata_scale[:, adata.var_names]
        
        # Update spatial metadata
        self._update_spatial_info(self.params['sample_id'])

        # Get smoothed library size
        LOGGER.info('Smoothing library size by taking averaging with neighbor spots...')
        log_lib = np.log1p(self.adata.X.sum(1))
        self.log_lib = np.squeeze(np.asarray(log_lib)) if log_lib.ndim > 1 else log_lib
        
        win_loglib_temp_all = []
        for i in self.params['sample_id']:
            win_loglib_temp = get_windowed_library(self.adata[self.adata.obs['sample']==i],
                                               self.map_info[self.adata.obs['sample']==i],
                                               self.log_lib[self.adata.obs['sample']==i],
                                    window_size=self.params['window_size']
                                               )
            win_loglib_temp_all.append(pd.DataFrame(win_loglib_temp))
        
        self.win_loglib = pd.concat(win_loglib_temp_all)
        self.win_loglib = np.array(self.win_loglib)

        # Retrieve & normalize signature gexp
        LOGGER.info('Retrieving & normalizing signature gene expressions...')
        sig_mean_temp = []
        for i in individual_args.keys():
            sig_mean_temp.append(individual_args[i].sig_mean)
        
        self.sig_mean = pd.concat(sig_mean_temp,axis=0)
        
        sig_mean_norm_temp = []
        for i in individual_args.keys():
            sig_mean_norm_temp.append(individual_args[i].sig_mean_norm)

        self.sig_mean_norm = pd.concat(sig_mean_norm_temp,axis=0)
          
        # Get anchor spots
        LOGGER.info('Identifying anchor spots (highly expression of specific cell-type signatures)...')
        anchor_info = self._compute_anchors()

        self.pure_spots, self.pure_dict, self.pure_idx = anchor_info        
        del self.adata.raw, self.adata_norm.raw 

    def get_adata(self):
        """Return adata after preprocessing & HVG gene selection"""
        return self.adata, self.adata_norm

    def get_anchors(self):
        """Return indices of anchor spots for each cell type"""
        anchors_df = pd.DataFrame.from_dict(self.pure_dict, orient='index')
        anchors_df = anchors_df.transpose()
        
        # Check whether empty anchors detected for any factor
        empty_indices = np.where(
            (~pd.isna(anchors_df)).sum(0) == 0
        )[0]
        
        if len(empty_indices) > 0:
            raise ValueError("Cell type(s) {} has no anchors significantly enriched for its signatures,"
                             "please lower outlier stats `signif_level`".format(
                                 anchors_df.columns[empty_indices].to_list()
                            ))
        
        return anchors_df.applymap(
            lambda x:
            -1 if x is None else np.where(self.adata.obs.index == x)[0][0]
        )

    def get_img_patches(self):
        assert self.img_patches is not None, "Please run Starfysh PoE first"
        return self.img_patches

    def append_factors(self, arche_markers):
        """
        Append list of archetypes (w/ corresponding markers) as additional cell type(s) / state(s) to the `gene_sig`
        """
        self.gene_sig = pd.concat((self.gene_sig, arche_markers), axis=1)

        # Update factor names & anchor spots
        self.adata.uns['cell_types'] = list(self.gene_sig.columns)
        self._update_anchors()
        return None

    def replace_factors(self, factors_to_repl, arche_markers):
        """
        Replace factor(s) with archetypes & their corresponding markers in the `gene_sig`
        """
        if isinstance(factors_to_repl, str):
            assert isinstance(arche_markers, pd.Series),\
                "Please pick only one archetype to replace the factor {}".format(factors_to_repl)
            factors_to_repl = [factors_to_repl]
            archetypes = [arche_markers.name]
        else:
            assert len(factors_to_repl) == len(arche_markers.columns), \
                "Unequal # cell types & archetypes to replace with"
            archetypes = arche_markers.columns

        self.gene_sig.rename(
            columns={
                f: a
                for (f, a) in zip(factors_to_repl, archetypes)
            }, inplace=True
        )
        self.gene_sig[archetypes] = pd.DataFrame(arche_markers)

        # Update factor names & anchor spots
        self.adata.uns['cell_types'] = list(self.gene_sig.columns)
        self._update_anchors()
        return None

    # --- Private methods ---
    def _compute_anchors(self):
        """
        Calculate top `anchor_spots` significantly enriched for given cell type(s)
        determined by gene set scores from signatures
        """
        score_df = self.sig_mean_norm
        signif_level = self.params['signif_level']
        n_anchor = self.params['n_anchors']

        top_expr_spots = (-score_df.values).argsort(axis=0)[:n_anchor, :]
        pure_spots = np.transpose(np.array(score_df.index)[top_expr_spots])

        pure_dict = {
            ct: spot
            for (spot, ct) in zip(pure_spots, score_df.columns)
        }

        pure_indices = np.zeros([score_df.shape[0], 1])
        idx = [np.where(score_df.index == i)[0][0] 
               for i in sorted({x for v in pure_dict.values() for x in v})]
        pure_indices[idx] = 1
        return pure_spots, pure_dict, pure_indices   
    
    def _update_anchors(self):
        """Re-calculate anchor spots given updated gene signatures"""
        self.sig_mean = self._get_sig_mean()
        self.sig_mean_norm = self._calc_gene_scores()
        self.adata.uns['cell_types'] = list(self.gene_sig.columns)

        LOGGER.info('Recalculating anchor spots (highly expression of specific cell-type signatures)...')
        anchor_info = self._compute_anchors()
        self.sig_mean_norm[self.sig_mean_norm < 0] = self.eps
        self.sig_mean_norm.fillna(1/self.sig_mean_norm.shape[1], inplace=True)
        self.pure_spots, self.pure_dict, self.pure_idx = anchor_info
              
    def _get_sig_mean(self):
        sig_mean_expr = pd.DataFrame()
        cnt_df = self.adata_norm.to_df()
                    
        # Calculate avg. signature expressions for each cell type
        for i, cell_type in enumerate(self.gene_sig.columns):
            sigs = np.intersect1d(cnt_df.columns, self.gene_sig.iloc[:, i].astype(str))
            
            if len(sigs) == 0:
                raise ValueError("Empty signatures for {},"
                                 "please double check your `gene_sig` input or set a higher"
                                 "`n_gene` threshold upon dataloading".format(cell_type))
                
            else:
                sig_mean_expr[cell_type] = cnt_df.loc[:, sigs].mean(axis=1)
        
        sig_mean_expr.index = self.adata.obs_names
        sig_mean_expr.columns = self.gene_sig.columns
        return sig_mean_expr

    def _update_spatial_info(self, sample_id):
        """Update paired spatial information to ST adata"""
        # Update image channel count for RGB input (`y`)
        if self.img is not None and self.img[sample_id.iloc[0]].ndim == 3: 
            self.params['n_img_chan'] = 3

        if 'spatial' not in self.adata.uns_keys():
            self.adata.uns['spatial'] = {
                i: {
                    'images': {'hires': (self.img[i] - self.img[i].min()) / (self.img[i].max() - self.img[i].min())},
                    'scalefactors': self.scalefactor
                } for i in sample_id
            }

            self.adata_norm.uns['spatial'] = {
                i: {
                    'images': {'hires': (self.img[i] - self.img[i].min()) / (self.img[i].max() - self.img[i].min())},
                    'scalefactors': self.scalefactor[i]
                } for i in sample_id
            }
            self.adata.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
            self.adata_norm.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values

        # Typecast: spatial coords.
        self.adata_norm.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
        self.adata_norm.obsm['spatial'] = self.adata_norm.obsm['spatial'].astype(np.float32) 
        return None

    def _update_img_patches(self, dl_poe):
        dl_poe.spot_img_stack = np.array(dl_poe.spot_img_stack)
        dl_poe.spot_img_stack = dl_poe.spot_img_stack.reshape(dl_poe.spot_img_stack.shape[0], -1)
        imgs = torch.Tensor(dl_poe.spot_img_stack)
        self.img_patches = imgs
        return None

    def _norm_sig(self):
        # col-norm for each cell type: divided by mean
        gexp = self.sig_mean.apply(lambda x: x / x.mean(), axis=0)
        return gexp
        
    def _calc_gene_scores(self):
        """Calculate gene set enrichment scores for each signature sets"""
        adata = self.adata_scale.copy()
        #adata = self.adata_norm.copy()
        for cell_type in self.gene_sig.columns:
            sig = self.gene_sig[cell_type][~pd.isna(self.gene_sig[cell_type])].to_list()
            sc.tl.score_genes(adata, sig, score_name=cell_type+'_score',use_raw=False)
            
        gsea_df = adata.obs[[cell_type+'_score' for cell_type in self.gene_sig.columns]]
        gsea_df.columns = self.gene_sig.columns
        return gsea_df

    
def get_adata_wsig(adata, adata_norm, gene_sig):
    """
    Select intersection of HVGs from dataset & signature annotations
    """
    hvgs = adata.var_names[adata.var.highly_variable]
    unique_sigs = np.unique(gene_sig.values[~pd.isna(gene_sig)])
    genes_to_keep = np.union1d(
        hvgs,
        np.intersect1d(adata.var_names, unique_sigs)
    )
    return adata[:, genes_to_keep], adata_norm[:, genes_to_keep]


def get_windowed_library(adata_sample, map_info, library, window_size):
    library_n = []

    for i in adata_sample.obs_names:
        window_size = window_size
        dist_arr = np.sqrt(
            (map_info.loc[:, 'array_col'] - map_info.loc[i, 'array_col']) ** 2 +
            (map_info.loc[:, 'array_row'] - map_info.loc[i, 'array_row']) ** 2
        )

        library_n.append(library[dist_arr < window_size].mean())
    library_n = np.array(library_n)
    
    return library_n


def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.kaiming_uniform_(module.weight)

    elif type(module) == nn.BatchNorm1d:
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        
def run_starfysh(
        visium_args,
        n_repeats=3,
        lr=1e-4,
        epochs=100,
        batch_size=32,
        alpha_mul=50,
        poe=False,
        device=torch.device('cpu'),
        verbose=True,
        
):
    """
    Wrapper to run starfysh deconvolution.
    
    Parameters
    ----------
    visium_args : VisiumArguments
        Preprocessed metadata calculated from input visium matrix:
        e.g. mean signature expression, library size, anchor spots, etc.

    n_repeats : int
        Number of restart to run Starfysh

    epochs : int
        Max. number of iterations

    poe : bool
        Whether to perform inference with Poe w/ image integration

    Returns
    -------
    best_model : starfysh.AVAE or starfysh.AVAE_PoE
        Trained Starfysh model with deconvolution results

    loss : np.ndarray
        Training losses
    """
    np.random.seed(0)

    # Loading parameters
    adata = visium_args.adata
    win_loglib = visium_args.win_loglib
    gene_sig, sig_mean_norm = visium_args.gene_sig, visium_args.sig_mean_norm

    models = [None] * n_repeats
    losses = []
    loss_c_list = np.repeat(np.inf, n_repeats)

    if poe:
        dl_func = IntegrativePoEDataset  # dataloader
        train_func = train_poe  # training wrapper
    else:
        dl_func = IntegrativeDataset
        train_func = train

    trainset = dl_func(adata=adata, args=visium_args)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Running Starfysh with multiple starts
    LOGGER.info('Running Starfysh with {} restarts, choose the model with best parameters...'.format(n_repeats))
    for i in range(n_repeats):
        if verbose:
            LOGGER.info(" ===  Restart Starfysh {0} === \n".format(i + 1))
        best_loss_c = np.inf
        
        
        if poe:
            
            model = AVAE_PoE(
                adata=adata,
                gene_sig=sig_mean_norm,
                patch_r=visium_args.params['patch_r'],
                win_loglib=win_loglib,
                alpha_mul=alpha_mul,
                n_img_chan=visium_args.params['n_img_chan']
            )
            # Update patched & flattened image patches
            visium_args._update_img_patches(trainset)
        else:
            model = AVAE(
                adata=adata,
                gene_sig=sig_mean_norm,
                win_loglib=win_loglib,
                alpha_mul=alpha_mul
            )

        model = model.to(device)
        loss_dict = {
            'reconst': [],
            'c': [],
            'u': [],
            'z': [],
            'n': [],
            'tot': []
        }

        # Initialize model params
        #if verbose:
        #    LOGGER.info('Initializing model parameters...')
            
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        for epoch in range(epochs):
            result = train_func(model, trainloader, device, optimizer)
            torch.cuda.empty_cache()

            loss_tot, loss_reconst, loss_u, loss_z, loss_c, loss_n, corr_list = result
            if loss_c < best_loss_c:
                models[i] = model
                best_loss_c = loss_c

            torch.cuda.empty_cache()

            loss_dict['tot'].append(loss_tot)
            loss_dict['reconst'].append(loss_reconst)
            loss_dict['u'].append(loss_u)
            loss_dict['z'].append(loss_z)
            loss_dict['c'].append(loss_c)
            loss_dict['n'].append(loss_n)

            if (epoch + 1) % 10 == 0 and verbose:
                LOGGER.info("Epoch[{}/{}], train_loss: {:.4f}, train_reconst: {:.4f}, train_u: {:.4f},train_z: {:.4f},train_c: {:.4f},train_n: {:.4f}".format(
                    epoch + 1, epochs, loss_tot, loss_reconst, loss_u, loss_z, loss_c, loss_n)
                )
            scheduler.step()

        losses.append(loss_dict)
        loss_c_list[i] = best_loss_c
        if verbose:
            LOGGER.info('Saving the best-performance model...')
            LOGGER.info(" === Finished training === \n")

    idx = np.argmin(loss_c_list)
    best_model = models[idx]
    loss = losses[idx]

    return best_model, loss
