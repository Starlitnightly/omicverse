import os
import cv2
import json
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import median_abs_deviation
from torch.utils.data import DataLoader

import sys



# Module import
from ._starfysh import LOGGER
from .dataloader import VisiumDataset, VisiumPoEDataSet
from ._starfysh import AVAE, AVAE_PoE, train, train_poe


# -------------------
# Model Parameters
# -------------------

class VisiumArguments:
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
        **kwargs
    ):

        self.adata = adata
        self.adata_norm = adata_norm
        self.gene_sig = gene_sig
        self.map_info = img_metadata['map_info'].iloc[:, :4].astype(float)
        self.scalefactor = img_metadata['scalefactor']
        self.img = img_metadata['img']
        self.img_patches = None 
        self.eps = 1e-6

        self.params = {
            'sample_id': 'ST', 
            'n_anchors': int(adata.shape[0]),
            'patch_r': 16,
            'signif_level': 3,
            'window_size': 1,
            'n_img_chan': 1
        }

        # Update parameters for library smoothing & anchor spot identification
        for k, v in kwargs.items():
            if k in self.params.keys():
                self.params[k] = v

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
        
        self.win_loglib = get_windowed_library(self.adata,
                                               self.map_info,
                                               self.log_lib,
                                               window_size=self.params['window_size'])

        # Retrieve & normalize signature gene expressions by 
        # comparing w/ randomized base expression (`sc.tl.score_genes`) 
        LOGGER.info('Retrieving & normalizing signature gene expressions...')
        self.sig_mean = self._get_sig_mean()
        self.sig_mean_norm = self._calc_gene_scores()
       
        # Get anchor spots
        LOGGER.info('Identifying anchor spots (highly expression of specific cell-type signatures)...')
        anchor_info = self._compute_anchors()

        # row-norm; "ReLU" on signature scores for valid dirichlet param
        self.sig_mean_norm[self.sig_mean_norm < 0] = self.eps
        self.sig_mean_norm = self.sig_mean_norm.div(self.sig_mean_norm.sum(1), axis=0)
        self.sig_mean_norm.fillna(1/self.sig_mean_norm.shape[1], inplace=True)

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
                             "please lower outlier stats `signif_level` or try zscore-based signature".format(
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
        if self.img is not None and self.img.ndim == 3: 
            self.params['n_img_chan'] = 3

        if 'spatial' not in self.adata.uns_keys():
            self.adata.uns['spatial'] = {
                sample_id: {
                    'images': {'hires': (self.img - self.img.min()) / (self.img.max() - self.img.min())},
                    'scalefactors': self.scalefactor
                },
            }

            self.adata_norm.uns['spatial'] = {
                sample_id: {
                    'images': {'hires': (self.img - self.img.min()) / (self.img.max() - self.img.min())},
                    'scalefactors': self.scalefactor
                },
            }

            self.adata.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
            self.adata_norm.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values

        # Typecast: spatial coords.
        self.adata_norm.obsm['spatial'] = self.map_info[['imagecol', 'imagerow']].values
        self.adata_norm.obsm['spatial'] = self.adata_norm.obsm['spatial'].astype(np.float32) 
        return None

    def _update_img_patches(self, dl_poe):
        imgs = torch.Tensor(dl_poe.spot_img_stack)
        self.img_patches = imgs.reshape(imgs.shape[0], -1)
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
            sc.tl.score_genes(adata, sig, use_raw=False, score_name=cell_type+'_score')
            
        gsea_df = adata.obs[[cell_type+'_score' for cell_type in self.gene_sig.columns]]
        gsea_df.columns = self.gene_sig.columns
        return gsea_df

# --------------------------------
# Running starfysh with 3-restart
# --------------------------------

def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

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
    seed=0,
    verbose=True
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
    np.random.seed(seed)

    # Loading parameters
    adata = visium_args.adata
    win_loglib = visium_args.win_loglib
    gene_sig, sig_mean_norm = visium_args.gene_sig, visium_args.sig_mean_norm

    models = [None] * n_repeats
    losses = []
    loss_c_list = np.repeat(np.inf, n_repeats)

    if poe:
        dl_func = VisiumPoEDataSet  # dataloader
        train_func = train_poe  # training wrapper
    else:
        dl_func = VisiumDataset
        train_func = train

    trainset = dl_func(adata=adata, args=visium_args)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Running Starfysh with multiple starts
    LOGGER.info('Running Starfysh with {} restarts, choose the model with best parameters...'.format(n_repeats))
    
    count = 0
    while count < n_repeats:
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
            #LOGGER.info('Initializing model parameters...')
            
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        
        try:
            from tqdm import tqdm

            # Initialize the tqdm progress bar
            progress_bar = tqdm(range(epochs), desc="Training Epochs")

            for epoch in progress_bar:
                result = train_func(model, trainloader, device, optimizer)
                torch.cuda.empty_cache()

                loss_tot, loss_reconst, loss_u, loss_z, loss_c, loss_n, corr_list = result
                if loss_c < best_loss_c:
                    models[count] = model
                    best_loss_c = loss_c

                torch.cuda.empty_cache()

                loss_dict['tot'].append(loss_tot)
                loss_dict['reconst'].append(loss_reconst)
                loss_dict['u'].append(loss_u)
                loss_dict['z'].append(loss_z)
                loss_dict['c'].append(loss_c)
                loss_dict['n'].append(loss_n)

                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0 and verbose:
                    log_message = "Epoch[{}/{}], train_loss: {:.4f}, train_reconst: {:.4f}, train_u: {:.4f},train_z: {:.4f},train_c: {:.4f},train_l: {:.4f}".format(
                        epoch + 1, epochs, loss_tot, loss_reconst, loss_u, loss_z, loss_c, loss_n)
                    progress_bar.write(log_message)

                scheduler.step()

            losses.append(loss_dict)
            loss_c_list[count] = best_loss_c

            count += 1
        except ValueError as ve: # Bad model initialization -> numerical instability
            LOGGER.warning(f"Bad model initialization -> numerical instability: {ve}")
            continue
        
        if verbose:
                LOGGER.info('Saving the best-performance model...')
                LOGGER.info(" === Finished training === \n")

    idx = np.argmin(loss_c_list)
    best_model = models[idx]
    loss = losses[idx]

    return best_model, loss


# -------------------
# Preprocessing & IO
# -------------------

def preprocess(
    adata_raw,
    min_perc=None,
    max_perc=None,
    n_top_genes=2000,
    mt_thld=100,
    verbose=True,
    multiple_data=False
):
    """
    Preprocessing ST gexp matrix, remove Ribosomal & Mitochondrial genes

    Parameters
    ----------
    adata_raw : annData
        Spot x Bene raw expression matrix [S x G]

    min_perc : float
        lower-bound percentile of non-zero gexps for filtering spots

    max_perc : float
        upper-bound percentile of non-zero gexps for filtering spots

    n_top_genes: float
        number of the variable genes

    mt_thld : float
        max. percentage of mitochondrial gexps for filtering spots
        with excessive MT expressions

    multiple_data: bool
        whether the study need integrate datasets
    """
    adata = adata_raw.copy()

    if min_perc and max_perc:
        assert 0 < min_perc < max_perc < 100, \
            "Invalid thresholds for cells: {0}, {1}".format(min_perc, max_perc)
        min_counts = np.percentile(adata.obs['total_counts'], min_perc)
        sc.pp.filter_cells(adata, min_counts=min_counts)

    # Remove cells with excessive MT expressions
    # Remove MT & RB genes
    if verbose:
        LOGGER.info('Preprocessing1: delete the mt and rp')
        
    adata.var['mt'] = np.logical_or(
        adata.var_names.str.startswith('MT-'),
        adata.var_names.str.startswith('mt-')
    )
    adata.var['rb'] = adata.var_names.str.startswith(('RP', 'Rp', 'rp'))

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < mt_thld
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

    adata = adata[mask_cell, mask_gene]
    sc.pp.filter_genes(adata, min_cells=1)

    # Normalize & take log-transform
    if verbose:
        LOGGER.info('Preprocessing2: Normalize')
    if multiple_data:
        sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
    else:
        sc.pp.normalize_total(adata, inplace=True)

    # Preprocessing3: Logarithm
    if verbose:
        LOGGER.info('Preprocessing3: Logarithm')
    sc.pp.log1p(adata)

    # Preprocessing4: Find the variable genes
    if verbose:
        LOGGER.info('Preprocessing4: Find the variable genes')
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, inplace=True)

    # Filter corresponding `obs` & `var` in raw-count matrix
    adata_raw = adata_raw[adata.obs_names, adata.var_names]
    adata_raw.var['highly_variable'] = adata.var['highly_variable']
    adata_raw.obs = adata.obs

    return adata_raw, adata


def load_adata(data_folder, sample_id, n_genes, multiple_data=False):
    """
    load visium adata with raw counts, preprocess & extract highly variable genes

    Parameters
    ----------
        data_folder : str
            Root directory of the data

        sample_id : str
            Sample subdirectory under `data_folder`

        n_genes : int
            the number of the gene for training

        multiple_data: bool
            whether the study include multiple datasets

    Returns
    -------
        adata : sc.AnnData
            Processed ST raw counts

        adata_norm : sc.AnnData
            Processed ST normalized & log-transformed data
    """
    has_feature_h5 = os.path.isfile(
        os.path.join(data_folder, sample_id, 'filtered_feature_bc_matrix.h5')
    ) # whether dataset stored in h5 with spatial info.

    if has_feature_h5:
        adata = sc.read_visium(path=os.path.join(data_folder, sample_id), library_id=sample_id)
        adata.var_names_make_unique()
        adata.obs['sample'] = sample_id
    elif sample_id.startswith('simu'): # simulations
        adata = sc.read_csv(os.path.join(data_folder, sample_id, 'counts.st_synth.csv'))
    else:
        filenames = [
            f[:-5] for f in os.listdir(os.path.join(data_folder, sample_id))
            if f[-5:] == '.h5ad'
        ]
        assert len(filenames) == 1, \
            "None or more than `h5ad` file in the data directory," \
            "please contain only 1 target ST file in the given directory"
        adata = sc.read_h5ad(os.path.join(data_folder, sample_id, filenames[0] + '.h5ad'))
        adata.var_names_make_unique()
        adata.obs['sample'] = sample_id

    if '_index' in adata.var.columns:
        adata.var_names = adata.var['_index']
        adata.var_names.name = 'Genes'
        adata.var.drop('_index', axis=1, inplace=True)

    adata, adata_norm = preprocess(adata, n_top_genes=n_genes, multiple_data=multiple_data)
    return adata, adata_norm


def load_signatures(filename, adata):
    """
    load annotated signature gene sets

    Parameters
    ----------
    filename : str
        Signature file

    adata : sc.AnnData
        ST count matrix

    Returns
    -------
    gene_sig : pd.DataFrame
        signatures per cell type / state
    """
    assert os.path.isfile(filename), "Unable to find the signature file"
    gene_sig = pd.read_csv(filename, index_col=0)
    gene_sig = filter_gene_sig(gene_sig, adata.to_df())
    sigs = np.unique(
        gene_sig.apply(
            lambda x:
            pd.unique(x[~pd.isna(x)])
        ).values
    )

    return gene_sig, np.unique(sigs)


def preprocess_img(
    data_path,
    sample_id,
    adata_index,
    rgb_channels=True
):
    """
    Load and preprocess visium paired H&E image & spatial coords

    Parameters
    ----------
    data_path : str
        Root directory of the data

    sample_id : str
        Sample subdirectory under `data_path`

    rgb_channels : bool
        Whether to apply binary color deconvolution to extract 1D `eosin` channel
        Please refer to:
        https://digitalslidearchive.github.io/HistomicsTK/examples/color_deconvolution.html

    Returns
    -------
    img : np.ndarray
        Processed histology image

    map_info : np.ndarray
        Spatial coords of spots (dim: [S, 2])
    """
    from skimage import io
    

    filename = os.path.join(data_path, sample_id, 'spatial', 'tissue_hires_image.png')
    if os.path.isfile(filename):
        if rgb_channels:
            img = io.imread(filename)
            #img = (img-img.min())/(img.max()-img.min())
        
        else:
            try:
                import histomicstk as htk
                
            except ImportError:
                raise ImportError("Please install `histomicstk` package to process histology images\n\
                                Using `pip install histomicstk --find-links https://girder.github.io/large_image_wheels`")
            import histomicstk as htk
            img = io.imread(filename)
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
        
            # Create stain matrix
            stains = ['hematoxylin','eosin', 'null']
            stain_cmap = htk.preprocessing.color_deconvolution.stain_color_map             
            W = np.array([stain_cmap[st] for st in stains]).T

            # Color deconvolution    
            imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(img, W)
            img = imDeconvolved.Stains[:,:,0]   # H-channel

            # Take inverse of H-channel (approx. cell density)      
            img = (img - img.min()) / (img.max()-img.min())       
            img = img.max() - img  
            img = (img*255).astype(np.uint8)
    else:
        img = None

    # Mapping images to location
    f = open(os.path.join(data_path, sample_id, 'spatial', 'scalefactors_json.json', ))
    json_info = json.load(f)
    f.close()

    tissue_position_list = pd.read_csv(os.path.join(data_path, sample_id, 'spatial', 'tissue_positions_list.csv'), header=None, index_col=0)
    tissue_position_list = tissue_position_list.loc[adata_index, :]
    map_info = tissue_position_list.iloc[:, -4:-2]
    map_info.columns = ['array_row', 'array_col']
    map_info.loc[:, 'imagerow'] = tissue_position_list.iloc[:, -2]
    map_info.loc[:, 'imagecol'] = tissue_position_list.iloc[:, -1]
    map_info.loc[:, 'sample'] = sample_id

    return {
        'img': img,
        'map_info': map_info,
        'scalefactor': json_info
    }


def get_adata_wsig(adata, adata_norm, gene_sig):
    """
    Select intersection of HVGs from dataset & signature annotations
    """
    # TODO: in-place operators for `adata`
    if 'highly_variable' in adata.var.columns:
        hvgs = adata.var_names[adata.var.highly_variable]
    elif 'space_variable_features' in adata.var.columns:
        hvgs = adata.var_names[adata.var.space_variable_features]
    elif 'highly_variable_features' in adata.var.columns:
        hvgs = adata.var_names[adata.var.highly_variable_features]
    else:
        raise ValueError("No highly variable genes found in adata.var.columns")
    unique_sigs = np.unique(gene_sig.values[~pd.isna(gene_sig)])
    genes_to_keep = np.union1d(
        hvgs,
        np.intersect1d(adata.var_names, unique_sigs)
    )
    return adata[:, genes_to_keep], adata_norm[:, genes_to_keep]


def filter_gene_sig(gene_sig, adata_df):
    for i in range(gene_sig.shape[0]):
        for j in range(gene_sig.shape[1]):
            gene = gene_sig.iloc[i, j]
            if gene in adata_df.columns:
                # We don't filter signature genes based on expression level (prev: threshold=20)
                if adata_df.loc[:, gene].sum() < 0:
                    gene_sig.iloc[i, j] = 'NaN'
    return gene_sig


def get_umap(adata_sample, display=False):
    sc.tl.pca(adata_sample, svd_solver='arpack')
    sc.pp.neighbors(adata_sample, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_sample, min_dist=0.2)
    if display:
        sc.pl.umap(adata_sample)
    umap_plot = pd.DataFrame(adata_sample.obsm['X_umap'],
                             columns=['umap1', 'umap2'],
                             index=adata_sample.obs_names)
    return umap_plot


def get_simu_map_info(umap_plot):
    map_info = []
    map_info = [-umap_plot['umap2'] * 10, umap_plot['umap1'] * 10]
    map_info = pd.DataFrame(np.transpose(map_info),
                            columns=['array_row', 'array_col'],
                            index=umap_plot.index)
    return map_info


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


def append_sigs(gene_sig, factor, sigs, n_genes=10):
    """
    Append list of genes to a given cell type as additional signatures or
    add novel cell type / states & their signatures
    """
    assert len(sigs) > 0, "Signature list must have positive length"
    gene_sig_new = gene_sig.copy()

    if not isinstance(sigs, list):
        sigs = sigs.to_list()
    if n_genes < len(sigs):
        sigs = sigs[:n_genes]

    markers = set([i for i in gene_sig[factor] if str(i) != 'nan'] +
                  [i for i in sigs if str(i) != 'nan'])
    nrow_diff = int(np.abs(len(markers)-gene_sig_new.shape[0]))
    if len(markers) > gene_sig_new.shape[0]:
        df_dummy = pd.DataFrame([[np.nan] * gene_sig.shape[1]] * nrow_diff, 
                                columns=gene_sig_new.columns)
        gene_sig_new = pd.concat([gene_sig_new, df_dummy], ignore_index=True)
    else:
        markers = list(markers) + [np.nan]*nrow_diff
    gene_sig_new[factor] = list(markers)

    return gene_sig_new


def refine_anchors(
    visium_args,
    aa_model,
    anchor_threshold=0.1,
    n_genes=10
):
    """
    Refine anchor spots & marker genes with archetypal analysis. We append DEGs
    computed from archetypes to their best-matched anchors followed by re-computing
    new anchor spots

    Parameters
    ----------
    visium_args : VisiumArgument
        Default parameter set for Starfysh upon dataloading

    aa_model : ArchetypalAnalysis
        Pre-computed archetype object

    anchor_threshold : float
        Top percent of anchor spots per cell-type
        for archetypal mapping

    n_genes : int
        # archetypal marker genes to append per refinement iteration

    Returns
    -------
    visimu_args : VisiumArgument
        updated parameter set for Starfysh
    """
    # TODO: integrate into `visium_args` class

    n_spots = visium_args.adata.shape[0]
    gene_sig = visium_args.gene_sig.copy()
    anchors_df = visium_args.get_anchors()
    n_top_anchors = int(anchor_threshold*n_spots)

    # Retrieve anchor-archetype mapping scores
    map_df, map_dict = aa_model.assign_archetypes(anchor_df=anchors_df[:n_top_anchors],
                                           r=n_top_anchors) 
    markers_df = aa_model.find_markers(display=False)

    # (1). Update signatures
    for cell_type, archetype in map_dict.items():
        gene_sig = append_sigs(gene_sig=gene_sig,
                               factor=cell_type,
                               sigs=markers_df[archetype],
                               n_genes=n_genes)

    # (2). Update data args.
    visium_args.gene_sig = gene_sig
    visium_args._update_anchors()
    return visium_args


# -------------------
# Post-processing
# -------------------

def extract_feature(adata, key):
    """
    Extract generative / inference output from adata.obsm
    generate dummy tmp. adata for plotting
    """
    assert key in adata.obsm.keys(), "Unfounded Starfysh generative / inference output: {}".format(key)

    if key == 'qc_m':
        cols = adata.uns['cell_types']  # cell type deconvolution
    elif key == 'qz_m':
        cols = ['z'+str(i) for i in range(adata.obsm[key].shape[1])]  # inferred qz (low-dim manifold)
    elif '_inferred_exprs' in key:
        cols = adata.var_names  # inferred cell-type specific expressions
    else:
        cols = ['density']
    adata_dummy = adata.copy()
    adata_dummy.obs = pd.DataFrame(adata.obsm[key], index=adata.obs.index, columns=cols)
    return adata_dummy


def get_reconst_img(args, img_patches):
    """
    Reconst original histology image (H x W) from the given patched image (S x P)
    """
    reconst_img = np.zeros_like(args.img, dtype=np.float64)
    r = args.params['patch_r']
    patch_size = (r * 2, r * 2, 3) if args.img.ndim == 3 else (r * 2, r * 2)
    scale_factor = args.scalefactor['tissue_hires_scalef']
    img_col = args.map_info['imagecol'] * scale_factor
    img_row = args.map_info['imagerow'] * scale_factor

    for i in range(len(img_col)):
        patch_y = slice(int(img_row[i]) - r, int(img_row[i]) + r)
        patch_x = slice(int(img_col[i]) - r, int(img_col[i]) + r)

        sy, sx = reconst_img[patch_y, patch_x].shape[:2]
        img_patch = img_patches[i].reshape(patch_size)
        reconst_img[patch_y, patch_x] = img_patch[:sy, :sx]  # edge patch cases

    return reconst_img

    
